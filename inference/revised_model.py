"""Exact prompt accounting and multimodal verification for revised inference.

Uses the bundled conversation template and language model. The legacy chat()
path is untouched. A KV prefix is reusable only when BOTH token IDs and visual
frame identities match; answer tokens are always rescored in full.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from conversation import get_conv_template

TASK_PROMPT = (
    'You are an expert in real-time streaming video description. '
    'I will provide video frames sequentially, and you need to comprehend '
    "each frame's content in real-time while dynamically generating concise "
    'descriptions. Use transitional phrases to maintain textual coherence and '
    'avoid repeating already described content.\n')
MEMORY_HEADER = ('Earlier retrieved evidence (historical, not the current scene; '
                 'use only when relevant and do not repeat it as a new event):\n')


@dataclass(eq=False)
class VisualFrame:
    frame_id: int
    timestamp: float
    features: torch.Tensor  # [1, num_image_token, hidden], CPU
    image: object = None   # PIL image, released once no longer needed for indexing
    score: float = 0.0


@dataclass
class Prompt:
    ids: list
    frames: list
    answer_start: int = 0
    answer_end: int = 0


def common_prefix(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


class ModelAdapter:
    def __init__(self, model, tokenizer, use_kv=True):
        self.model, self.tok, self.use_kv = model, tokenizer, use_kv
        self.image_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.model.img_context_token_id = self.image_token_id
        self.num_image_token = model.num_image_token
        self.cache, self.cache_signature = None, []
        self.last_reused_tokens = 0

    def ids(self, text, special=True):
        return self.tok(text, add_special_tokens=special).input_ids

    def caption(self, text, limit=96):
        ids = self.ids(text, special=False)
        result = self.tok.decode(ids[:limit], skip_special_tokens=True)
        # Some tokenizers do not round-trip a truncated terminal token exactly.
        while len(self.ids(result, special=False)) > limit:
            ids = ids[:-1]
            result = self.tok.decode(ids[:limit], skip_special_tokens=True)
        return result

    def image_markup(self, frame):
        if frame.features.shape[:2] != (1, self.num_image_token):
            raise ValueError('Expected one static-resolution frame with model.num_image_token features')
        return ('<img>' + '<IMG_CONTEXT>' * self.num_image_token + '</img>')

    def frame_text(self, frame):
        return 'Frame-%d at %.3fs: %s\n' % (
            frame.frame_id, frame.timestamp, self.image_markup(frame))

    def memory_text(self, events):
        if not events:
            return ''
        blocks = [MEMORY_HEADER]
        for event in events:
            blocks.append('[Event %d, %.3f–%.3fs]\n%sCaption: %s\n[/Event]\n' % (
                event.node_id, event.t_start, event.t_end,
                self.frame_text(event.visual), self.caption(event.caption)))
        return ''.join(blocks)

    def render(self, clips, current, events=(), question=None, answer=None):
        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        frames = []
        for i, clip in enumerate(clips):
            text = (TASK_PROMPT if i == 0 else '')
            text += ''.join(self.frame_text(frame) for frame in clip.frames)
            frames.extend(clip.frames)
            template.append_message(template.roles[0], text)
            template.append_message(template.roles[1], clip.caption)
        text = TASK_PROMPT if not clips else ''
        text += self.memory_text(events)
        frames.extend(event.visual for event in events)
        text += self.frame_text(current)
        frames.append(current)
        text += ('Answer the question using only observations up to %.3fs: %s' %
                 (current.timestamp, question)) if question is not None else ''
        template.append_message(template.roles[0], text)
        template.append_message(template.roles[1], None)
        prefix = self.ids(template.get_prompt())
        if answer is None:
            return Prompt(prefix, frames)
        template.messages[-1][1] = answer
        full = self.ids(template.get_prompt())
        # Locate the answer independently of hard-coded role token offsets.
        start = common_prefix(prefix, full)
        suffix = self.ids(template.sep, special=False)
        end = len(full) - len(suffix) if suffix and full[-len(suffix):] == suffix else len(full)
        return Prompt(full, frames, start, end)

    def memory_tokens(self, events):
        return len(self.ids(self.memory_text(events), special=False))

    def _inputs(self, prompt):
        ids = torch.tensor([prompt.ids], device=self.model.device, dtype=torch.long)
        embeddings = self.model.language_model.get_input_embeddings()(ids).clone()
        features = torch.cat([frame.features for frame in prompt.frames], dim=0)
        selected = ids[0] == self.image_token_id
        if int(selected.sum()) != features.shape[0] * features.shape[1]:
            raise ValueError('Visual placeholders and cached frame features disagree')
        embeddings[0, selected] = features.reshape(-1, features.shape[-1]).to(embeddings)
        identities = iter(frame.frame_id for frame in prompt.frames
                          for _ in range(self.num_image_token))
        signature = [(token, next(identities) if token == self.image_token_id else None)
                     for token in prompt.ids]
        return ids, embeddings, signature

    @torch.inference_mode()
    def verify(self, prompt):
        if prompt.answer_end <= prompt.answer_start:
            raise ValueError('Cannot verify an empty answer')
        ids, embeddings, signature = self._inputs(prompt)
        reuse = 0
        cache = None
        if self.use_kv and self.cache is not None:
            # Keep the logit preceding the first answer token in the fresh suffix.
            reuse = min(common_prefix(signature, self.cache_signature),
                        self.cache[0][0].shape[2], max(0, prompt.answer_start - 1))
            if reuse:
                cache = tuple((k[:, :, :reuse], v[:, :, :reuse]) for k, v in self.cache)
        self.last_reused_tokens = reuse
        output = self.model.language_model(
            inputs_embeds=embeddings[:, reuse:],
            attention_mask=torch.ones_like(ids),  # includes the cached prefix
            position_ids=torch.arange(reuse, len(prompt.ids), device=ids.device)[None],
            past_key_values=cache, use_cache=self.use_kv, return_dict=True)
        start, end = max(1, prompt.answer_start), prompt.answer_end
        logits = output.logits[:, start - 1 - reuse:end - 1 - reuse].float()
        targets = ids[:, start:end]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        if self.use_kv:
            self.cache, self.cache_signature = output.past_key_values, signature
        return math.exp(min(float(loss), 80.0))

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=128):
        ids, embeddings, _ = self._inputs(prompt)
        template = get_conv_template(self.model.template)
        eos = self.tok.convert_tokens_to_ids(template.sep.strip())
        output = self.model.language_model.generate(
            inputs_embeds=embeddings, attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            repetition_penalty=1.05, eos_token_id=eos, use_cache=True,
            return_dict_in_generate=True)
        # The bundled InternLM returns (sequences, KV); standard HF returns an object.
        sequences = output[0] if isinstance(output, tuple) else getattr(output, 'sequences', output)
        return self.tok.batch_decode(sequences, skip_special_tokens=True)[0].split(
            template.sep.strip())[0].strip()
