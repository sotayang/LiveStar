# ===================================================================
# LiveStar - Live Streaming Assistant 
# for Real-World Online Video Understanding
# ===================================================================
# Modified from: InternVL (Original Copyright (c) 2024 OpenGVLab)
# Licensed under The MIT License [see LICENSE for details]
# ===================================================================

import re
import warnings
import os
from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_livestar_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_livestar_vit import InternVisionModel, has_flash_attn
from .modeling_livestar_llm import InternLM2ForCausalLM

import time

logger = logging.get_logger(__name__)



def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int, 
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).
    """
    protected = 0

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    assert r > 0, r

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src) # , reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.36.2', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.local_num_frames = 1
        self.num_tome_tokens = 16
        self.config = config
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        # self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2)) //
        self.num_image_token = self.num_tome_tokens // self.local_num_frames
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    def merge_tokens(self, x, target_num_token):
        r"""
        x = torch.randn(10, 2560, c)
        x = merge_tokens(x, r_merge_list=[1280])
        """
        size = None
        b, p, c = x.shape
        tmp_p = p
        r_merge_list = []
        assert tmp_p > target_num_token, f"{tmp_p} should greater than {target_num_token}"
        while tmp_p != target_num_token:
            if tmp_p - target_num_token <= (tmp_p // 2):
                r_merge_list.append(tmp_p - target_num_token)
                break
            else:
                r_merge_list.append(tmp_p // 2)
                tmp_p = tmp_p - (tmp_p // 2)
                
        
        head = self.config.llm_config.num_attention_heads

        dim = c // head
        for r in r_merge_list:
            metric = x.reshape(b, p, head, dim).mean(2) # [b, p, c//head]
            merge, _ = bipartite_soft_matching(
                metric, 
                r
            )
            x, size = merge_wavg(merge, x, size)
            _, p, _ = x.shape
        # x = x.reshape(-1, c)  # 300, 1024
        return x
    
    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def streaming_evaluate(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        mask = labels != -100
        padded_mask = torch.cat([torch.full((1, 1), False, device=mask.device), mask], dim=1)
        start_indices = (padded_mask[:, 1:] & ~padded_mask[:, :-1]).nonzero()[:, 1]
        padded_mask_end = torch.cat([mask, torch.full((1, 1), False, device=mask.device)], dim=1)
        end_indices = (padded_mask_end[:, :-1] & ~padded_mask_end[:, 1:]).nonzero()[:, 1]         
        
        position_ids = attention_mask.long().cumsum(-1) - 1
        mask_zero = attention_mask == 0
        batch_size, seq_len = mask_zero.shape
        offset = torch.zeros((batch_size, seq_len), dtype=torch.long, device=position_ids.device)
        current = torch.zeros(batch_size, dtype=torch.long, device=position_ids.device)
        for i in range(1, seq_len):
            current = torch.where(mask_zero[:, i], current + 1, 0)
            offset[:, i] = current
        position_ids += offset * mask_zero.long()
        
        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits
        
        lm_ppl_list, lm_correctness_list = [], []

        assert len(start_indices) == len(start_indices)
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            current_logits = logits[:, start-1:end+1, :]
            current_labels = labels[:, start-1:end+1]
            shift_logits = current_logits[:, :-1, :].contiguous()
            shift_labels = current_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            ppl = torch.exp(loss)
            lm_ppl_list.append(ppl)
                           
            max_indices = torch.argmax(shift_logits, dim=-1)
            num_correct = (max_indices == shift_labels).sum()
            lm_correctness_list.append(num_correct / shift_labels.numel())
                
            max_indices = torch.argmax(shift_logits, dim=-1)
            num_correct = (max_indices == shift_labels).sum()
            accuracy = num_correct.float() / shift_labels.numel()
            lm_correctness_list.append(accuracy)

        lm_ppl = torch.stack(lm_ppl_list).mean() if lm_ppl_list else None
        lm_correctness = torch.stack(lm_correctness_list).float().mean() if lm_correctness_list else None
        return lm_ppl, lm_correctness


    def streaming_infer(
            self,
            pixel_values: torch.FloatTensor,
            visual_features: Optional[torch.FloatTensor] = None,
            use_kvcache: bool = False,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = visual_features if visual_features is not None else self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        if past_key_values is not None and use_kvcache:
            past_len = past_key_values[0][0].shape[2]
            input_embeds = input_embeds[:, past_len:, :]
            if labels is not None:
                labels = labels[:, past_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, past_len:]

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=(use_cache or use_kvcache),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0] // self.local_num_frames, -1, vit_embeds.shape[-1])
        vit_embeds = self.merge_tokens(vit_embeds, self.num_tome_tokens)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0] * self.local_num_frames, -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, past_key_values=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False, check_answer=None, self_check=False, eval=False, conversations=None, use_kvcache=False, visual_features=None, past_input_ids=None):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question
        
        if self_check == True:
            last_pixel_frame = pixel_values[-1:]
            pixel_values = torch.cat((pixel_values, last_pixel_frame), dim=0)
            last_patch_frame = num_patches_list[-1]
            num_patches_list.append(last_patch_frame)
            if visual_features is not None:
                visual_features = torch.cat((visual_features, visual_features[-1:]), dim=0)

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        if eval:
            for human, gpt in zip(conversations[::2], conversations[1::2]):
                template.append_message(template.roles[0], human["value"])
                template.append_message(template.roles[1], gpt["value"])
                query = template.get_prompt()
                query_prefix = template.get_prompt()
                
        else:
            history = [] if history is None else history
            for (old_question, old_answer) in history:
                template.append_message(template.roles[0], old_question)
                template.append_message(template.roles[1], old_answer)
            template.append_message(template.roles[0], question)
            query_prefix = template.get_prompt()
            template.append_message(template.roles[1], check_answer)
            query = template.get_prompt()
            
        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')
            
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            img_index = query.rfind('<image>')
            query = query[:img_index] + image_tokens + query[img_index + len('<image>'):]
            img_index = query_prefix.rfind('<image>')
            query_prefix = query_prefix[:img_index] + image_tokens + query_prefix[img_index + len('<image>'):]

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
    
        if eval:
            labels = torch.full_like(input_ids, -100)
            for i, seq in enumerate(input_ids):
                start_indices, end_indices, stack = [], [], []
                for idx, token in enumerate(seq):
                    if token == 92544 and stack:
                        start_indices.append(idx)
                        end_indices.append(stack.pop())
                    elif token == 92545 and idx + 2 < len(seq) and seq[idx + 2] == 92542:
                        stack.append(idx + 1)

                last_end_idx = stack.pop() if stack else len(seq)
                last_start_idx = len(seq)

                for s, e in zip(start_indices, end_indices):
                    labels[i][e+7:min(s-9, len(seq))] = seq[e+7:min(s-9, len(seq))]

                labels[i][last_end_idx+7:min(last_start_idx-1, len(seq))] = seq[last_end_idx+7:min(last_start_idx-1, len(seq))]

            lm_ppl, lm_correctness = self.streaming_evaluate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=torch.tensor(num_patches_list),
                labels=labels,
            )
            return lm_ppl, lm_correctness
        
        if check_answer is not None:
            
            input_ids_prefix = tokenizer(query_prefix)['input_ids']
            answer_len = input_ids.shape[1] - len(input_ids_prefix)
            labels = torch.full_like(input_ids, -100)
            labels[:, -answer_len+4:-1] = input_ids[:, -answer_len+4:-1]
            kv_in = None
            if use_kvcache and past_key_values is not None and past_input_ids is not None:
                a, b = past_input_ids[0], input_ids[0]
                mn = min(a.shape[0], b.shape[0], past_key_values[0][0].shape[2])
                P = 0
                while P < mn and int(a[P]) == int(b[P]):
                    P += 1
                P = min(P, input_ids.shape[1] - 1)
                if P > 0:
                    kv_in = tuple((k[:, :, :P, :], v[:, :, :P, :]) for k, v in past_key_values)
            check_output = self.streaming_infer(
                pixel_values=pixel_values,
                visual_features=visual_features,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=torch.tensor(num_patches_list),
                labels=labels,
                past_key_values=kv_in,
                use_kvcache=use_kvcache,
            )
            if use_kvcache:
                return torch.exp(check_output.loss).item(), (check_output.past_key_values, input_ids)
            return torch.exp(check_output.loss).item(), None
        
        else:
            generation_output, new_past_key_values = self.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_kvcache=use_kvcache,
                visual_features=visual_features,
                **generation_config
            )
            response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
            response = response.split(template.sep.strip())[0].strip()
            history.append((question, response))
            if return_history:
                return response, history, new_past_key_values
            else:
                query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
                query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
                if verbose:
                    print(query_to_print, response)
                return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            past_key_values=None,
            use_kvcache=False,
            use_cache=True,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            
        if past_key_values is not None and use_kvcache:
            past_key_len = past_key_values[0][0].shape[2]  # [B, num_heads, past_len, head_dim]
            input_embeds = input_embeds[:, past_key_len:, :]
            attention_mask = attention_mask[:, past_key_len:] if attention_mask is not None else None
        
        outputs, new_past_key_values = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            use_kvcache=use_kvcache,
            return_dict_in_generate=True,
            **generate_kwargs,
        )
        if use_kvcache:
            return outputs, new_past_key_values
        else:
            return outputs, None
