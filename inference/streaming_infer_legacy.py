"""LiveStarPro streaming inference: SVeD + TSHM (Peak-End + Recursive Event Tree
+ retrieval-augmented generation) on the real model.

Mirrors demo.py's proven SVeD perplexity-verification calling convention (the
verified frame is re-shown in `question`, captions live inline in `history`),
but replaces the unbounded cumulative context with a TSHM-managed active window
(stm.clips is the single source of truth) and re-injects retrieved historical
events on decode activation.
"""

import time
import argparse

import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from tshm import (ShortTermMemory, RecursiveEventTree, StreamingKVCache,
                  embed_text)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_IMAGE_TOKEN = 16
TASK_PROMPT = (
    "You are an expert in real-time streaming video description. "
    "I will provide video frames sequentially, and you need to comprehend "
    "each frame's content in real-time while dynamically generating concise "
    "descriptions. Use transitional phrases to maintain textual coherence and "
    "avoid repeating already described content.\n")


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


def find_closest_aspect_ratio(ar, target_ratios, w, h, image_size):
    best_diff, best = float("inf"), (1, 1)
    area = w * h
    for ratio in target_ratios:
        tar = ratio[0] / ratio[1]
        diff = abs(ar - tar)
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    w, h = image.size
    ar = w / h
    target_ratios = sorted(
        set((i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if min_num <= i * j <= max_num),
        key=lambda x: x[0] * x[1])
    tar = find_closest_aspect_ratio(ar, target_ratios, w, h, image_size)
    tw, th = image_size * tar[0], image_size * tar[1]
    blocks = tar[0] * tar[1]
    resized = image.resize((tw, th))
    out = []
    for i in range(blocks):
        box = ((i % (tw // image_size)) * image_size,
               (i // (tw // image_size)) * image_size,
               ((i % (tw // image_size)) + 1) * image_size,
               ((i // (tw // image_size)) + 1) * image_size)
        out.append(resized.crop(box))
    if use_thumbnail and len(out) != 1:
        out.append(image.resize((image_size, image_size)))
    return out


def load_video(video_path, input_size=448, max_num=1, num_segments=32, sample_fps=1):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    transform = build_transform(input_size=input_size)
    interval = max(1, int(fps / sample_fps))
    frame_indices = list(range(0, max_frame + 1, interval))[:num_segments]
    px_per_frame = []
    for idx in frame_indices:
        img = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=input_size,
                                   use_thumbnail=True, max_num=max_num)
        px_per_frame.append(torch.stack([transform(t) for t in tiles]))
    return px_per_frame


class StreamingSession:
    """SVeD gate + TSHM memory over a live frame stream.

    stm.clips is the single source of truth for the active context. history and
    pixel_values are derived from the *captioned* clips each call so that every
    <image> marker has exactly one matching frame (demo.py convention).
    """

    def __init__(self, model, tokenizer, gen_cfg, frames_px, args):
        self.model = model
        self.tok = tokenizer
        self.gen_cfg = gen_cfg
        self.frames_px = frames_px
        self.args = args
        self.stm = ShortTermMemory(l_max=(10**9 if getattr(args, "no_tshm", False) else args.l_max))
        self.tree = RecursiveEventTree(sigma=args.sigma, beta=args.beta,
                                       beam_k=args.beam_k)
        self.kv = StreamingKVCache()
        self.vit_cache = {}
        self.verify_kv = None
        self.verify_ids = None
        self.t_ppl = 0.0
        self.t_gen = 0.0
        self.t_vit = 0.0
        self.recall_by_clip = {}
        self.threshold = None
        self.narration = []
        self.recall_used = []
        self.n_decode = self.n_silent = self.n_retrieval = 0

    # -- helpers ---------------------------------------------------------- #
    def _cap_tokens(self, text):
        return len(self.tok(text).input_ids)

    def _emb(self, text):
        return embed_text(self.model, self.tok, text)

    def _trace(self, msg):
        if self.args.trace:
            print(msg, flush=True)

    def _captioned_clips(self):
        return [c for c in self.stm.clips if c.caption]

    def _committed_gids(self):
        return [f.t for c in self._captioned_clips() for f in c.frames]

    def _recall_text(self, rc):
        if not rc:
            return ""
        if getattr(self.args, "recall_verbatim", False):
            return "[Recall relevant earlier events: " + rc + "] "
        return ("[For continuity only - earlier you already described: " + rc +
                ". Do NOT restate it; describe only what is NEW in the current frame.] ")

    def _derive_history(self):
        hist = []
        for i, c in enumerate(self._captioned_clips()):
            ut = TASK_PROMPT if i == 0 else ""
            ut += self._recall_text(self.recall_by_clip.get(c.clip_id))
            ut += "".join("Frame-%d: <image>\n" % f.t for f in c.frames)
            hist.append((ut, c.caption))
        return hist

    def _pixels(self, gids):
        px = torch.cat([self.frames_px[g] for g in gids])
        return px.to(torch.bfloat16).to(self.model.device), [1] * len(gids)

    # -- model calls (mirror demo.py) ------------------------------------- #
    def _ppl(self, t, check_answer, self_check):
        committed = self._committed_gids()
        gids = committed + ([t] if t not in committed else [])
        px, npatch = self._pixels(gids)
        history = self._derive_history()
        question = "Frame-%d: <image>\n" % t
        total = 0.0
        _t = time.time()
        kv_last = None
        for _ in range(self.args.num_runs):
            ppl, kv_out = self.model.chat(
                self.tok, px, question, self.gen_cfg,
                num_patches_list=list(npatch), history=list(history),
                return_history=False, check_answer=check_answer,
                self_check=self_check, visual_features=self._vis(gids),
                use_kvcache=getattr(self.args, "kv", False),
                past_key_values=self.verify_kv, past_input_ids=self.verify_ids)
            total += ppl
            kv_last = kv_out
        if getattr(self.args, "kv", False) and kv_last is not None:
            self.verify_kv, self.verify_ids = kv_last
        self.t_ppl += time.time() - _t
        return total / self.args.num_runs

    def _generate(self, t, recall):
        committed = self._committed_gids()
        gids = committed + ([t] if t not in committed else [])
        px, npatch = self._pixels(gids)
        history = self._derive_history()
        question = self._recall_text(recall) + "Frame-%d: <image>\n" % t
        if not history:
            question = TASK_PROMPT + question
        _t = time.time()
        caption, _, _ = self.model.chat(
            self.tok, px, question, self.gen_cfg, num_patches_list=list(npatch),
            history=list(history), return_history=True, visual_features=self._vis(gids))
        self.t_gen += time.time() - _t
        return caption

    # -- TSHM maintenance ------------------------------------------------- #
    def _compress(self):
        if self.stm.token_count() <= self.stm.l_max:
            return
        evicted, dropped = self.stm.compress()
        for u in evicted:
            if getattr(self.args, "no_retrieval", False):
                continue
            u.embedding = self._emb(u.caption)
            tag = self.tree.insert(u)
            self._trace("    [compress] evict t=%d-%d '%s' -> %s"
                        % (u.t_start, u.t_end, u.caption[:38], tag))
        if evicted:
            self._trace("    [compress] active=%d tok, tree=%s"
                        % (self.stm.token_count(), self.tree.stats()))
        if dropped:
            self.kv.prune(dropped)

    def _precompute_visual(self):
        if getattr(self.args, "no_viscache", False):
            return
        if self.vit_cache:
            return
        # Causal streaming: encode each frame independently (batch=1) once on
        # arrival, so a frame embedding never leaks future frames and stays
        # fixed thereafter (prerequisite for a valid LLM KV cache).
        _t = time.time()
        for g in range(len(self.frames_px)):
            px = self.frames_px[g].to(torch.bfloat16).to(self.model.device)
            self.vit_cache[g] = self.model.extract_feature(px)
        self.t_vit = time.time() - _t

    def _vis(self, gids):
        if getattr(self.args, "no_viscache", False) or not self.vit_cache:
            return None
        return torch.cat([self.vit_cache[g] for g in gids])

    # -- main loop -------------------------------------------------------- #
    def run(self, num_frames):
        self._precompute_visual()
        t0 = time.time()
        n = min(num_frames, len(self.frames_px))
        for t in range(n):
            self._init_frame() if t == 0 else self._step_frame(t)
            self._compress()
        dt = time.time() - t0
        return dict(frames=n, seconds=round(dt, 2), fps=round(n / dt, 3),
                    decode=self.n_decode, silent=self.n_silent,
                    retrieval=self.n_retrieval, tree=self.tree.stats(),
                    active_clips=len(self.stm.clips),
                    active_tokens=self.stm.token_count(),
                    t_ppl=round(self.t_ppl, 2), t_gen=round(self.t_gen, 2),
                    t_vit=round(self.t_vit, 2))

    def _init_frame(self):
        self.stm.start_clip()
        self.stm.add_frame(t=0, score=0.0, n_tokens=NUM_IMAGE_TOKEN)
        caption = self._generate(0, None)
        self.stm.clips[-1].caption = caption
        ref = self._ppl(0, check_answer=caption, self_check=True)
        self.stm.clips[-1].frames[0].score = ref
        self.threshold = ref
        self.narration.append((0, caption))
        self.n_decode += 1
        self._trace("Frame-0 [DECODE] ppl_ref=%.3f | %s" % (ref, caption[:80]))

    def _step_frame(self, t):
        cur = self.stm.clips[-1]
        active_caption = cur.caption
        ppl = self._ppl(t, check_answer=active_caption, self_check=False)

        if ppl > self.threshold * self.args.alpha:
            self.stm.finalize_clip(cur.caption, self._cap_tokens(cur.caption),
                                   self._emb(cur.caption))
            recall = None
            if len(self.tree) > 0 and not getattr(self.args, "no_retrieval", False):
                hits, n_eval = self.tree.retrieve(self._emb(cur.caption),
                                                  k=self.args.beam_k)
                gap = getattr(self.args, "recall_min_gap", 0)
                if gap > 0:
                    hits = [h for h in hits if (t - h.t_end) >= gap]
                caps = [h.caption for h in hits][: self.args.max_recall]
                if caps:
                    recall = " | ".join(caps)
                    self.n_retrieval += 1
                    self._trace("    [retrieve] %d sims / %d nodes -> %s"
                                % (n_eval, len(self.tree),
                                   " || ".join(c[:26] for c in caps)))
            caption = self._generate(t, recall)
            self.stm.start_clip()
            self.stm.add_frame(t=t, score=0.0, n_tokens=NUM_IMAGE_TOKEN)
            self.stm.clips[-1].caption = caption
            if recall:
                self.recall_by_clip[self.stm.clips[-1].clip_id] = recall
            ref = self._ppl(t, check_answer=caption, self_check=True)
            self.stm.clips[-1].frames[-1].score = ref
            self.threshold = ref
            self.narration.append((t, caption))
            self.recall_used.append(recall)
            self.n_decode += 1
            self._trace("Frame-%d [DECODE] ppl=%.3f (>%.3f*%.2f) | %s"
                        % (t, ppl, self.threshold, self.args.alpha, caption[:80]))
        else:
            self.stm.add_frame(t=t, score=ppl, n_tokens=NUM_IMAGE_TOKEN)
            self.n_silent += 1
            self._trace("Frame-%d [silent] ppl=%.3f <= %.3f*%.2f"
                        % (t, ppl, self.threshold, self.args.alpha))


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="./")
    p.add_argument("--video", default="../assets/videos/HPtIGhOsViM.mp4")
    p.add_argument("--num-segments", type=int, default=64)
    p.add_argument("--num-frames", type=int, default=0)
    p.add_argument("--sample-fps", type=float, default=1.0)
    p.add_argument("--max-num", type=int, default=1)
    p.add_argument("--l-max", type=int, default=160)
    p.add_argument("--alpha", type=float, default=1.06)
    p.add_argument("--sigma", type=float, default=0.75)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--beam-k", type=int, default=3)
    p.add_argument("--max-recall", type=int, default=2)
    p.add_argument("--num-runs", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--no-tshm", action="store_true")
    p.add_argument("--no-retrieval", action="store_true")
    p.add_argument("--no-viscache", action="store_true")
    p.add_argument("--kv", action="store_true")
    p.add_argument("--recall-verbatim", action="store_true")
    p.add_argument("--recall-min-gap", type=int, default=0)
    p.add_argument("--pro", action="store_true",
                   help="LiveStarPro: long-term Recursive Event Tree + gated retrieval + KV cache")
    return p.parse_args()


def main():
    args = build_args()
    if args.pro:
        args.no_retrieval = False
        args.kv = True
        if args.recall_min_gap == 0:
            args.recall_min_gap = 8
    else:
        args.no_retrieval = True
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = (AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
             .half().cuda().to(torch.bfloat16).eval())
    gen_cfg = dict(temperature=0.0, max_new_tokens=args.max_new_tokens,
                   top_p=0.1, num_beams=1, repetition_penalty=1.05)
    frames_px = load_video(args.video, max_num=args.max_num,
                           num_segments=args.num_segments,
                           sample_fps=args.sample_fps)
    n = args.num_frames or len(frames_px)
    mode = "LiveStarPro" if args.pro else "LiveStar (base)"
    print("== %s | %d frames | L_max=%d alpha=%.2f | retrieval=%s gap=%d kv=%s ==" %
          (mode, n, args.l_max, args.alpha, (not args.no_retrieval),
           args.recall_min_gap, args.kv))
    with torch.no_grad():
        sess = StreamingSession(model, tok, gen_cfg, frames_px, args)
        summary = sess.run(n)
    print("\n== SUMMARY ==")
    for k, v in summary.items():
        print("  %-14s %s" % (k, v))
    print("\n== NARRATION (spoken moments) ==")
    for t, cap in sess.narration:
        print("  t=%-3d %s" % (t, cap))


if __name__ == "__main__":
    main()
