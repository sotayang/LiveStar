"""Tree-Structured Hierarchical Memory (TSHM) for LiveStarPro."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def l2norm(x, dim=-1):
    return F.normalize(x, p=2, dim=dim)


@torch.no_grad()
def embed_text(model, tokenizer, text):
    """Caption embedding = mean-pooled LLM input embeddings (no forward pass)."""
    if not text:
        text = " "
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    emb = model.language_model.get_input_embeddings()(ids)
    vec = emb.mean(dim=1).squeeze(0).float()
    return l2norm(vec).cpu()


@dataclass
class Frame:
    t: int
    score: float
    n_tokens: int
    pos_start: int = -1
    pos_end: int = -1


@dataclass
class Clip:
    clip_id: int
    frames: List[Frame] = field(default_factory=list)
    caption: str = ""
    caption_tokens: int = 0
    caption_pos_start: int = -1
    caption_pos_end: int = -1
    summary_emb: Optional[torch.Tensor] = None
    t_start: int = -1
    t_end: int = -1

    def token_count(self):
        return sum(f.n_tokens for f in self.frames) + self.caption_tokens


@dataclass
class MemoryUnit:
    caption: str
    embedding: torch.Tensor
    visual_tokens: Optional[torch.Tensor] = None
    children: List["MemoryUnit"] = field(default_factory=list)
    t_start: int = -1
    t_end: int = -1
    depth: int = 0


class ShortTermMemory:
    """Active context window with Peak-End dynamic pruning."""

    def __init__(self, l_max, min_frames_per_clip=1):
        self.l_max = l_max
        self.min_frames_per_clip = min_frames_per_clip
        self.clips = []
        self._next_clip_id = 0
        self._cur = None

    def start_clip(self):
        self._cur = Clip(clip_id=self._next_clip_id)
        self._next_clip_id += 1
        self.clips.append(self._cur)
        return self._cur

    def add_frame(self, t, score, n_tokens, pos_start=-1, pos_end=-1):
        if self._cur is None:
            self.start_clip()
        fr = Frame(t=t, score=score, n_tokens=n_tokens,
                   pos_start=pos_start, pos_end=pos_end)
        self._cur.frames.append(fr)
        self._cur.t_start = self._cur.frames[0].t
        self._cur.t_end = t

    def finalize_clip(self, caption, caption_tokens, summary_emb=None,
                      caption_pos=(-1, -1)):
        if self._cur is None:
            return
        self._cur.caption = caption
        self._cur.caption_tokens = caption_tokens
        self._cur.summary_emb = summary_emb
        self._cur.caption_pos_start, self._cur.caption_pos_end = caption_pos

    def token_count(self):
        return sum(c.token_count() for c in self.clips)

    def compress(self):
        evicted = []
        dropped = []
        for clip in self.clips:
            if self.token_count() <= self.l_max:
                break
            self._peak_end_prune_clip(clip, dropped)
        while self.token_count() > self.l_max and len(self.clips) > 1:
            oldest = self.clips[0]
            if oldest is self._cur:
                break
            self.clips.pop(0)
            evicted.append(self._clip_to_unit(oldest))
            for fr in oldest.frames:
                if fr.pos_start >= 0:
                    dropped.extend(range(fr.pos_start, fr.pos_end))
            if oldest.caption_pos_start >= 0:
                dropped.extend(range(oldest.caption_pos_start,
                                     oldest.caption_pos_end))
        return evicted, sorted(set(dropped))

    def _peak_end_prune_clip(self, clip, dropped):
        if len(clip.frames) <= self.min_frames_per_clip:
            return
        scores = sorted(f.score for f in clip.frames)
        mid = len(scores) // 2
        median = scores[mid] if len(scores) % 2 else 0.5 * (scores[mid - 1] + scores[mid])
        keep, drop = [], []
        for fr in clip.frames:
            (keep if fr.score <= median else drop).append(fr)
        if not keep:
            keep = [min(clip.frames, key=lambda f: f.score)]
            drop = [f for f in clip.frames if f is not keep[0]]
        for fr in drop:
            if fr.pos_start >= 0:
                dropped.extend(range(fr.pos_start, fr.pos_end))
        clip.frames = keep

    def _clip_to_unit(self, clip):
        return MemoryUnit(
            caption=clip.caption,
            embedding=(clip.summary_emb if clip.summary_emb is not None
                       else torch.zeros(1)),
            visual_tokens=None,
            t_start=clip.t_start,
            t_end=clip.t_end,
        )


class RecursiveEventTree:
    """Recursive event tree with momentum parents + beam-descent retrieval."""

    def __init__(self, sigma=0.75, beta=0.3, beam_k=3):
        self.sigma = sigma
        self.beta = beta
        self.beam_k = beam_k
        self.roots = []
        self._size = 0

    def __len__(self):
        return self._size

    def _all_nodes(self):
        out, stack = [], list(self.roots)
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children)
        return out

    @staticmethod
    def _cos(a, b):
        return float(torch.dot(a, b))

    def insert(self, unit):
        self._size += 1
        nodes = self._all_nodes()
        if not nodes:
            unit.depth = 0
            self.roots.append(unit)
            return "root#0 (empty tree)"
        best, best_sim = None, -1.0
        for n in nodes:
            s = self._cos(unit.embedding, n.embedding)
            if s > best_sim:
                best, best_sim = n, s
        if best_sim >= self.sigma:
            unit.depth = best.depth + 1
            best.children.append(unit)
            self._momentum_update(best, unit.embedding)
            return "child of [%s] (sim=%.3f>=sigma)" % (best.caption[:24], best_sim)
        unit.depth = 0
        self.roots.append(unit)
        return "new root (best sim=%.3f<sigma)" % best_sim

    def _momentum_update(self, parent, child_emb):
        parent.embedding = l2norm(
            (1 - self.beta) * parent.embedding + self.beta * child_emb, dim=0)

    def retrieve(self, query, k=None):
        if not self.roots:
            return [], 0
        k = k or self.beam_k
        n_eval = 0
        frontier = list(self.roots)
        n_eval += len(frontier)
        frontier = self._topk(query, frontier, k)
        hits = list(frontier)
        while any(n.children for n in frontier):
            candidates = []
            for n in frontier:
                candidates.extend(n.children)
            if not candidates:
                break
            n_eval += len(candidates)
            frontier = self._topk(query, candidates, k)
            hits.extend(frontier)
        expanded = {id(h): h for h in hits}
        parent_of = self._parent_map()
        for h in hits:
            p = parent_of.get(id(h))
            if p is not None:
                expanded[id(p)] = p
            for c in h.children:
                expanded[id(c)] = c
        return list(expanded.values()), n_eval

    def _topk(self, query, nodes, k):
        scored = sorted(nodes, key=lambda n: self._cos(query, n.embedding),
                        reverse=True)
        return scored[:k]

    def _parent_map(self):
        pm = {}
        stack = [(None, r) for r in self.roots]
        while stack:
            par, n = stack.pop()
            pm[id(n)] = par
            for c in n.children:
                stack.append((n, c))
        return pm

    def stats(self):
        nodes = self._all_nodes()
        if not nodes:
            return {"size": 0, "roots": 0, "branching_factor": 0.0, "height": 0}
        internal = [n for n in nodes if n.children]
        bf = (sum(len(n.children) for n in internal) / len(internal)
              if internal else 0.0)
        height = max(self._height(r) for r in self.roots)
        return {"size": len(nodes), "roots": len(self.roots),
                "branching_factor": round(bf, 2), "height": height}

    def _height(self, node):
        if not node.children:
            return 1
        return 1 + max(self._height(c) for c in node.children)


class StreamingKVCache:
    """append/prune wrapper over HF past_key_values (tuple of (k,v), [B,H,S,D])."""

    def __init__(self):
        self.past_key_values = None

    @property
    def seq_len(self):
        if self.past_key_values is None:
            return 0
        return self.past_key_values[0][0].shape[2]

    def set(self, pkv):
        self.past_key_values = pkv

    def prune(self, dropped_positions):
        if self.past_key_values is None or not dropped_positions:
            return
        total = self.seq_len
        drop = set(p for p in dropped_positions if 0 <= p < total)
        if not drop:
            return
        keep = torch.tensor([i for i in range(total) if i not in drop],
                            dtype=torch.long,
                            device=self.past_key_values[0][0].device)
        new_pkv = []
        for k, v in self.past_key_values:
            new_pkv.append((k.index_select(2, keep), v.index_select(2, keep)))
        self.past_key_values = tuple(new_pkv)


def _selftest():
    torch.manual_seed(0)
    print("== TSHM self-test (CPU, dummy embeddings) ==")
    dim = 32
    themes = [l2norm(torch.randn(dim), dim=0) for _ in range(3)]
    stm = ShortTermMemory(l_max=200)
    tree = RecursiveEventTree(sigma=0.75, beta=0.3, beam_k=2)
    pos = 0
    all_evicted = []
    for cid in range(6):
        stm.start_clip()
        theme = themes[cid % 3]
        for fi in range(8):
            score = float(torch.rand(1))
            n_tok = 16
            stm.add_frame(t=cid * 8 + fi, score=score, n_tokens=n_tok,
                          pos_start=pos, pos_end=pos + n_tok)
            pos += n_tok
        emb = l2norm(theme + 0.15 * torch.randn(dim), dim=0)
        stm.finalize_clip(caption="event-%d (theme %d)" % (cid, cid % 3),
                          caption_tokens=10, summary_emb=emb,
                          caption_pos=(pos, pos + 10))
        pos += 10
        evicted, dropped = stm.compress()
        for u in evicted:
            tag = tree.insert(u)
            print("  evict clip -> %-24s | %s" % (u.caption, tag))
        all_evicted += evicted
    print("active tokens after stream: %d (L_max=200)" % stm.token_count())
    print("clips still active: %s" % [c.clip_id for c in stm.clips])
    print("tree stats: %s" % tree.stats())
    q = l2norm(themes[0] + 0.1 * torch.randn(dim), dim=0)
    hits, n_eval = tree.retrieve(q, k=2)
    print("retrieval (query~theme0): %d sims evaluated, %d units w/ path:"
          % (n_eval, len(hits)))
    for h in hits:
        print("    - %s (depth %d)" % (h.caption, h.depth))
    assert stm.token_count() <= 200 or len(stm.clips) == 1
    assert len(tree) == len(all_evicted)
    assert n_eval <= len(tree) + tree.beam_k * 4
    print("== self-test passed ==")


if __name__ == "__main__":
    _selftest()
