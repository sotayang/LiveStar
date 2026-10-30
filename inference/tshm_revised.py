"""Revision TSHM: semantic indexing, ancestor updates and mixed-depth leaf search.

The July implementation remains in tshm.py. This module does not alter it.
External event storage grows with stream length; only injected context is capped.
"""
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F


def normalized(vector):
    vector = vector.detach().float().cpu().flatten()
    if not torch.isfinite(vector).all() or vector.norm() <= 1e-12:
        raise ValueError('Embedding must be finite and nonzero')
    return F.normalize(vector, dim=0)


class OpenCLIPIndex:
    """Frozen 768-d joint space. CLIP and generator tokenizers are independent."""
    model_name = 'ViT-L-14'
    pretrained = 'laion2b_s32b_b82k'

    def __init__(self, device='cuda'):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=device)
        self.model.eval().requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(self.model_name, context_length=77)

    @torch.inference_mode()
    def text(self, caption):
        tokens = self.tokenizer([caption]).to(self.device)
        return self._checked(self.model.encode_text(tokens)[0])

    @torch.inference_mode()
    def image(self, frame):
        pixels = self.preprocess(frame).unsqueeze(0).to(self.device)
        return self._checked(self.model.encode_image(pixels)[0])

    @staticmethod
    def _checked(vector):
        result = normalized(vector)
        if result.numel() != 768:
            raise ValueError('Expected 768-dimensional OpenCLIP embeddings')
        return result

    def event(self, caption, frame):
        image = self.image(frame)
        return normalized(self.text(caption) + image), image

    @staticmethod
    def implicit_query(peak_embeddings):
        peaks = list(peak_embeddings)[-4:]
        if not peaks:
            raise ValueError('An implicit query needs at least one observed peak frame')
        return normalized(torch.stack([normalized(p) for p in peaks]).mean(0))


@dataclass(eq=False)
class Event:
    caption: str
    original_embedding: torch.Tensor
    visual: Any  # CPU generator features and source timestamp, NOT CLIP features.
    t_start: float
    t_end: float
    node_id: int = -1
    parent: Optional['Event'] = field(default=None, repr=False)
    children: list = field(default_factory=list, repr=False)
    embedding: torch.Tensor = field(init=False, repr=False)
    depth: int = 0

    def __post_init__(self):
        self.original_embedding = normalized(self.original_embedding).clone()
        self.embedding = self.original_embedding.clone()


@dataclass
class Hit:
    event: Event
    score: float


class RecursiveEventTree:
    def __init__(self, sigma=0.75, beta=0.3, beam_k=4):
        if not -1 <= sigma <= 1 or not 0 <= beta <= 1 or beam_k < 1:
            raise ValueError('Invalid tree parameters')
        self.sigma, self.beta, self.beam_k = sigma, beta, beam_k
        self.nodes, self.roots = [], []

    def __len__(self):
        return len(self.nodes)

    def insert(self, event):
        if event.node_id != -1 or event.parent is not None or event.children:
            raise ValueError('Only fresh events may be inserted')
        # nodes are stored in creation order: the first maximum breaks ties.
        best, best_score = None, float('-inf')
        for node in self.nodes:
            score = float(torch.dot(node.embedding, event.original_embedding))
            if score > best_score:
                best, best_score = node, score
        event.node_id = len(self.nodes)
        if best is not None and best_score >= self.sigma:
            event.parent, event.depth = best, best.depth + 1
            best.children.append(event)
            ancestor = best
            while ancestor is not None:
                ancestor.embedding = normalized(
                    (1 - self.beta) * ancestor.embedding
                    + self.beta * event.original_embedding)
                ancestor = ancestor.parent
        else:
            self.roots.append(event)
        self.nodes.append(event)
        return event

    def retrieve(self, query, k=None):
        query, k = normalized(query), self.beam_k if k is None else k
        if k < 1:
            raise ValueError('Beam width must be positive')
        completed, active, evaluated = [], [], 0
        candidates = self.roots
        while candidates:
            internal = []
            for node in candidates:
                hit = Hit(node, float(torch.dot(query, node.embedding)))
                evaluated += 1
                (internal if node.children else completed).append(hit)
            internal.sort(key=lambda hit: (-hit.score, hit.event.node_id))
            active = internal[:k]
            candidates = [child for hit in active for child in hit.event.children]
        completed.sort(key=lambda hit: (-hit.score, hit.event.node_id))
        return completed[:k], evaluated

    def stats(self):
        internal = [node for node in self.nodes if node.children]
        return dict(size=len(self.nodes), roots=len(self.roots),
                    height=max((node.depth + 1 for node in self.nodes), default=0),
                    branching_factor=(sum(len(n.children) for n in internal) / len(internal)
                                      if internal else 0.0))


def expansion_order(hits):
    """All ranked leaves first, then ancestors by distance/leaf score/creation ID.

    Parent pointers avoid scanning the complete stored tree at retrieval time.
    """
    seen = set()
    for hit in hits:
        if hit.event.node_id not in seen:
            seen.add(hit.event.node_id)
            yield hit.event
    frontier = [(hit.event.parent, hit.score) for hit in hits if hit.event.parent]
    while frontier:
        frontier.sort(key=lambda item: (-item[1], item[0].node_id))
        next_frontier = []
        for node, score in frontier:
            if node.node_id not in seen:
                seen.add(node.node_id)
                yield node
            if node.parent is not None:
                next_frontier.append((node.parent, score))
        frontier = next_frontier


def pack_events(hits, fits, max_events=8):
    """fits(candidate_events) measures actual rendered generator tokens.

    Skip an entire event that cannot fit, then try later smaller events.
    """
    selected = []
    for event in expansion_order(hits):
        if len(selected) == max_events:
            break
        if fits(selected + [event]):
            selected.append(event)
    return selected
