import math
import torch
import pytest

from tshm_revised import Event, Hit, RecursiveEventTree, OpenCLIPIndex, expansion_order, normalized, pack_events


def event(vector, caption='event'):
    return Event(caption, torch.tensor(vector, dtype=torch.float32), None, 0, 1)


def test_insertion_threshold_ties_and_original_ancestor_updates():
    tree = RecursiveEventTree(sigma=0.75, beta=0.3)
    root = tree.insert(event([1., 0.]))
    second = tree.insert(event([0.8, 0.6]))
    root_before, second_before = root.embedding.clone(), second.embedding.clone()
    third = tree.insert(event([0.75, math.sqrt(1 - 0.75**2)]))
    assert third.parent is second
    assert torch.allclose(second.embedding, normalized(.7 * second_before + .3 * third.original_embedding))
    assert torch.allclose(root.embedding, normalized(.7 * root_before + .3 * third.original_embedding))
    assert torch.equal(root.original_embedding, torch.tensor([1., 0.]))
    # Exact inclusive threshold and earliest-created tie.
    exact = RecursiveEventTree(sigma=0.0, beta=0.0)
    a = exact.insert(event([1., 0., 0.]))
    exact.insert(event([0., 1., 0.]))
    c = exact.insert(event([0., 0., 1.]))
    assert c.parent is a
    with pytest.raises(ValueError):
        exact.insert(c)


def connect(parent, child):
    child.parent = parent
    child.depth = parent.depth + 1
    parent.children.append(child)


def test_leaf_completion_does_not_compete_with_active_beam():
    # Strong shallow leaf plus weaker internal route to an even stronger deep leaf.
    tree = RecursiveEventTree(beam_k=1)
    shallow, internal, deep = event([.8, .6]), event([.1, .995]), event([1., 0.])
    for i, node in enumerate((shallow, internal, deep)):
        node.node_id = i
    connect(internal, deep)
    tree.roots, tree.nodes = [shallow, internal], [shallow, internal, deep]
    hits, count = tree.retrieve(torch.tensor([1., 0.]))
    assert [h.event for h in hits] == [deep]
    assert count == 3


def test_internal_beam_prunes_routes_and_final_leaf_ties_use_ids():
    tree = RecursiveEventTree(beam_k=1)
    nodes = [event([1., 0.]), event([.8, .6]), event([.9, .1]), event([1., 0.])]
    for i, node in enumerate(nodes):
        node.node_id = i
    connect(nodes[0], nodes[2])
    connect(nodes[1], nodes[3])
    tree.roots, tree.nodes = nodes[:2], nodes
    hits, count = tree.retrieve(torch.tensor([1., 0.]))
    assert hits[0].event is nodes[2]
    assert count == 3  # pruned branch's child is not scored
    tree.roots, tree.nodes = [nodes[3], nodes[2]], nodes
    nodes[2].embedding = nodes[3].embedding.clone()
    hits, _ = tree.retrieve(torch.tensor([1., 0.]), k=2)
    assert [hit.event.node_id for hit in hits] == [2, 3]


def test_ancestor_distance_dedup_and_budget_skips():
    root, parent, a, b = [event([1., 0.], str(i)) for i in range(4)]
    for i, node in enumerate((root, parent, a, b)):
        node.node_id = i
    connect(root, parent)
    connect(parent, a)
    connect(parent, b)
    hits = [Hit(a, .9), Hit(b, .8)]
    assert list(expansion_order(hits)) == [a, b, parent, root]
    costs = {a.node_id: 11, b.node_id: 3, parent.node_id: 5, root.node_id: 2}
    packed = pack_events(hits, lambda es: sum(costs[e.node_id] for e in es) <= 10, max_events=2)
    assert packed == [b, parent]


def test_implicit_query_normalizes_and_uses_latest_four():
    peaks = [torch.tensor([0., 100.])] + [torch.tensor([3., 0.])] * 4
    assert torch.equal(OpenCLIPIndex.implicit_query(peaks), torch.tensor([1., 0.]))
    assert torch.equal(OpenCLIPIndex.implicit_query(peaks[:1]), torch.tensor([0., 1.]))
    with pytest.raises(ValueError):
        OpenCLIPIndex.implicit_query([])


def test_deep_tree_retrieval_and_stats_do_not_recurse_or_scan_parent_map():
    tree = RecursiveEventTree()
    nodes = [event([1., 0.]) for _ in range(1200)]
    for i, node in enumerate(nodes):
        node.node_id = i
        if i:
            connect(nodes[i - 1], node)
    tree.roots, tree.nodes = [nodes[0]], nodes
    hits, evaluated = tree.retrieve(torch.tensor([1., 0.]))
    assert hits[0].event is nodes[-1]
    assert evaluated == 1200
    assert tree.stats()['height'] == 1200
    assert len(pack_events(hits, lambda es: True)) == 8


def test_pinned_openclip_tokenizer_and_checkpoint_contract():
    import open_clip
    tokenizer = open_clip.get_tokenizer(OpenCLIPIndex.model_name, context_length=77)
    tokens = tokenizer(['event ' * 200])
    assert tokens.shape == (1, 77)
    assert tokens[0, -1] == 49407  # truncation preserves the end-of-text token
    assert open_clip.get_model_config(OpenCLIPIndex.model_name)['embed_dim'] == 768
    assert (OpenCLIPIndex.model_name, OpenCLIPIndex.pretrained) in open_clip.list_pretrained()
