from types import SimpleNamespace
import json
import torch
import pytest

from revised_model import Prompt, VisualFrame
from streaming_infer import select_method
from streaming_infer_revised import RevisedSession, build_args, sampled_indices, run_schedule
from tshm_revised import OpenCLIPIndex


class FakeAdapter:
    def __init__(self):
        self.model = SimpleNamespace(device='cpu')
        self.prompts = []
        self.seen = 0

    def render(self, clips, current, events=(), question=None, answer=None):
        # Includes timestamp, image, caption and role overhead; intentionally
        # small budgets force eviction throughout the integration test.
        n = 20 + sum(16 * len(c.frames) + len(c.caption) for c in clips)
        n += self.memory_tokens(events) + len(question or '') + len(answer or '')
        return Prompt(list(range(n)), [current] + [e.visual for e in events])

    def memory_tokens(self, events):
        return sum(25 + len(e.caption[:96]) for e in events)

    def generate(self, prompt, max_new_tokens):
        self.prompts.append(prompt)
        return 'scene'

    def verify(self, prompt):
        self.seen += 1
        return 1. if self.seen % 2 else 2.


class FakeIndex:
    image = staticmethod(lambda image: torch.tensor([1., 0., 0.]))
    text = staticmethod(lambda text: torch.tensor([0.9, 0.1, 0.]))
    implicit_query = staticmethod(OpenCLIPIndex.implicit_query)


def frame(i):
    return VisualFrame(i, float(i), torch.ones(1, 2, 16), image=object())


def test_default_and_legacy_selection():
    assert select_method(['--video', 'x']) == ('revised', ['--video', 'x'])
    assert select_method(['--method', 'legacy', '--pro']) == ('legacy', ['--pro'])
    args = build_args([])
    assert (args.alpha, args.beam_k, args.max_events, args.memory_tokens) == (1.03, 4, 8, 1024)
    assert args.context_window - args.max_new_tokens - args.memory_tokens == 7040


def test_stream_eviction_index_retrieval_and_actual_budget():
    args = build_args(['--context-window', '240', '--memory-tokens', '100', '--max-new-tokens', '10'])
    session = RevisedSession(FakeAdapter(), FakeIndex(), args)
    for i in range(50):
        session.observe(frame(i))
    assert len(session.tree) > 10
    assert session.retrieval_log
    assert all(log['memory_tokens'] <= 100 and len(log['event_ids']) <= 8 for log in session.retrieval_log)
    assert session.max_input_tokens + args.max_new_tokens <= args.context_window
    assert sum(len(c.frames) for c in session.clips) < 10
    assert all(e.visual.features.device.type == 'cpu' and e.visual.image is None for e in session.tree.nodes)
    assert session.decisions[0]['initialization']
    assert all(not decision['initialization'] for decision in session.decisions[1:])
    narration, threshold = list(session.narration), session.threshold
    session.answer('Where?', 'q1', 49.5)
    assert session.narration == narration and session.threshold == threshold
    assert session.answers[0]['last_observed_timestamp'] <= session.answers[0]['query_timestamp']
    json.dumps(session.summary(1.), allow_nan=False)
    assert session.timer.calls['index_encoding'] > 0
    assert session.timer.calls['tree_maintenance'] == len(session.tree)


def test_score_ties_cannot_make_active_context_unbounded():
    adapter = FakeAdapter()
    adapter.verify = lambda prompt: 1.
    args = build_args(['--context-window', '180', '--memory-tokens', '80', '--max-new-tokens', '10'])
    session = RevisedSession(adapter, FakeIndex(), args)
    for i in range(70):
        session.observe(frame(i))
    assert len(session.clips) == 1
    assert len(session.clips[0].frames) < 6
    assert session.clips[0].peak.frame_id == 0


def test_queries_are_answered_before_future_frame_decode_and_encode():
    log = []
    class Session:
        def observe(self, frame):
            log.append(('observe', frame.timestamp))
        def answer(self, text, query_id, timestamp):
            log.append(('answer', timestamp))
    def read(i, t):
        log.append(('decode_and_encode', t))
        return frame(i)
    queries = [dict(query_id=str(t), timestamp=t, question='?') for t in (0, .5, 1, 1.5, 2.5)]
    run_schedule(Session(), [(0, 0), (1, 1), (2, 2)], read, queries, end_timestamp=2.9)
    assert log.index(('answer', .5)) < log.index(('decode_and_encode', 1))
    assert log.index(('observe', 1)) < log.index(('answer', 1))
    assert log.index(('answer', 1.5)) < log.index(('decode_and_encode', 2))
    assert log[-1] == ('answer', 2.5)
    with pytest.raises(ValueError):
        run_schedule(Session(), [(0, 0)], read, queries)


def test_fractional_fps_sampling_and_limits():
    samples = list(sampled_indices(100, 29.97, 1, 3))
    assert [i for i, _ in samples] == [0, 29, 59]
    assert all(t <= i for i, (_, t) in enumerate(samples))
    with pytest.raises(ValueError):
        list(sampled_indices(100, 0, 1))
