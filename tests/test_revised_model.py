from types import SimpleNamespace
from pathlib import Path
import torch
import pytest
from transformers import AutoTokenizer
from inference.configuration_livestar_llm import InternLM2Config
from inference.modeling_livestar_llm import InternLM2ForCausalLM
from revised_model import ModelAdapter, VisualFrame, Prompt
from tshm_revised import Event


@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(str(Path(__file__).resolve().parents[1] / 'inference'),
                                         trust_remote_code=True, local_files_only=True)


def make_adapter(tok, kv=True):
    torch.manual_seed(5)
    config = InternLM2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                            max_position_embeddings=8192, attn_implementation='eager')
    model = SimpleNamespace(language_model=InternLM2ForCausalLM(config).eval(), device=torch.device('cpu'),
                            num_image_token=2, template='internvl2_5', system_message='Test.')
    return ModelAdapter(model, tok, use_kv=kv)


def frame(i):
    return VisualFrame(i, float(i), torch.full((1, 2, 16), float(i + 1)))


def test_real_tokenizer_memory_caption_and_visual_accounting(tokenizer):
    adapter = make_adapter(tokenizer)
    ev = Event('hello world ' * 300, torch.ones(4), frame(0), 0, 1, node_id=0)
    assert len(adapter.ids(adapter.caption(ev.caption), special=False)) <= 96
    prompt = adapter.render([], frame(3), [ev])
    ids, embeddings, _ = adapter._inputs(prompt)
    positions = (ids[0] == adapter.image_token_id).nonzero().flatten()
    assert len(positions) == 4
    assert torch.equal(embeddings[0, positions[:2]], frame(0).features[0])
    assert torch.equal(embeddings[0, positions[2:]], frame(3).features[0])
    assert adapter.memory_tokens([ev]) > 96 + 2  # timestamps, wrappers, separators included


def test_kv_matches_uncached_ppl_and_checks_visual_identity(tokenizer):
    cached = make_adapter(tokenizer)
    fresh = ModelAdapter(cached.model, tokenizer, use_kv=False)
    f = frame(0)
    clip = SimpleNamespace(frames=[f], caption='A person enters.')
    prompt = cached.render([clip], frame(1), answer='A person walks.')
    first = cached.verify(prompt)
    assert first == pytest.approx(fresh.verify(prompt), rel=1e-6)
    repeated = cached.verify(prompt)
    assert cached.last_reused_tokens > 0
    assert repeated == pytest.approx(first, rel=1e-6)
    # Same text IDs, new visual content identity: invalidate at the first image.
    changed = Prompt(prompt.ids, [frame(9), frame(1)], prompt.answer_start, prompt.answer_end)
    value = cached.verify(changed)
    first_image = prompt.ids.index(cached.image_token_id)
    assert cached.last_reused_tokens <= first_image
    assert value == pytest.approx(fresh.verify(changed), rel=1e-6)
    # Pruned/reordered context cannot reuse a stale prefix.
    short = cached.render([], frame(2), answer='The scene changes.')
    assert cached.verify(short) == pytest.approx(fresh.verify(short), rel=1e-6)


def test_bundled_llm_generation_api_smoke(tokenizer):
    adapter = make_adapter(tokenizer)
    text = adapter.generate(adapter.render([], frame(0)), max_new_tokens=2)
    assert isinstance(text, str)


def test_real_adapter_session_packing_and_eviction(tokenizer):
    from streaming_infer_revised import RevisedSession, build_args
    from test_revised_session import FakeIndex
    adapter = make_adapter(tokenizer)
    # Fixed captions isolate stream/packing logic; verification uses the real
    # bundled InternLM with the real tokenizer and synthetic visual features.
    adapter.generate = lambda prompt, max_new_tokens: 'A person walks.'
    args = build_args(['--context-window', '360', '--memory-tokens', '150',
                      '--max-new-tokens', '8', '--alpha', '0.1'])
    session = RevisedSession(adapter, FakeIndex(), args)
    for i in range(12):
        f = frame(i)
        f.image = object()
        session.observe(f)
    assert len(session.tree) > 0
    assert any(log['event_ids'] for log in session.retrieval_log)
    assert session.max_input_tokens <= args.context_window
    assert max(log['memory_tokens'] for log in session.retrieval_log) <= 150
