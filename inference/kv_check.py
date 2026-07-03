"""Within-trajectory validation of the verification KV cache.

Share ONE fixed vit_cache across both runs so per-frame features are identical
-> LLM deterministic -> --kv must reproduce no-kv EXACTLY if the cache is correct.
"""
from types import SimpleNamespace
import torch
from transformers import AutoModel, AutoTokenizer
from streaming_infer import load_video, StreamingSession


def args(**kw):
    b = dict(l_max=100000, alpha=1.06, sigma=0.75, beta=0.3, beam_k=3, max_recall=2,
             num_runs=2, max_new_tokens=128, trace=False, no_tshm=False,
             no_retrieval=False, no_viscache=False, kv=False)
    b.update(kw); return SimpleNamespace(**b)


tok = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
model = AutoModel.from_pretrained("./", trust_remote_code=True).half().cuda().to(torch.bfloat16).eval()
gc = dict(temperature=0.0, max_new_tokens=256, top_p=0.1, num_beams=1, repetition_penalty=1.05)
fr = load_video("../assets/videos/HPtIGhOsViM.mp4", max_num=1, num_segments=48, sample_fps=2)

# build ONE fixed visual cache, shared by both runs
with torch.no_grad():
    s0 = StreamingSession(model, tok, gc, fr, args())
    s0._precompute_visual()
    shared = s0.vit_cache


def run(a):
    with torch.no_grad():
        s = StreamingSession(model, tok, gc, fr, a)
        s.vit_cache = dict(shared)   # inject fixed features
        summ = s.run(len(fr))
    return summ, [c for _, c in s.narration]


sN, nN = run(args(kv=False))
sK, nK = run(args(kv=True))

print("\n== KV CACHE (shared features, fixed trajectory) ==")
print("  no-kv : total=%.2fs t_ppl=%.2f t_gen=%.2f decode=%d silent=%d"
      % (sN["seconds"], sN["t_ppl"], sN["t_gen"], sN["decode"], sN["silent"]))
print("  kv    : total=%.2fs t_ppl=%.2f t_gen=%.2f decode=%d silent=%d"
      % (sK["seconds"], sK["t_ppl"], sK["t_gen"], sK["decode"], sK["silent"]))
if sK["t_ppl"] > 0:
    print("  t_ppl speedup: %.2fx" % (sN["t_ppl"] / sK["t_ppl"]))
print("  narration identical: %s (len %d vs %d)" % (nN == nK, len(nN), len(nK)))
if nN != nK:
    for i, (x, y) in enumerate(zip(nN, nK)):
        if x != y:
            print("    DIFF#%d\n      no-kv: %s\n      kv   : %s" % (i, x[:100], y[:100])); break
