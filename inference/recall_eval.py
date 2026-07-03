"""Quality eval of the recall fix: off vs verbatim(old bug) vs reframed(new).

Shares ONE fixed vit_cache across modes to remove ViT non-determinism.
Metrics (no external judge):
  - decode/silent counts
  - redundancy: mean max-cosine of each spoken caption to any earlier one (lower=better)
  - ECHO: mean cosine(generated caption, the recalled text) over decodes where
    recall fired (lower=better; high means the model parroted the recall).
"""
from types import SimpleNamespace
import torch
from transformers import AutoModel, AutoTokenizer
from streaming_infer import load_video, StreamingSession
from tshm import embed_text


def args(**kw):
    b = dict(l_max=72, alpha=1.06, sigma=0.75, beta=0.3, beam_k=3, max_recall=2,
             num_runs=2, max_new_tokens=160, trace=False, no_tshm=False,
             no_retrieval=False, no_viscache=False, kv=False, recall_verbatim=False, recall_min_gap=0)
    b.update(kw); return SimpleNamespace(**b)


tok = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
model = AutoModel.from_pretrained("./", trust_remote_code=True).half().cuda().to(torch.bfloat16).eval()
gc = dict(temperature=0.0, max_new_tokens=160, top_p=0.1, num_beams=1, repetition_penalty=1.05)
fr = load_video("../assets/videos/HPtIGhOsViM.mp4", max_num=1, num_segments=40, sample_fps=1)

with torch.no_grad():
    s0 = StreamingSession(model, tok, gc, fr, args()); s0._precompute_visual()
    shared = s0.vit_cache


def emb(txt):
    return embed_text(model, tok, txt)


def run(a):
    with torch.no_grad():
        s = StreamingSession(model, tok, gc, fr, a)
        s.vit_cache = dict(shared)
        summ = s.run(len(fr))
    caps = [c for _, c in s.narration]
    E = [emb(c) for c in caps]
    redun = []
    for i in range(1, len(E)):
        redun.append(max(float(torch.dot(E[i], E[j])) for j in range(i)))
    mean_redun = sum(redun) / len(redun) if redun else 0.0
    echo = []
    for c, rc in zip(caps, s.recall_used):
        if rc:
            echo.append(float(torch.dot(emb(c), emb(rc))))
    mean_echo = sum(echo) / len(echo) if echo else float("nan")
    return summ, caps, mean_redun, mean_echo, len(echo)


modes = [("off", args(no_retrieval=True)),
         ("verbatim(old)", args(recall_verbatim=True)),
         ("reframed(new)", args()),
         ("reframed+gate10", args(recall_min_gap=10))]

results = {}
for name, a in modes:
    summ, caps, mr, me, ne = run(a)
    results[name] = (summ, caps, mr, me, ne)
    print("%-14s decode=%d silent=%d | redundancy=%.3f | echo=%.3f (n=%d)"
          % (name, summ["decode"], summ["silent"], mr, me, ne))

print("\nLower redundancy & echo = better. Transcripts:")
for name in results:
    print("\n== %s ==" % name)
    for t_c in results[name][1]:
        print("  " + t_c[:105])
