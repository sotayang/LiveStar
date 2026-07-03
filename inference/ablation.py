"""TSHM ablation: full vs no-retrieval vs no-TSHM(baseline) on identical frames.

Redundancy proxy: for each spoken caption, its max cosine similarity to any
earlier spoken caption. redundant_repeats = #captions with max-prev-sim > 0.9
(i.e. the model re-narrates a scene it already described). Lower is better.
Loads the 8B model once and reuses it across the three modes.
"""

import argparse
from types import SimpleNamespace

import torch
from transformers import AutoModel, AutoTokenizer

from streaming_infer import load_video, StreamingSession
from tshm import embed_text


def make_args(**kw):
    base = dict(l_max=96, alpha=1.06, sigma=0.75, beta=0.3, beam_k=3,
                max_recall=2, num_runs=2, max_new_tokens=256, trace=False,
                no_tshm=False, no_retrieval=False)
    base.update(kw)
    return SimpleNamespace(**base)


def narrative_stats(model, tok, narration):
    caps = [c for _, c in narration]
    embs = [embed_text(model, tok, c) for c in caps]
    max_prev = []
    for i in range(1, len(embs)):
        max_prev.append(max(float(torch.dot(embs[i], embs[j])) for j in range(i)))
    redundant = sum(1 for s in max_prev if s > 0.9)
    mean_mp = round(sum(max_prev) / len(max_prev), 3) if max_prev else 0.0
    return dict(spoken=len(caps), redundant_repeats=redundant,
                mean_max_prev_sim=mean_mp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="./")
    ap.add_argument("--video", default="../assets/videos/HPtIGhOsViM.mp4")
    ap.add_argument("--num-segments", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = (AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
             .half().cuda().to(torch.bfloat16).eval())
    gen_cfg = dict(temperature=0.0, max_new_tokens=256, top_p=0.1,
                   num_beams=1, repetition_penalty=1.05)
    frames = load_video(args.video, max_num=1, num_segments=args.num_segments,
                        sample_fps=1)
    n = len(frames)

    modes = [("full-TSHM", make_args()),
             ("no-retrieval", make_args(no_retrieval=True)),
             ("no-TSHM(baseline)", make_args(no_tshm=True))]

    results = {}
    for name, a in modes:
        with torch.no_grad():
            sess = StreamingSession(model, tok, gen_cfg, frames, a)
            summ = sess.run(n)
        stats = narrative_stats(model, tok, sess.narration)
        results[name] = (summ, stats, sess.narration)
        print("\n### %-18s decode=%d silent=%d retrieval=%d tree=%s"
              % (name, summ["decode"], summ["silent"], summ["retrieval"],
                 summ["tree"]))
        print("    active_tokens=%d fps=%.2f | spoken=%d redundant_repeats=%d "
              "mean_max_prev_sim=%.3f" % (summ["active_tokens"], summ["fps"],
              stats["spoken"], stats["redundant_repeats"],
              stats["mean_max_prev_sim"]))

    print("\n" + "=" * 70)
    print("REDUNDANCY (lower=better): re-narrating already-described scenes")
    for name in results:
        s = results[name][1]
        print("  %-18s repeats=%d/%d  mean_max_prev_sim=%.3f"
              % (name, s["redundant_repeats"], s["spoken"], s["mean_max_prev_sim"]))

    for name in results:
        print("\n== NARRATION [%s] ==" % name)
        for t, cap in results[name][2]:
            print("  t=%-3d %s" % (t, cap[:110]))


if __name__ == "__main__":
    main()
