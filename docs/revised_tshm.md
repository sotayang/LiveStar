# Revised TSHM implementation

`inference/streaming_infer.py` defaults to `--method revised`. The original
July 2026 implementation is preserved in `streaming_infer_legacy.py` and
`tshm.py`; select it explicitly with `--method legacy` (add `--pro` for its tree
mode). The existing `ablation.py`, `recall_eval.py`, and `kv_check.py` continue to
import the legacy implementation. Their results are not revised-method results.

## Run

Install the existing environment and weights as described in the README, then:

```bash
pip install -r requirements-revised.txt
python inference/streaming_infer.py --video /path/to/video.mp4 \
  --num-frames 300 --output outputs/revised.json

# Original LiveStarPro implementation and original default settings:
python inference/streaming_infer.py --method legacy --pro \
  --model-path inference --video /path/to/video.mp4 --num-segments 300

# Original LiveStar base implementation:
python inference/streaming_infer.py --method legacy \
  --model-path inference --video /path/to/video.mp4 --num-segments 300
```

The revised path uses the bundled model's `extract_feature` and InternLM
language model interface, but its own prompt construction and verification
adapter. Point `--model-path` to the bundled `inference/` directory containing
the downloaded weights. No weight conversion or model retraining is performed.
OpenCLIP downloads its own frozen indexing weights on the first run. By default,
`--num-frames 0` processes the whole video; no visual features are precomputed
for future frames. Static 448×448 single-frame processing is used.

For scheduled LMR/CDQ/TBR or other explicit questions, pass `--queries queries.json`:

```json
[
  {"query_id": "example-1", "timestamp": 10.5, "question": "What object did the person pick up earlier?"},
  {"query_id": "example-2", "timestamp": 20.0, "question": "What changed on the table?"}
]
```

Timestamps are seconds in the source video. Questions bypass the SVeD response
trigger. A query between sampled frames is answered before the next frame is
decoded or encoded. Empty answers remain in the JSON output. Queries beyond the
processed range produce an error, not an answer from a silently truncated video.
The harness provides causal inference; it does not include benchmark data or an
automatic correctness judge.

## Defined defaults and implementation

| Component | Revised default |
|---|---|
| SVeD | α=1.03; first response initializes the state and has no gate decision |
| Sampling | 1 FPS, monotonically increasing source-frame indices and timestamps |
| Index | OpenCLIP ViT-L-14 / `laion2b_s32b_b82k`, frozen 768-dimensional joint space |
| Indexed event | Normalized caption and peak-image embeddings, equally combined then normalized |
| CLIP caption | 77 OpenCLIP tokens including special tokens; no learned indexing projection |
| Explicit query | Same CLIP text encoder |
| Implicit query | Normalized average of the latest up to four completed-event peak-image embeddings |
| Insertion | All existing roots/internal nodes/leaves scored; best cosine ≥0.75 attaches, otherwise a new root |
| Ties | Ascending creation ID |
| Ancestor update | Every ancestor uses the same original new-event embedding, β=0.3, normalized after update |
| Retrieval | Score all roots; keep four active internal nodes per level; completed leaves do not occupy active beam slots |
| Retrieval termination | No active internal branch remains; jointly rank all completed leaves from all depths, retain four |
| Expansion | Matched leaves first; then ancestors by distance, descending matched-leaf score, ascending creation ID; deduplicate IDs |
| Memory injection | At most eight events and 1,024 actual generator input tokens, including peak visual tokens, captions, timestamps, wrappers and separators |
| Injected caption | At most 96 generator-tokenizer tokens, independently of CLIP's 77-token limit |
| Total window | 8,192 tokens including 128 reserved output tokens: at most 8,064 input, at most 7,040 non-memory input |
| Overflow | Skip complete retrieved events that cannot fit; prune active clips at their median score, then evict oldest completed clips; tied single-clip overflow drops non-peak frames |
| Visual storage | Encode each arriving frame once; active and external peak features reside on CPU and only the current bounded prompt is transferred to the GPU |
| KV reuse | Verification reuses the identical token **and visual-identity** prefix; all current answer tokens are rescored; generation uses native intra-response KV caching |

Historical peak visual features are actually injected with matching visual token
positions, not merely counted in a budget. Non-memory input includes the system
prompt, conversation delimiters, question, active images, and active captions.
The final prompt is tokenized with the bundled generator tokenizer before each
model call. A query too large to fit the window fails explicitly. Memory from the
current narration is retained for its subsequent perplexity checks subject to the
same packing limits; scheduled answers do not change narration or its reference
perplexity.

There is no temporal-neighbor, sibling, or leaf-child expansion. Parent pointers
are stored at insertion; retrieval does not reconstruct a parent map by scanning
the entire tree. Insertion costs O(nd+hd). Similarity scoring at retrieval costs
O(d(r+kbh)); a logarithmic height requires balanced growth and controlled root and
branch counts. No such guarantee is imposed. External nodes and peak-feature
storage grow with history; the eight-event cap limits injection, not nodes scored.

## Timing and run artifacts

The JSON records the configuration, narration, gate decisions, scheduled answers,
retrieved event IDs, scored-node counts, actual memory tokens, maximum input token
count, tree statistics, and measured timings. Initialization is explicitly labeled
and can be excluded from gate confusion matrices.

- `seconds` / `fps`: from reader startup to the last inference/maintenance step,
  including decode, preprocessing, visual encoding, SVeD, generation, OpenCLIP
  indexing, insertion and ancestor maintenance, retrieval, and CPU bookkeeping.
  Model/weight loading and result-file serialization are excluded. This is offline
  processing throughput without live arrival waiting, not a live display rate.
- `retrieval_mean_ms`: query encoding, tree search, ancestor expansion, deduplication,
  token-budget checks, and context assembly. Historical indexing/maintenance is
  excluded from this component and remains included in the end-to-end time.
- `timing_seconds.retrieval` contains its subcomponents (`query_encoding`,
  `tree_search`, `context_assembly`); do not sum parent and children twice.
  CUDA operations are synchronized at timing boundaries.

No reported paper accuracy, latency, or FPS is hard-coded or claimed as reproduced
by this release. The new implementation must be evaluated with the intended
checkpoint, frozen questions, hardware and configuration before attributing
experimental results to this commit. CPU tests validate logic and the bundled
language-model API; they do not establish full-checkpoint quality or GPU speed.

## Checks

```bash
pip install pytest
python -m pytest tests -q
python -m compileall -q inference/tshm_revised.py inference/revised_model.py \
  inference/streaming_infer_revised.py inference/streaming_infer.py
```

Tests use the repository tokenizer, a small randomly initialized instance of the
bundled InternLM, and synthetic streams. They cover cached/uncached perplexity,
visual identity invalidation, mixed-depth leaves, ancestor updates, tie handling,
packing, causal query scheduling, and bounded active context. GPU follow-up should
compare `--no-kv` with the default on the same checkpoint and frozen video, then
measure the intended long-span tasks and timing on the target hardware.

OpenCLIP API/weight references: [implementation](https://github.com/mlfoundations/open_clip),
[checkpoint](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K).
