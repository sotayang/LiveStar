"""Default LiveStarPro revision implementation; original remains in *_legacy.py."""
import argparse
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import time

import torch

from revised_model import ModelAdapter, VisualFrame
from tshm_revised import Event, OpenCLIPIndex, RecursiveEventTree, pack_events


class Timings:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.seconds, self.calls = defaultdict(float), defaultdict(int)

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name):
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.seconds[name] += time.perf_counter() - start
            self.calls[name] += 1


@dataclass
class Clip:
    frames: list
    caption: str
    t_start: float
    t_end: float
    peak: VisualFrame
    peak_embedding: object = None

    def add(self, frame):
        self.frames.append(frame)
        self.t_end = frame.timestamp
        if (frame.score, frame.frame_id) < (self.peak.score, self.peak.frame_id):
            self.peak, self.peak_embedding = frame, None


class RevisedSession:
    def __init__(self, adapter, index, args):
        self.adapter, self.index, self.args = adapter, index, args
        self.tree = RecursiveEventTree(args.sigma, args.beta, args.beam_k)
        self.clips, self.narration, self.answers, self.decisions = [], [], [], []
        self.recent_peaks = deque(maxlen=4)
        self.timer = Timings(adapter.model.device)
        self.threshold = None
        self.current = None
        self.frames_seen = 0
        self.max_input_tokens = 0
        self.retrieval_log = []
        self.active_recall, self.prepared_events = [], []

    def _trace(self, text):
        if self.args.trace:
            print(text, flush=True)

    def _peak_embedding(self, clip):
        if clip.peak_embedding is None:
            with self.timer.measure('index_encoding'):
                clip.peak_embedding = self.index.image(clip.peak.image)
        return clip.peak_embedding

    def _finish(self, clip):
        if self.index is not None:
            self.recent_peaks.append(self._peak_embedding(clip))

    def _archive(self, clip):
        if self.index is None:
            return
        image_embedding = self._peak_embedding(clip)
        with self.timer.measure('index_encoding'):
            text_embedding = self.index.text(clip.caption)
        # External storage keeps only the peak's generator features, on CPU.
        peak = VisualFrame(clip.peak.frame_id, clip.peak.timestamp,
                           clip.peak.features, score=clip.peak.score)
        with self.timer.measure('tree_maintenance'):
            self.tree.insert(Event(clip.caption, text_embedding + image_embedding,
                                   peak, clip.t_start, clip.t_end))

    def _prune_once(self):
        changed = False
        with self.timer.measure('peak_end'):
            for clip in self.clips:
                if len(clip.frames) > 1:
                    median = statistics.median(f.score for f in clip.frames)
                    kept = [f for f in clip.frames if f.score <= median]
                    changed |= len(kept) < len(clip.frames)
                    clip.frames = kept
        return changed

    def _base_prompt(self, current, question=None, answer=None):
        # Reserve a dedicated memory budget; overhead, history and current image
        # all count as other input. Nothing relies on nominal 16-token estimates.
        other_limit = self.args.context_window - self.args.max_new_tokens - self.args.memory_tokens
        while True:
            base = self.adapter.render(self.clips, current, question=question, answer=answer)
            if len(base.ids) <= other_limit:
                return base
            if self._prune_once():
                continue
            if len(self.clips) > 1:
                self._archive(self.clips.pop(0))
                continue
            # Tied scores may keep all frames. Preserve the best peak; discard
            # the oldest tied non-peak frame until the single active clip fits.
            if self.clips and len(self.clips[0].frames) > 1:
                clip = self.clips[0]
                removable = [f for f in clip.frames if f is not clip.peak]
                victim = max(removable, key=lambda f: (f.score, -f.frame_id))
                clip.frames.remove(victim)
                continue
            raise ValueError('Prompt cannot fit the non-memory input budget; '
                             'shorten the query or increase --context-window')

    def prepare(self, current, question=None, recall=False, answer=None, fixed_events=None):
        base = self._base_prompt(current, question, answer)
        selected, scored = [], 0
        if recall and self.index is not None and len(self.tree):
            with self.timer.measure('retrieval'):
                with self.timer.measure('query_encoding'):
                    if question is not None:
                        query = self.index.text(question)
                    else:
                        peaks = list(self.recent_peaks)
                        if not peaks:
                            peaks = [self._peak_embedding(self.clips[-1])]
                        query = self.index.implicit_query(peaks)
                with self.timer.measure('tree_search'):
                    hits, scored = self.tree.retrieve(query)
                with self.timer.measure('context_assembly'):
                    def fits(events):
                        candidate = self.adapter.render(self.clips, current, events, question)
                        memory_tokens = max(self.adapter.memory_tokens(events),
                                            len(candidate.ids) - len(base.ids))
                        return (memory_tokens <= self.args.memory_tokens and
                                len(candidate.ids) + self.args.max_new_tokens <= self.args.context_window)
                    selected = pack_events(hits, fits, self.args.max_events)
                    base = self.adapter.render(self.clips, current, selected, question)
            self.retrieval_log.append(dict(timestamp=current.timestamp, nodes_scored=scored,
                                           event_ids=[e.node_id for e in selected],
                                           memory_tokens=self.adapter.memory_tokens(selected)))
        if fixed_events:
            selected = []
            for event in fixed_events:
                candidate = self.adapter.render(self.clips, current, selected + [event], question, answer)
                if (max(self.adapter.memory_tokens(selected + [event]), len(candidate.ids) - len(base.ids))
                        <= self.args.memory_tokens and len(candidate.ids) <= self.args.context_window):
                    selected.append(event)
            base = self.adapter.render(self.clips, current, selected, question, answer)
        self.prepared_events = selected
        reserve = self.args.max_new_tokens if answer is None else 0
        if len(base.ids) + reserve > self.args.context_window:
            raise AssertionError('Actual generator prompt exceeds context budget')
        self.max_input_tokens = max(self.max_input_tokens, len(base.ids))
        return base

    def _generate(self, current, question=None):
        prompt = self.prepare(current, question=question, recall=True)
        with self.timer.measure('generation'):
            return self.adapter.generate(prompt, self.args.max_new_tokens)

    def _verify(self, current, caption):
        prompt = self.prepare(current, answer=caption, fixed_events=self.active_recall)
        with self.timer.measure('verification'):
            return self.adapter.verify(prompt)

    def observe(self, frame):
        if self.current is not None and frame.timestamp <= self.current.timestamp:
            raise ValueError('Frames must arrive in strictly increasing time order')
        self.current = frame
        self.frames_seen += 1
        previous = self.clips[-1] if self.clips else None
        reference_before = self.threshold
        ppl = self._verify(frame, previous.caption) if previous and previous.caption else None
        trigger = ppl is None or self.threshold is None or ppl > self.threshold * self.args.alpha
        if trigger:
            if previous:
                self._finish(previous)
            caption = self._generate(frame)
            self.active_recall = list(self.prepared_events)
            clip = Clip([frame], caption, frame.timestamp, frame.timestamp, frame)
            self.clips.append(clip)
            # Self-verification shows the just-described frame again, matching
            # the original convention, with exact duplicate visual accounting.
            ref = self._verify(frame, caption) if caption else None
            frame.score = ref if ref is not None else float('inf')
            self.threshold = ref
            self.narration.append(dict(timestamp=frame.timestamp, text=caption))
        else:
            frame.score = ppl
            previous.add(frame)
        self.decisions.append(dict(timestamp=frame.timestamp, initialization=previous is None,
                                   triggered=bool(trigger), ppl=ppl,
                                   reference_ppl=reference_before, updated_reference_ppl=self.threshold,
                                   response_nonempty=bool(caption) if trigger else False))
        self._trace('t=%.3fs %s | tree=%d' % (frame.timestamp, 'decode' if trigger else 'silent', len(self.tree)))

    def answer(self, question, query_id=None, query_timestamp=None):
        if self.current is None:
            raise ValueError('No observed frame precedes the query')
        timestamp = self.current.timestamp if query_timestamp is None else query_timestamp
        if timestamp < self.current.timestamp:
            raise ValueError('Query precedes an already observed frame')
        # Query responses do not mutate the background narration or SVeD state.
        text = self._generate(self.current, question=('Query timestamp: %.3fs. %s' % (timestamp, question)))
        result = dict(query_id=query_id, query_timestamp=timestamp,
                      last_observed_timestamp=self.current.timestamp, question=question, answer=text)
        self.answers.append(result)
        return result

    def summary(self, seconds):
        retrieval_calls = self.timer.calls.get('retrieval', 0)
        return dict(method='revised', frames=self.frames_seen, seconds=seconds,
                    fps=self.frames_seen / seconds if seconds else 0,
                    timing_seconds=dict(self.timer.seconds), timing_calls=dict(self.timer.calls),
                    retrieval_mean_ms=(1000 * self.timer.seconds['retrieval'] / retrieval_calls
                                       if retrieval_calls else None),
                    tree=self.tree.stats(), max_input_tokens=self.max_input_tokens,
                    active_frames=sum(len(c.frames) for c in self.clips),
                    narration=self.narration, answers=self.answers, decisions=self.decisions,
                    retrieval_log=self.retrieval_log, config=vars(self.args))


def sampled_indices(total, fps, sample_fps, max_frames=0):
    if not math.isfinite(fps) or fps <= 0 or not 0 < sample_fps <= fps:
        raise ValueError('Require finite source FPS and 0 < sample FPS <= source FPS')
    previous, sample = -1, 0
    while not max_frames or sample < max_frames:
        index = int(math.floor(sample * fps / sample_fps + 1e-9))
        if index >= total:
            break
        if index != previous:
            yield index, index / fps
            previous = index
        sample += 1


def read_queries(path):
    if path is None:
        return []
    queries = json.loads(Path(path).read_text())
    seen = set()
    for query in queries:
        if (not math.isfinite(query['timestamp']) or query['timestamp'] < 0 or
                not isinstance(query['question'], str) or not query['question'].strip()):
            raise ValueError('Each query needs a finite nonnegative timestamp and nonempty question')
        if query['query_id'] in seen:
            raise ValueError('Duplicate query_id')
        seen.add(query['query_id'])
    return sorted(queries, key=lambda q: q['timestamp'])


def run_schedule(session, samples, read_frame, queries, end_timestamp=None):
    """Inspect metadata, answer queries, THEN decode the next sampled frame.

    read_frame is intentionally called after earlier queries, not by a prefetching
    frame iterator. Empty answers remain in session.answers.
    """
    queries = deque(queries)
    for frame_id, timestamp in samples:
        while queries and queries[0]['timestamp'] < timestamp:
            query = queries.popleft()
            session.answer(query['question'], query['query_id'], query['timestamp'])
        session.observe(read_frame(frame_id, timestamp))
        while queries and queries[0]['timestamp'] == timestamp:
            query = queries.popleft()
            session.answer(query['question'], query['query_id'], query['timestamp'])
    while queries and end_timestamp is not None and queries[0]['timestamp'] <= end_timestamp:
        query = queries.popleft()
        session.answer(query['question'], query['query_id'], query['timestamp'])
    if queries:
        raise ValueError('Unanswered query lies beyond processed frames; increase --num-frames '
                         'or use an in-range timestamp')


def build_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', default=str(Path(__file__).parent))
    p.add_argument('--video', default=str(Path(__file__).parent.parent / 'assets/videos/HPtIGhOsViM.mp4'))
    p.add_argument('--num-frames', type=int, default=0, help='0 processes the whole video')
    p.add_argument('--sample-fps', type=float, default=1)
    p.add_argument('--alpha', type=float, default=1.03)
    p.add_argument('--sigma', type=float, default=0.75)
    p.add_argument('--beta', type=float, default=0.3)
    p.add_argument('--beam-k', type=int, default=4)
    p.add_argument('--max-events', type=int, default=8)
    p.add_argument('--memory-tokens', type=int, default=1024)
    p.add_argument('--context-window', type=int, default=8192)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--no-retrieval', action='store_true')
    p.add_argument('--no-kv', action='store_true', help='Disable verification-prefix KV reuse')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--queries', help='JSON list of query_id, timestamp (seconds), question')
    p.add_argument('--output', default='outputs/revised_run.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--pro', action='store_true', help='Compatibility flag; revised already enables Pro')
    args = p.parse_args(argv)
    if (args.num_frames < 0 or not math.isfinite(args.alpha) or args.alpha <= 0 or
            args.max_new_tokens < 1 or args.max_events < 1 or args.memory_tokens < 0 or
            args.context_window <= args.memory_tokens + args.max_new_tokens):
        p.error('Invalid frame, threshold or context budget')
    return args


def main():
    args = build_args()
    # Heavy optional imports remain lazy so --help and CPU unit tests work.
    from decord import VideoReader, cpu
    from PIL import Image
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.device(args.device).type == 'cuda' else torch.float32
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
                                      torch_dtype=dtype).to(args.device).eval()
    if model.local_num_frames != 1:
        raise ValueError('Causal per-frame encoding requires local_num_frames=1')
    index = None if args.no_retrieval else OpenCLIPIndex(args.device)
    adapter = ModelAdapter(model, tok, use_kv=not args.no_kv)
    session = RevisedSession(adapter, index, args)
    queries = read_queries(args.queries)
    transform = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                           T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # Model loading is excluded. Reader startup, decode, preprocessing, visual
    # encoding, indexing, retrieval, generation and all CPU bookkeeping are included.
    session.timer.sync()
    start = time.perf_counter()
    with session.timer.measure('decode_preprocess'):
        reader = VideoReader(args.video, ctx=cpu(0), num_threads=1)
        fps = float(reader.get_avg_fps())
    def read_frame(frame_id, timestamp):
        with session.timer.measure('decode_preprocess'):
            image = Image.fromarray(reader[frame_id].asnumpy()).convert('RGB')
            pixels = transform(image).unsqueeze(0).to(device=args.device, dtype=dtype)
        with session.timer.measure('visual_encoding'):
            features = model.extract_feature(pixels).detach().cpu()
        return VisualFrame(frame_id, timestamp, features, image)

    with torch.inference_mode():
        samples = sampled_indices(len(reader), fps, args.sample_fps, args.num_frames)
        # The last sampled frame can precede the video's actual end. Only a full
        # run may answer queries in this final sampling interval.
        end_timestamp = (len(reader) - 1) / fps if not args.num_frames else None
        run_schedule(session, samples, read_frame, queries, end_timestamp)
    session.timer.sync()
    result = session.summary(time.perf_counter() - start)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('method', 'frames', 'seconds', 'fps',
                                                 'retrieval_mean_ms', 'tree')}, indent=2))
    print('Run artifact:', output)


if __name__ == '__main__':
    main()
