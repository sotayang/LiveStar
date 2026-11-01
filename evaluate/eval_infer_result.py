from collections import defaultdict
import json
import re
from pathlib import Path    

def parse_timestamp_to_seconds(time_string: str) -> int:
    """
    Converts a time string in HH:MM:SS, MM:SS, or SS format to total seconds.

    Args:
        time_string (str): The time string to convert (e.g., "01:23:45", "23:45", "45").
                           Parts are separated by colons.

    Returns:
        int: The total number of seconds.
    """
    parts = list(map(int, time_string.strip().split(":")))
    if len(parts) == 3:  # HH:MM:SS
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:  # MM:SS
        return parts[0] * 60 + parts[1]
    elif len(parts) == 1:  # SS
        return parts[0]
    else:
        raise ValueError(f"Invalid time string format: {time_string}")


def _load_ground_truth_segments(ground_truth_file_path: str) -> dict[str, list[tuple[int, int, str]]]:
    """
    Loads ground truth data from a text file and structures it into time segments.

    The GT file format expected:
    - Lines ending with '.mp4' denote a new video ID.
    - Subsequent lines starting with 'Timestamp: HH:MM:SS Caption: text' provide captions.

    Args:
        ground_truth_file_path (str): Path to the ground truth text file.

    Returns:
        dict[str, list[tuple[int, int, str]]]: A dictionary mapping video IDs to a list of
        segments. Each segment is a tuple (segment_start_sec, segment_end_sec, caption).
        - The first segment for a video is (timestamp, timestamp, caption).
        - Subsequent segments are (prev_timestamp + 1, current_timestamp, caption).
    """
    ground_truth_captions_by_video = defaultdict(list)
    current_video_id = None
    # Regex assumes time is HH:MM:SS, MM:SS, or SS (no milliseconds part)
    timestamp_caption_regex = re.compile(r'^Timestamp:\s*([\d:]+)\s*Caption:\s*(.+)$', re.IGNORECASE)

    with open(ground_truth_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.lower().endswith('.mp4'): # Case-insensitive check for .mp4
                current_video_id = Path(line).stem # Extracts filename without extension
            elif line.lower().startswith('timestamp:') and current_video_id:
                match = timestamp_caption_regex.match(line)
                if match:
                    time_str, caption = match.groups()
                    try:
                        start_seconds = parse_timestamp_to_seconds(time_str)
                        ground_truth_captions_by_video[current_video_id].append((start_seconds, caption))
                    except ValueError as e:
                        print(f"Warning: Skipping invalid time format '{time_str}' for video '{current_video_id}': {e}")
                else:
                    print(f"Warning: Line started with 'Timestamp:' but did not match regex for video '{current_video_id}': {line}")
            elif current_video_id is None and line.lower().startswith('timestamp:'):
                print(f"Warning: Encountered timestamp line before a video ID was specified: {line}")


    ground_truth_segments_by_video = {}
    for video_id, timed_captions in ground_truth_captions_by_video.items():
        if not timed_captions:
            continue
        # Sort by timestamp just in case they are not ordered in the file
        timed_captions.sort(key=lambda x: x[0])
        
        segments = []
        for i, (current_ts, current_caption) in enumerate(timed_captions):
            if i == 0:
                # First segment: defined by its own timestamp for both start and end
                segments.append((current_ts, current_ts, current_caption))
            else:
                # Subsequent segments: start 1s after the previous caption's timestamp,
                # and end at the current caption's timestamp.
                previous_ts = timed_captions[i - 1][0]
                segments.append((previous_ts + 1, current_ts, current_caption))
        ground_truth_segments_by_video[video_id] = segments
    
    return ground_truth_segments_by_video


def match_predictions_to_gt_with_metrics(
    prediction_results_path: str,
    ground_truth_file_path: str,
    output_path: str,
    timing_diff_mode: str = "cumulative"
):
    """
    Matches prediction results with ground truth (GT) segments and computes metrics.

    For each GT segment, it takes the *first* prediction caption whose timestamp
    falls within that GT segment (used for the caption pairing / semantic metrics).
    It also calculates the timing difference (TimDiff), redundancy, and coverage metrics.

    TimDiff (``timing_diff_mode``):
        - ``"cumulative"`` (default): matches the paper definition, where "multiple
          outputs are penalized by cumulative time". For each GT segment the timing
          difference is the *sum* of ``(pred_sec - gt_start_sec)`` over **every**
          prediction that falls inside the segment. A model that emits many redundant
          outputs inside a segment is therefore penalized proportionally, instead of
          only being scored on its first (earliest) output.
        - ``"first_match"``: the legacy behavior, which only scores the first matched
          prediction per segment (``first_matched_time - gt_start_sec``). Kept for
          reproducibility of previously reported numbers.
        In both modes, a segment with no matching prediction receives the maximum
        penalty equal to the full segment length (``gt_end_sec - gt_start_sec``).

    Args:
        prediction_results_path (str): Path to the JSONL file containing prediction results.
        ground_truth_file_path (str): Path to the ground truth text file.
        output_path (str): Path to save the matched pairs and metrics in JSONL format.
        timing_diff_mode (str): Either ``"cumulative"`` (paper definition) or
            ``"first_match"`` (legacy). Defaults to ``"cumulative"``.
    """
    if timing_diff_mode not in ("cumulative", "first_match"):
        raise ValueError(
            f"Unknown timing_diff_mode '{timing_diff_mode}'. "
            "Expected 'cumulative' or 'first_match'."
        )
    ground_truth_segments = _load_ground_truth_segments(ground_truth_file_path)
    output_data_with_metrics = []
    
    all_videos_avg_time_diff = []
    all_videos_avg_redundancy = []
    all_videos_avg_coverage = []

    with open(prediction_results_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                prediction_data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON on line {line_number} in {prediction_results_path}. Skipping.")
                continue
            
            video_id = prediction_data.get("video_id")
            predictions = prediction_data.get("result")

            if not video_id or predictions is None:
                print(f"Warning: Missing 'video_id' or 'result' on line {line_number} in {prediction_results_path}. Skipping.")
                continue

            if video_id not in ground_truth_segments:
                print(f"Warning: Video ID '{video_id}' from predictions not found in ground truth. Skipping.")
                continue
            
            try:
                timed_predictions = sorted(
                    [(parse_timestamp_to_seconds(time_str), caption) for time_str, caption in predictions],
                    key=lambda x: x[0]
                )
            except ValueError as e:
                print(f"Warning: Error parsing prediction times for video '{video_id}': {e}. Skipping video.")
                continue
            except TypeError:
                print(f"Warning: 'result' for video '{video_id}' is not in the expected format (list of [time, caption]). Skipping video.")
                continue

            prediction_idx = 0
            gt_prediction_pairs = []
            segment_match_counts = [] # Count of predictions per GT segment
            segment_time_differences = [] # Time difference for each GT segment

            for gt_start_sec, gt_end_sec, gt_caption in ground_truth_segments[video_id]:
                first_matched_prediction_caption = ""
                predictions_in_segment_count = 0
                first_matched_time_in_segment = None
                cumulative_time_diff_in_segment = 0 # Sum of (pred_sec - gt_start_sec) over all matches
                
                temp_prediction_idx = prediction_idx # Use a temp index for iterating within this segment
                while temp_prediction_idx < len(timed_predictions):
                    pred_sec, pred_caption = timed_predictions[temp_prediction_idx]
                    
                    if gt_start_sec <= pred_sec <= gt_end_sec:
                        if first_matched_time_in_segment is None: # First match in this segment
                            first_matched_time_in_segment = pred_sec
                            first_matched_prediction_caption = pred_caption # Store only the first matched caption
                        cumulative_time_diff_in_segment += pred_sec - gt_start_sec
                        predictions_in_segment_count += 1
                        prediction_idx = temp_prediction_idx + 1 # Crucial: advance the main index
                        temp_prediction_idx += 1
                    elif pred_sec > gt_end_sec:
                        break # Prediction is past this segment
                    else: # pred_sec < gt_start_sec
                        # This prediction is before this segment, advance main index
                        prediction_idx = temp_prediction_idx + 1
                        temp_prediction_idx += 1
                
                # Calculate time difference (TimDiff) for the segment.
                if first_matched_time_in_segment is not None:
                    if timing_diff_mode == "cumulative":
                        # Paper definition: redundant outputs within a segment are
                        # penalized by their cumulative timing offset from the GT onset.
                        time_diff = cumulative_time_diff_in_segment
                    else: # "first_match" (legacy): only the earliest matched output counts
                        time_diff = first_matched_time_in_segment - gt_start_sec
                else:
                    # No prediction landed in this segment: apply the maximum penalty
                    # equal to the full segment length.
                    time_diff = gt_end_sec - gt_start_sec
                segment_time_differences.append(time_diff)
                
                gt_prediction_pairs.append([gt_caption, first_matched_prediction_caption])
                segment_match_counts.append(predictions_in_segment_count)

            # Calculate metrics for the current video
            video_avg_time_diff = sum(segment_time_differences) / len(segment_time_differences) if segment_time_differences else 0
            
            # Redundancy: average absolute difference from 1 match per segment
            if segment_match_counts:
                total_count_redundancy = sum(abs(1 - count) for count in segment_match_counts)
                video_redundancy_metric = total_count_redundancy / len(segment_match_counts)
                
                # Coverage: ratio of segments with at least one match
                segments_with_any_match = sum(1 for count in segment_match_counts if count > 0)
                video_coverage_metric = segments_with_any_match / len(segment_match_counts)
            else: # No segments for this video in GT (should be caught earlier) or no match counts
                video_redundancy_metric = 0
                video_coverage_metric = 0

            all_videos_avg_time_diff.append(video_avg_time_diff)
            all_videos_avg_redundancy.append(video_redundancy_metric)
            all_videos_avg_coverage.append(video_coverage_metric)
            
            output_data_with_metrics.append({
                "video_id": video_id,
                "avg_timediff": video_avg_time_diff,
                "avg_redundancy": video_redundancy_metric, # Per-video redundancy
                "avg_coverage": video_coverage_metric,   # Per-video coverage
                "match_count_per_segment": segment_match_counts, # Renamed for clarity
                "gt_result_pair": gt_prediction_pairs
            })

    # Save per-video results
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in output_data_with_metrics:
            json.dump(item, f_out, ensure_ascii=False)
            f_out.write('\n')
            
    # Calculate and print overall averages
    overall_avg_timediff = sum(all_videos_avg_time_diff) / len(all_videos_avg_time_diff) if all_videos_avg_time_diff else 0
    overall_avg_redundancy = sum(all_videos_avg_redundancy) / len(all_videos_avg_redundancy) if all_videos_avg_redundancy else 0
    overall_avg_coverage = sum(all_videos_avg_coverage) / len(all_videos_avg_coverage) if all_videos_avg_coverage else 0
    
    print(f"Saved matched pairs and metrics for {len(output_data_with_metrics)} videos to {output_path}")
    print(f"Overall Average Time Difference: {overall_avg_timediff:.4f}")
    print(f"Overall Average Redundancy: {overall_avg_redundancy:.4f}")
    print(f"Overall Average Coverage: {overall_avg_coverage:.4f}")


def merge_concatenated_gt_and_prediction_texts(
    matched_pairs_jsonl_path: str,
    merged_texts_jsonl_path: str
):
    """
    Merges ground truth (GT) and prediction texts from a JSONL file of matched pairs.

    It reads a JSONL file where each entry contains "gt_result_pair" (a list of
    [gt_caption, prediction_caption] pairs). It concatenates all GT captions into one
    string and all prediction captions into another string for each video.

    Args:
        matched_pairs_jsonl_path (str): Path to the input JSONL file containing
                                        GT-prediction pairs (e.g., output of
                                        match_predictions_to_ground_truth).
        merged_texts_jsonl_path (str): Path to save the output JSONL file with
                                       merged texts.
    """
    with open(matched_pairs_jsonl_path, 'r', encoding='utf-8') as infile, \
         open(merged_texts_jsonl_path, 'w', encoding='utf-8') as outfile:
        for line_number, line in enumerate(infile, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON on line {line_number} in {matched_pairs_jsonl_path}. Skipping.")
                continue

            video_id = data.get("video_id")
            gt_prediction_pairs = data.get("gt_result_pair")

            if video_id is None or gt_prediction_pairs is None:
                print(f"Warning: Missing 'video_id' or 'gt_result_pair' on line {line_number}. Skipping.")
                continue

            # Join non-empty/non-None captions with a space
            concatenated_gt_text = " ".join(pair[0] for pair in gt_prediction_pairs if pair[0])
            concatenated_prediction_text = " ".join(pair[1] for pair in gt_prediction_pairs if pair[1])

            merged_entry = {
                "video_id": video_id,
                "text_pair": [concatenated_gt_text, concatenated_prediction_text] # Stored as a list/tuple
            }
            outfile.write(json.dumps(merged_entry, ensure_ascii=False) + "\n")
    print(f"Saved merged GT and prediction texts to {merged_texts_jsonl_path}")
    
if __name__ == "__main__":
    
    match_predictions_to_gt_with_metrics(
        "/your_path/online_inference_result.jsonl",
        "/your_path/raw_text_input_file.txt",
        "/your_path/matched_result.jsonl"
    )
    
    merge_concatenated_gt_and_prediction_texts(
        "/your_path/matched_result.jsonl",
        "/your_path/merged_result.jsonl"
    )
