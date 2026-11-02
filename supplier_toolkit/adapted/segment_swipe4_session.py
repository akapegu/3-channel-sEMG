"""Create cue-bounded swipe/rest derivatives without altering the source."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

PARTITIONS = ("train", "validation", "test")
GESTURES = ("down", "left", "right", "top")
CLASSES = ("rest", *GESTURES)
EXPECTED_REPS = 8


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def pair_labeled_events(events, n_samples):
    pending = {}
    intervals = []
    for event in events:
        if event.get("event") not in ("label_start", "label_stop"):
            continue
        phase = event.get("phase_index")
        if phase is None:
            raise ValueError("A labeled cue event has no phase_index")
        phase = int(phase)
        if event["event"] == "label_start":
            if phase in pending:
                raise ValueError(f"Duplicate label_start for phase {phase}")
            label = event.get("label")
            partition = event.get("partition")
            group = event.get("acquisition_group")
            if label not in CLASSES or partition not in PARTITIONS or not group:
                raise ValueError(f"Invalid label/partition/group for phase {phase}: {label}/{partition}/{group}")
            pending[phase] = event
            continue
        if phase not in pending:
            raise ValueError(f"label_stop without label_start for phase {phase}")
        start = pending.pop(phase)
        if (event.get("label") != start.get("label") or
                event.get("partition") != start.get("partition") or
                event.get("acquisition_group") != start.get("acquisition_group")):
            raise ValueError(f"Cue metadata changed within phase {phase}")
        a, b = int(start["sample_index"]), int(event["sample_index"])
        if not (0 <= a < b <= n_samples):
            raise ValueError(f"Invalid sample range [{a}, {b}) for phase {phase} of {n_samples}")
        intervals.append({
            "partition": start["partition"], "group": start["acquisition_group"],
            "label": start["label"], "source_start": a, "source_stop": b,
            "phase_index": phase, "task_id": start.get("task_id"),
            "gesture": start.get("gesture"), "repetition": start.get("repetition"),
            "cue_start_host_unix_ns": start.get("host_unix_ns"),
            "cue_stop_host_unix_ns": event.get("host_unix_ns"),
        })
    if pending:
        raise ValueError(f"Unclosed labeled intervals: {sorted(pending)}")
    intervals.sort(key=lambda x: x["source_start"])
    for left, right in zip(intervals, intervals[1:]):
        if right["source_start"] < left["source_stop"]:
            raise ValueError(f"Overlapping source intervals: {left} / {right}")
    return intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--cue-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--compliance-confirmed", action="store_true",
                        help="Record the user's confirmation that all cues were followed.")
    args = parser.parse_args()

    source = args.source_csv.resolve()
    cue_path = args.cue_json.resolve()
    output_dir = args.output_dir.resolve()
    cue = json.loads(cue_path.read_text())
    if cue.get("status") != "complete":
        raise SystemExit("Cue status is incomplete; refusing to segment.")
    if not args.compliance_confirmed:
        raise SystemExit("User cue-compliance confirmation is required; no outputs written.")
    plan = cue["plan"]
    if tuple(plan.get("partitions", [])) != PARTITIONS:
        raise SystemExit(f"Expected chronological partitions {PARTITIONS}")

    header, rows = read_csv(source)
    if not rows or not {"counter_dev1", "ch1", "ch2", "ch3", "accel_x", "accel_y", "accel_z"}.issubset(header):
        raise SystemExit(f"Unexpected source CSV schema: {header}")
    intervals = pair_labeled_events(cue.get("events", []), len(rows))

    # Count active prompt cues separately from background/rest intervals.
    cue_reps = Counter((e.get("partition"), e.get("gesture"))
                       for e in cue.get("events", [])
                       if e.get("event") == "cue_start" and
                       e.get("gesture") in GESTURES and e.get("label") == e.get("gesture"))
    expected = Counter({(part, gesture): EXPECTED_REPS
                        for part in PARTITIONS for gesture in GESTURES})
    if cue_reps != expected:
        raise SystemExit(f"Unexpected cue repetition counts: {dict(cue_reps)}")

    # The initial five-second settle was explicitly cued as relaxed. Attach it
    # to the TRAIN REST partition so it is represented as no-command data.
    session_start = next((e for e in cue["events"] if e.get("event") == "session_start"), None)
    settle_stop = next((e for e in cue["events"] if e.get("event") == "session_settle_stop"), None)
    if not session_start or not settle_stop:
        raise SystemExit("Missing explicit initial relaxed-settle markers")
    a, b = int(session_start["sample_index"]), int(settle_stop["sample_index"])
    if not (0 <= a < b <= len(rows)):
        raise SystemExit(f"Invalid settle interval [{a}, {b})")
    intervals.append({"partition": "train", "group": f"{cue['acquisition_id']}_train",
                      "label": "rest", "source_start": a, "source_stop": b,
                      "phase_index": None, "task_id": None, "gesture": None,
                      "repetition": None, "cue_start_host_unix_ns": session_start.get("host_unix_ns"),
                      "cue_stop_host_unix_ns": settle_stop.get("host_unix_ns"),
                      "note": "initial relaxed settle, explicitly instructed by the cue page"})
    intervals.sort(key=lambda x: x["source_start"])
    for left, right in zip(intervals, intervals[1:]):
        if right["source_start"] < left["source_stop"]:
            raise ValueError(f"Overlapping source intervals: {left} / {right}")

    grouped = {(part, label): [] for part in PARTITIONS for label in CLASSES}
    for interval in intervals:
        grouped[(interval["partition"], interval["label"])].append(interval)
    missing = [f"{part}/{label}" for part in PARTITIONS for label in CLASSES if not grouped[(part, label)]]
    if missing:
        raise SystemExit(f"Missing labeled data: {missing}")

    # Validate processing profile before creating any derivatives.
    processing_path = source.with_suffix(".processing.json")
    processing = json.loads(processing_path.read_text())
    if processing.get("sampling_rate") != 1000 or processing.get("preprocessing_profile") != cue.get("preprocessing_profile"):
        raise SystemExit("Source processing rate/profile does not match the cue protocol.")
    if not processing.get("raw_packet_files"):
        raise SystemExit("Processing sidecar does not point to the original raw packet log.")

    source_hash = sha256(source)
    raw_paths = processing["raw_packet_files"]
    manifest_files = []
    summary_files = []
    pending_outputs = []
    for partition in PARTITIONS:
        for label in CLASSES:
            segments = sorted(grouped[(partition, label)], key=lambda x: x["source_start"])
            chunks = [rows[x["source_start"]:x["source_stop"]] for x in segments]
            if any(not chunk for chunk in chunks):
                raise ValueError(f"Empty interval in {partition}/{label}")
            combined = [row for chunk in chunks for row in chunk]
            relpath = Path(partition) / f"{label}.csv"
            out = output_dir / relpath
            sidecar = out.with_suffix(".segments.json")
            if out.exists() or sidecar.exists():
                raise FileExistsError(f"Refusing to overwrite existing derivative: {out}")
            pending_outputs.append((out, sidecar, relpath, partition, label, segments, combined))

    for out, sidecar, relpath, partition, label, segments, combined in pending_outputs:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(combined)
        source_ranges = [[x["source_start"], x["source_stop"]] for x in segments]
        local_ranges = []
        local_cursor = 0
        for segment in segments:
            length = segment["source_stop"] - segment["source_start"]
            local_ranges.append([local_cursor, local_cursor + length])
            local_cursor += length
        metadata = {
            "derived_file": str(out), "derived_sha256": sha256(out),
            "label": label, "partition": partition,
            "acquisition_group": f"{cue['acquisition_id']}_{partition}",
            "user_cue_compliance_confirmed": True,
            "source_csv": str(source), "source_csv_sha256": source_hash,
            "source_sample_ranges": source_ranges,
            "derived_segment_ranges": local_ranges,
            "source_interval_metadata": segments,
            "source_raw_packet_files": raw_paths,
            "cue_json": str(cue_path), "sampling_rate_configured_hz": 1000,
            "preprocessing_profile": cue["preprocessing_profile"],
            "rows": len(combined), "segment_count": len(segments),
            "training_policy": "train partition may fit; validation may select; test partition is held out",
            "limitations": [
                "All partitions come from one continuous acquisition and one wearing/session.",
                "Direction-cue intervals include visual-dispatch/reaction timing and the full 1.5 s prompt; human movement onset is not measured independently.",
                "The first 5 s relaxed-settle interval is assigned to TRAIN/REST; trailing unsignaled save time is excluded.",
            ],
        }
        sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
        item = {"path": str((output_dir / relpath).resolve()), "partition": partition,
                "acquisition_group": f"{cue['acquisition_id']}_{partition}", "label": label,
                "rows": len(combined), "segments": len(segments),
                "source_sample_ranges": source_ranges, "derived_segment_ranges": local_ranges,
                "include_in_model_fit": partition == "train",
                "include_in_model_selection": partition in ("train", "validation"),
                "use_as_locked_test": partition == "test"}
        manifest_files.append(item)
        summary_files.append(item)

    manifest_path = args.manifest.resolve()
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "protocol": "swipe4 one-go cue segmentation",
        "source_csv": str(source), "source_csv_sha256": source_hash,
        "user_cue_compliance_confirmed": True,
        "split_rule": "Partition assigned before feature-window generation; never randomly split windows.",
        "limitation": "train/validation/test are chronological partitions from one session and one wearing; not independent rewear/day groups.",
        "files": manifest_files,
    }, indent=2) + "\n")
    result = {
        "status": "segmented_pending_model_training",
        "source_csv": str(source), "source_csv_sha256": source_hash,
        "source_untouched": True, "source_rows": len(rows),
        "cue_compliance_confirmed": True,
        "cue_repetitions_per_gesture_partition": EXPECTED_REPS,
        "derived_files": summary_files, "manifest": str(manifest_path),
        "note": "Cue-bounded derivatives have been created. No model has been fit. Keep TEST untouched until model/feature settings are frozen.",
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
