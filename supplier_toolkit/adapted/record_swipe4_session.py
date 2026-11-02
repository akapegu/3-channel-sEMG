"""Record one continuous visual-cue session for four directional swipes."""
import argparse
import asyncio
import json
from pathlib import Path

from bleak import BleakScanner
import record_gesture as recorder
from visual_swipe_session import run_session


async def main(args):
    root = Path(__file__).resolve().parent
    recorder.OUTPUT_ROOT = str(root / "training_data")
    recorder.SAMPLING_RATE = 1000
    recorder.VISUAL_CUES = True
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    subject_dir = Path(recorder.OUTPUT_ROOT) / plan["subject"] / "continuous"
    subject_dir.mkdir(parents=True, exist_ok=True)
    trial = len(list(subject_dir.glob("swipe4_session_trial*_merged_*.csv"))) + 1

    print("Scanning for the NPG-Lite band (no BLE connection yet)...", flush=True)
    found = await BleakScanner.discover(timeout=10, return_adv=True)
    matches = []
    for device, adv in found.values():
        name = (adv.local_name or device.name or "").lower()
        address = (device.address or "").lower()
        if "npg-lite-band" in name and ("27:bc" in name or "27:bc" in address):
            matches.append(device)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one NPG-Lite 27:BC band; found {len(matches)}")
    resolved = recorder.resolve_devices(matches)
    if len(resolved) != 1:
        raise RuntimeError(f"Expected one band; resolved {len(resolved)}")
    device, role, channels = resolved[0]
    device_recorder = recorder.DeviceRecorder(device, out_path=None, channels=channels, role=role)
    cue_path = Path(device_recorder.raw_path).with_suffix(".session_cues.json")
    device_recorder.visual_cue_path = str(cue_path)
    print(f"Band found: {device_recorder.name}. Not connected yet.", flush=True)

    error = None
    try:
        async def connect_and_start():
            await device_recorder.connect()
            await device_recorder.start_streaming()

        cue_result = await run_session([device_recorder], cue_path, plan_path,
                                       connect_and_start=connect_and_start)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"Session stopped; preserving any packets received: {error}", flush=True)
        cue_result = None
    finally:
        await device_recorder.stop_and_disconnect()

    saved = recorder.merge_and_save(str(subject_dir), "swipe4_session", trial,
                                    [device_recorder], reps=[], mode="reps")
    if saved is None:
        report = {"status": "no_samples", "continuous_source": None,
                  "raw_packet_file": device_recorder.raw_path,
                  "cue_file": str(cue_path), "error": error}
    else:
        report = {"status": device_recorder.visual_cue_status,
                  "continuous_source": saved["csv_path"],
                  "num_samples": saved["num_samples"],
                  "raw_packet_file": device_recorder.raw_path,
                  "cue_file": str(cue_path), "cue_result": cue_result,
                  "error": error}
    print("CAPTURE SUMMARY: " + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path,
                        default=Path(__file__).resolve().parent / "training_data" / "swipe4_session_plan.json")
    asyncio.run(main(parser.parse_args()))
