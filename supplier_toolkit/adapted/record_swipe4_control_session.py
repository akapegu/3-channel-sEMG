"""Record one guided four-swipe re-wear session with tilt-only negatives."""
import argparse
import asyncio
import json
from pathlib import Path

from bleak import BleakScanner
import record_gesture as recorder
from visual_swipe_control_session import run_session


async def main(args):
    adapted_dir = Path(__file__).resolve().parent
    recorder.OUTPUT_ROOT = str(adapted_dir / "training_data")
    recorder.SAMPLING_RATE = 1000
    recorder.VISUAL_CUES = True
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    session_dir = adapted_dir / "training_data" / plan["subject"] / plan["session"] / "continuous"
    session_dir.mkdir(parents=True, exist_ok=True)
    trial = len(list(session_dir.glob("swipe4_control_trial*_merged_*.csv"))) + 1

    print("Scanning for the NPG-Lite 27:BC band (not connecting yet)...", flush=True)
    found = await BleakScanner.discover(timeout=10, return_adv=True)
    matches = []
    for device, adv in found.values():
        name = (adv.local_name or device.name or "").lower()
        address = (device.address or "").lower()
        if "npg-lite-band" in name and ("27:bc" in name or "27:bc" in address):
            matches.append(device)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one NPG-Lite 27:BC band; found {len(matches)}")
    device, role, channels = recorder.resolve_devices(matches)[0]
    device_recorder = recorder.DeviceRecorder(device, out_path=None, channels=channels, role=role)
    cue_path = Path(device_recorder.raw_path).with_suffix(".session_cues.json")
    device_recorder.visual_cue_path = str(cue_path)
    print(f"Band found: {device_recorder.name} ({device.address}). Not connected yet.", flush=True)

    error, cue_result = None, None
    try:
        async def connect_and_start():
            await device_recorder.connect()
            await device_recorder.start_streaming()

        cue_result = await run_session([device_recorder], cue_path, plan_path,
                                       connect_and_start=connect_and_start)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"Session stopped; preserving received packets: {error}", flush=True)
    finally:
        await device_recorder.stop_and_disconnect()

    saved = recorder.merge_and_save(str(session_dir), "swipe4_control", trial,
                                    [device_recorder], reps=[], mode="reps")
    report = {"session": plan["session"], "status": device_recorder.visual_cue_status,
              "continuous_source": saved["csv_path"] if saved else None,
              "num_samples": saved["num_samples"] if saved else 0,
              "raw_packet_file": device_recorder.raw_path,
              "cue_file": str(cue_path), "cue_result": cue_result, "error": error}
    report_path = (Path(saved["csv_path"]).with_suffix(".capture.json") if saved
                   else session_dir / f"swipe4_control_trial{trial}_no_samples.capture.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("CAPTURE SUMMARY: " + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=(
        Path(__file__).resolve().parents[2] / "analysis" / "exp015_swipe4" /
        "robust_session_template.json"))
    asyncio.run(main(parser.parse_args()))
