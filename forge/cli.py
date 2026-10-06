#!/usr/bin/env python3
"""MIDI Forge CLI — Turn any MP3 into clean MIDI stems for sampling.

Usage:
    python -m forge.cli input.mp3 -o ./output

Pipeline:
    1. Prep (normalize, WAV convert)
    2. Two-pass Demucs separation
    3. Per-stem cleaning (EQ, gate)
    4. Per-stem MIDI transcription
    5. Verification + clean MIDI checklist
"""
import argparse
import os
import sys
import tempfile

from .pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser(description="MIDI Forge — MP3 to MIDI stems")
    ap.add_argument("input", help="Input MP3/WAV file")
    ap.add_argument("-o", "--output", default="./midi_forge_out",
                    help="Output directory")
    ap.add_argument("--vocals-midi", action="store_true",
                    help="Also transcribe vocals (default: skip)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix="forge_")

    print(f"🔨 Forging MIDI from: {args.input}")
    print(f"   Output: {args.output}")
    print()

    result = run_pipeline(
        args.input, work_dir,
        on_stage=lambda s: print(f"  → {s}"),
        skip_vocals_midi=not args.vocals_midi,
    )

    # Copy outputs to destination
    import shutil
    for stem, p in result["midi"].items():
        dst = os.path.join(args.output, f"{stem}.mid")
        shutil.copy(p, dst)
        print(f"  ✓ {dst}")
    shutil.copy(result["all_stems"], os.path.join(args.output, "all_stems.mid"))
    shutil.copy(result["zip"], os.path.join(args.output,
                os.path.basename(result["zip"])))
    shutil.copy(result["report"], os.path.join(args.output, "flip_report.md"))

    print()
    print(f"BPM: {result['bpm']}")
    print("Verification scores:")
    for stem, v in result["verification"].items():
        print(f"  {stem}: {v['score']} ({v['detail']})")
    print()
    print("Import all MIDIs at the ORIGINAL BPM.")
    print("Quantize to 0% first — see where it drags, that's your swing.")
    print("Then: bass/drums → 85% MPC-16 swing, keys → 50-60%.")


if __name__ == "__main__":
    main()
