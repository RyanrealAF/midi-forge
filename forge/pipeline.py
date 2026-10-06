"""Pipeline orchestrator: separate → analyze → transcribe → verify → retry."""
import gc
import os
import shutil
from pathlib import Path

from .separate import separate_stems, STEMS
from .analyze import analyze_stem
from .transcribe import transcribe_stem
from .verify import verify_midi

THRESHOLD = 0.65
MAX_ATTEMPTS = 3


def run_pipeline(audio_path: str, work_dir: str, on_stage=None) -> dict:
    """
    Full pipeline. Returns:
    {
        "track": str,
        "midi": {stem: path},
        "analysis": {stem: dict},
        "verification": {stem: dict},
        "report": path,
        "zip": path,
    }
    """
    import torch

    track = Path(audio_path).stem
    stage = on_stage or (lambda s: None)

    # 1. Separate
    stem_paths = separate_stems(audio_path, os.path.join(work_dir, "stems"), stage)

    # 2. Load transcription model once
    stage("Loading transcription model…")
    from basic_pitch.inference import Model
    from basic_pitch import ICASSP_2022_MODEL_PATH
    model = Model(ICASSP_2022_MODEL_PATH)

    # 3. Per-stem: analyze → transcribe → verify → retry
    midi_dir = os.path.join(work_dir, "midi")
    os.makedirs(midi_dir, exist_ok=True)

    midi, analysis, verification = {}, {}, {}
    for stem in STEMS:
        stage(f"Analyzing {stem}…")
        analysis[stem] = analyze_stem(stem_paths[stem])

        best_path, best_v, best_score = None, None, -1
        for attempt in range(MAX_ATTEMPTS):
            stage(f"Transcribing {stem} (attempt {attempt + 1})…")
            tmp = os.path.join(midi_dir, f"{stem}_a{attempt}.mid")
            transcribe_stem(stem, stem_paths[stem], tmp, model,
                            analysis[stem], attempt)

            stage(f"Verifying {stem}…")
            v = verify_midi(tmp, stem_paths[stem])
            if v["score"] > best_score:
                best_score, best_path, best_v = v["score"], tmp, v
            if v["score"] >= THRESHOLD:
                break

        final = os.path.join(midi_dir, f"{stem}.mid")
        shutil.copy(best_path, final)
        midi[stem] = final
        verification[stem] = best_v

    # 4. Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Report
    stage("Writing verification report…")
    lines = [f"# MIDI Forge — Verification Report: {track}", ""]
    for stem in STEMS:
        a, v = analysis[stem], verification[stem]
        ok = "✓ PASS" if v["score"] >= THRESHOLD else "⚠ REVIEW"
        lines += [
            f"## {stem} — {ok}",
            f"- Score: **{v['score']}** ({v['detail']})",
            f"- BPM: {a['bpm']} | Key: {a['key']} | Energy: {a['energy_db']} dB",
            f"- Pitch: {a['pitch_min_hz']}–{a['pitch_max_hz']} Hz | "
            f"Onsets/s: {a['onset_density']}",
            "",
        ]
    report_path = os.path.join(work_dir, "verification_report.md")
    Path(report_path).write_text("\n".join(lines))

    # 6. ZIP
    import zipfile
    stage("Packaging…")
    zip_path = os.path.join(work_dir, f"{track}_midi.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for stem, p in midi.items():
            zf.write(p, arcname=f"{stem}.mid")

    return {
        "track": track, "midi": midi, "analysis": analysis,
        "verification": verification, "report": report_path, "zip": zip_path,
    }
