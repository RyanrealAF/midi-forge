"""Pipeline: prep → separate (2-pass) → clean → transcribe (per-stem) → verify."""
import os
import shutil
import zipfile
from pathlib import Path

from .prep import prep_source
from .separate import separate_two_pass, STEMS
from .clean import clean_stem
from .transcribe import transcribe_drums, transcribe_mono, transcribe_poly
from .verify import verify_midi, check_clean_midi


def detect_bpm(wav_path: str) -> float:
    """Detect BPM from audio."""
    import librosa
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return round(float(tempo), 1)


def run_pipeline(input_path: str, work_dir: str, on_stage=None,
                 skip_vocals_midi: bool = True) -> dict:
    """
    Full offline pipeline.
    Returns {track, bpm, midi: {stem: path}, verification: {stem: dict},
             issues: {stem: [str]}, zip, all_stems_mid}.
    """
    stage = on_stage or (lambda s: print(f"  {s}"))
    track = Path(input_path).stem

    # 1. Prep
    stage("Prepping source (normalize, convert to WAV)…")
    prepped = prep_source(input_path, work_dir)
    bpm = detect_bpm(prepped)
    stage(f"Detected BPM: {bpm}")

    # 2. Two-pass separation
    stems = separate_two_pass(prepped, os.path.join(work_dir, "stems"), stage)

    # 3. Clean + 4. Transcribe per stem
    midi_dir = os.path.join(work_dir, "midi")
    os.makedirs(midi_dir, exist_ok=True)

    midi_paths, verifications, all_issues = {}, {}, {}

    for stem in STEMS:
        if stem == "vocals" and skip_vocals_midi:
            stage("Skipping vocals MIDI (muted per workflow)…")
            continue

        stage(f"Cleaning {stem}…")
        cleaned = clean_stem(stem, stems[stem],
                             os.path.join(work_dir, "cleaned"))

        stage(f"Transcribing {stem}…")
        midi_out = os.path.join(midi_dir, f"{stem}.mid")

        if stem == "drums":
            transcribe_drums(cleaned, midi_out)
        elif stem == "bass":
            transcribe_mono(cleaned, midi_out, fmin=30, fmax=300)
        elif stem == "vocals":
            transcribe_mono(cleaned, midi_out, fmin=80, fmax=1100)
        else:  # other
            transcribe_poly(cleaned, midi_out, sensitivity=0.72)

        # 5. Verify
        stage(f"Verifying {stem}…")
        verifications[stem] = verify_midi(midi_out, cleaned)

        # 6. Clean MIDI checklist
        issues = check_clean_midi(midi_out, stem)
        all_issues[stem] = issues
        if issues:
            stage(f"  ⚠ {stem}: {'; '.join(issues)}")

        midi_paths[stem] = midi_out

    # 7. All-stems MIDI (merge)
    stage("Merging all-stems MIDI…")
    import pretty_midi
    combined = pretty_midi.PrettyMIDI()
    for stem, mp in midi_paths.items():
        pm = pretty_midi.PrettyMIDI(mp)
        for inst in pm.instruments:
            combined.instruments.append(inst)
    all_path = os.path.join(midi_dir, "all_stems.mid")
    combined.write(all_path)

    # 8. ZIP bundle
    stage("Packaging…")
    zip_path = os.path.join(work_dir, f"{track}_midi_flip.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for stem, p in midi_paths.items():
            zf.write(p, arcname=f"{stem}.mid")
        zf.write(all_path, arcname="all_stems.mid")

    # 9. Report
    report = [f"# MIDI Flip: {track}", f"BPM: {bpm}", ""]
    for stem in midi_paths:
        v = verifications[stem]
        iss = all_issues[stem]
        report.append(f"## {stem} — score {v['score']} ({v['detail']})")
        if iss:
            report.append("Issues: " + "; ".join(iss))
        report.append("")
    report_path = os.path.join(work_dir, "flip_report.md")
    Path(report_path).write_text("\n".join(report))

    return {
        "track": track, "bpm": bpm,
        "midi": midi_paths, "all_stems": all_path,
        "verification": verifications, "issues": all_issues,
        "zip": zip_path, "report": report_path,
    }
