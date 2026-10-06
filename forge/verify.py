"""Step 5 (verification): Check MIDI matches stem audio.

Synthesize MIDI back to audio, compare chroma (pitch) + onset envelope (timing).
"""
import numpy as np


def verify_midi(midi_path: str, stem_wav_path: str) -> dict:
    import librosa
    import pretty_midi

    try:
        y_orig, sr = librosa.load(stem_wav_path, sr=22050, mono=True)
        pm = pretty_midi.PrettyMIDI(midi_path)
        y_synth = pm.synthesize(sr=sr)

        n = min(len(y_orig), len(y_synth))
        if n < sr:
            return {"score": 0.5, "detail": "too short"}

        y_orig, y_synth = y_orig[:n], y_synth[:n]

        c1 = librosa.feature.chroma_cqt(y=y_orig, sr=sr)
        c2 = librosa.feature.chroma_cqt(y=y_synth, sr=sr)
        f = min(c1.shape[1], c2.shape[1])
        r = np.corrcoef(c1[:, :f].flatten(), c2[:, :f].flatten())[0, 1]
        pitch = float(max(0, r)) if not np.isnan(r) else 0.5

        o1 = librosa.onset.onset_strength(y=y_orig, sr=sr)
        o2 = librosa.onset.onset_strength(y=y_synth, sr=sr)
        f = min(len(o1), len(o2))
        r = np.corrcoef(o1[:f], o2[:f])[0, 1]
        timing = float(max(0, r)) if not np.isnan(r) else 0.5

        score = 0.6 * pitch + 0.4 * timing
        return {"score": round(score, 3),
                "detail": f"pitch={pitch:.2f} timing={timing:.2f}"}
    except Exception as e:
        return {"score": 0.0, "detail": f"error: {e}"}


def check_clean_midi(midi_path: str, stem: str) -> list[str]:
    """
    Run the Clean MIDI Checklist. Returns list of issues (empty = clean).
    - No overlapping notes in bass
    - Velocity range 60-110
    - Drum notes no shorter than 1/32nd
    - Polyphonic < 6 notes at a time
    """
    import pretty_midi

    issues = []
    pm = pretty_midi.PrettyMIDI(midi_path)
    if not pm.instruments:
        return ["no instruments in MIDI"]

    inst = pm.instruments[0]
    notes = sorted(inst.notes, key=lambda n: n.start)

    # Velocity range
    vels = [n.velocity for n in notes]
    if vels:
        if min(vels) < 60 or max(vels) > 110:
            issues.append(f"velocity out of 60-110 range ({min(vels)}-{max(vels)})")
        if len(set(vels)) < 3:
            issues.append("velocity too flat — needs human variation")

    if stem == "bass":
        # No overlapping notes
        for i in range(len(notes) - 1):
            if notes[i].end > notes[i + 1].start:
                issues.append(f"overlapping bass notes at {notes[i].start:.2f}s")
                break

    if stem == "drums":
        # No notes shorter than 1/32nd (~62ms at 120bpm)
        short = [n for n in notes if (n.end - n.start) < 0.062]
        if short:
            issues.append(f"{len(short)} drum notes shorter than 1/32nd")

    if stem == "other":
        # Max 6 simultaneous
        # Simple check: count notes active at each note start
        for n in notes:
            simultaneous = sum(1 for m in notes
                               if m.start <= n.start < m.end)
            if simultaneous > 6:
                issues.append(f"polyphony >6 at {n.start:.2f}s")
                break

    return issues
