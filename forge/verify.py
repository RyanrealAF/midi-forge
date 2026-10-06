"""Verify a MIDI file matches its source stem audio.

Synthesizes the MIDI back to audio, then compares:
- Pitch: chroma feature correlation
- Timing: onset envelope correlation
Returns a 0-1 score plus breakdown.
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
        if n < sr:  # under 1 second — can't verify
            return {"score": 0.5, "pitch": 0.5, "timing": 0.5,
                    "detail": "too short to verify"}

        y_orig, y_synth = y_orig[:n], y_synth[:n]

        # Pitch: chroma correlation
        c1 = librosa.feature.chroma_cqt(y=y_orig, sr=sr)
        c2 = librosa.feature.chroma_cqt(y=y_synth, sr=sr)
        f = min(c1.shape[1], c2.shape[1])
        if f < 10:
            pitch = 0.5
        else:
            r = np.corrcoef(c1[:, :f].flatten(), c2[:, :f].flatten())[0, 1]
            pitch = float(max(0, r)) if not np.isnan(r) else 0.5

        # Timing: onset envelope correlation
        o1 = librosa.onset.onset_strength(y=y_orig, sr=sr)
        o2 = librosa.onset.onset_strength(y=y_synth, sr=sr)
        f = min(len(o1), len(o2))
        if f < 10:
            timing = 0.5
        else:
            r = np.corrcoef(o1[:f], o2[:f])[0, 1]
            timing = float(max(0, r)) if not np.isnan(r) else 0.5

        score = 0.6 * pitch + 0.4 * timing
        return {
            "score": round(score, 3),
            "pitch": round(pitch, 3),
            "timing": round(timing, 3),
            "detail": f"pitch={pitch:.2f} timing={timing:.2f}",
        }
    except Exception as e:
        return {"score": 0.0, "pitch": 0.0, "timing": 0.0,
                "detail": f"error: {e}"}
