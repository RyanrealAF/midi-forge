"""Per-stem audio analysis to guide transcription."""
import numpy as np


def analyze_stem(wav_path: str) -> dict:
    """
    Analyze a stem: BPM, key, pitch range, energy, onset density, brightness.
    Returns a dict of measurements used to adapt transcription parameters.
    """
    import librosa

    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    duration = len(y) / sr
    out = {"duration_sec": round(duration, 2)}

    # BPM
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        out["bpm"] = round(float(tempo), 1)
    except Exception:
        out["bpm"] = 0.0

    # Key (Krumhansl)
    out["key"] = "unknown"
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
        major = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minor = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        best, best_key = -1, "unknown"
        for i in range(12):
            for prof, suf in [(major," major"), (minor," minor")]:
                s = np.corrcoef(chroma, np.roll(prof, i))[0,1]
                if s > best:
                    best, best_key = s, names[i] + suf
        out["key"] = best_key
    except Exception:
        pass

    # Pitch range
    out["pitch_min_hz"] = 0.0
    out["pitch_max_hz"] = 0.0
    try:
        f0, voiced, _ = librosa.pyin(y, fmin=30, fmax=4000, sr=sr)
        vf = f0[voiced]
        if len(vf) > 10:
            out["pitch_min_hz"] = round(float(np.percentile(vf, 5)), 1)
            out["pitch_max_hz"] = round(float(np.percentile(vf, 95)), 1)
    except Exception:
        pass

    # Energy
    try:
        rms = librosa.feature.rms(y=y)
        out["energy_db"] = round(float(20 * np.log10(rms.mean() + 1e-10)), 1)
    except Exception:
        out["energy_db"] = -60.0

    # Onset density
    try:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        out["onset_density"] = round(len(onsets) / max(duration, 0.1), 2)
    except Exception:
        out["onset_density"] = 0.0

    # Brightness
    try:
        out["brightness_hz"] = round(float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()), 0)
    except Exception:
        out["brightness_hz"] = 0.0

    return out
