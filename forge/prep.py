"""Step 1: Prep the source.

Trim, normalize to -3dB, convert to 44.1kHz / 16-bit WAV.
MIDI detectors hate MP3 artifacts — always work from clean WAV.
"""
import numpy as np
from pathlib import Path


def prep_source(input_path: str, out_dir: str,
                normalize_db: float = -3.0,
                target_sr: int = 44100) -> str:
    """
    Prep an MP3/WAV for stem separation.
    Returns path to cleaned WAV.
    """
    import librosa
    import soundfile as sf

    y, sr = librosa.load(input_path, sr=target_sr, mono=True)

    # Normalize to -3dB peak
    peak = np.max(np.abs(y))
    if peak > 0:
        target_peak = 10 ** (normalize_db / 20)
        y = y * (target_peak / peak)

    out_path = str(Path(out_dir) / f"{Path(input_path).stem}_prepped.wav")
    sf.write(out_path, y, target_sr, subtype='PCM_16')
    return out_path
