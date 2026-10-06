"""Step 3: Clean each stem for conversion.

Bass:    low-pass at 250Hz, gate silence
Drums:   high-pass at 80Hz
Other:   band focus 300Hz–5kHz (keep a little grit — don't over-denoise)
Vocals:  leave as-is (or mute if not needed)

Export each as mono WAV — MIDI converters track mono 2x better.
"""
import numpy as np
from pathlib import Path


def clean_stem(stem: str, wav_path: str, out_dir: str) -> str:
    """
    Apply stem-specific cleaning. Returns path to cleaned mono WAV.
    """
    import librosa
    import soundfile as sf
    from scipy.signal import butter, filtfilt

    y, sr = librosa.load(wav_path, sr=44100, mono=True)

    def lowpass(data, cutoff, order=4):
        b, a = butter(order, cutoff / (sr / 2), btype='low')
        return filtfilt(b, a, data)

    def highpass(data, cutoff, order=4):
        b, a = butter(order, cutoff / (sr / 2), btype='high')
        return filtfilt(b, a, data)

    def bandpass(data, lo, hi, order=4):
        b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype='band')
        return filtfilt(b, a, data)

    def gate(data, threshold_db=-50):
        """Zero out sections below threshold."""
        rms = librosa.feature.rms(y=data, frame_length=2048, hop_length=512)[0]
        threshold = 10 ** (threshold_db / 20)
        # Expand mask to sample level
        mask = np.repeat(rms > threshold, 512)[:len(data)]
        return data * mask

    if stem == "bass":
        y = lowpass(y, 250)
        y = gate(y, threshold_db=-50)
    elif stem == "drums":
        y = highpass(y, 80)
        # Gentle transient emphasis (simple: mix with differentiated)
        # (Full transient shaper would need a dedicated lib)
    elif stem == "other":
        # Band focus but keep grit — light touch
        y = bandpass(y, 300, 5000)
    # vocals: leave as-is

    out_path = str(Path(out_dir) / f"{stem}_clean.wav")
    sf.write(out_path, y, sr, subtype='PCM_16')
    return out_path
