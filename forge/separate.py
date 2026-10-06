"""Stem separation via Demucs."""
import os
from pathlib import Path

STEMS = ["vocals", "drums", "bass", "other"]
MODEL = "htdemucs"


def separate_stems(audio_path: str, out_dir: str, on_stage=None) -> dict[str, str]:
    """
    Separate audio into stems using Demucs.
    Returns {stem_name: wav_path}.
    """
    import demucs.separate

    if on_stage:
        on_stage("Separating stems (Demucs)…")

    fmt = "--mp3" if audio_path.lower().endswith(".mp3") else "--wav"
    demucs.separate.main([
        fmt, "-n", MODEL, "--out", out_dir, audio_path,
    ])

    track = Path(audio_path).stem
    stem_dir = Path(out_dir) / MODEL / track
    paths = {}
    for stem in STEMS:
        p = stem_dir / f"{stem}.wav"
        if not p.exists():
            raise FileNotFoundError(f"Demucs missing output: {p}")
        paths[stem] = str(p)
    return paths
