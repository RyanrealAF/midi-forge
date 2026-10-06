"""Step 2: Two-pass stem separation.

Don't use "all in one." Run twice for max cleanliness:

Pass 1: Vocals / Instrumental (Demucs htdemucs_6s)
Pass 2: Instrumental → Drums / Bass / Other

This gives 4 clean stems: drums, bass, other, vocals.
"""
import os
from pathlib import Path

STEMS = ["vocals", "drums", "bass", "other"]


def separate_two_pass(wav_path: str, out_dir: str, on_stage=None) -> dict[str, str]:
    """
    Two-pass Demucs separation.
    Returns {stem: wav_path} for drums, bass, other, vocals.
    """
    import demucs.separate

    stage = on_stage or (lambda s: None)
    track = Path(wav_path).stem

    # Pass 1: vocals / instrumental
    stage("Pass 1: separating vocals / instrumental…")
    pass1_dir = os.path.join(out_dir, "pass1")
    demucs.separate.main([
        "--wav", "-n", "htdemucs_6s", "--out", pass1_dir, wav_path,
    ])
    p1_stems = Path(pass1_dir) / "htdemucs_6s" / track
    vocals_path = str(p1_stems / "vocals.wav")
    instrumental_path = str(p1_stems / "no_vocals.wav")
    if not Path(instrumental_path).exists():
        # fallback: some versions name it differently
        for f in p1_stems.glob("*.wav"):
            if "vocal" not in f.name.lower():
                instrumental_path = str(f)
                break

    # Pass 2: instrumental → drums / bass / other
    stage("Pass 2: separating drums / bass / other…")
    pass2_dir = os.path.join(out_dir, "pass2")
    demucs.separate.main([
        "--wav", "-n", "htdemucs", "--out", pass2_dir, instrumental_path,
    ])
    p2_stems = Path(pass2_dir) / "htdemucs" / track

    result = {"vocals": vocals_path}
    for stem in ["drums", "bass", "other"]:
        p = p2_stems / f"{stem}.wav"
        if not p.exists():
            raise FileNotFoundError(f"Missing stem: {p}")
        result[stem] = str(p)

    return result
