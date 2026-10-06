"""High-accuracy transcription via Basic-Pitch with adaptive parameters."""
from pathlib import Path

BASE_PARAMS = {
    "vocals": {"onset_threshold": 0.6, "frame_threshold": 0.3,
               "minimum_note_length": 100, "minimum_frequency": 80,
               "maximum_frequency": 1100, "multiple_pitch_bends": False,
               "melodia_trick": True},
    "drums":  {"onset_threshold": 0.3, "frame_threshold": 0.2,
               "minimum_note_length": 50, "minimum_frequency": 30,
               "maximum_frequency": 8000, "multiple_pitch_bends": False,
               "melodia_trick": False},
    "bass":   {"onset_threshold": 0.5, "frame_threshold": 0.25,
               "minimum_note_length": 80, "minimum_frequency": 30,
               "maximum_frequency": 300, "multiple_pitch_bends": False,
               "melodia_trick": True},
    "other":  {"onset_threshold": 0.5, "frame_threshold": 0.3,
               "minimum_note_length": 80, "minimum_frequency": 40,
               "maximum_frequency": 4000, "multiple_pitch_bends": True,
               "melodia_trick": True},
}


def adapt_params(stem: str, analysis: dict, attempt: int) -> dict:
    """Tune Basic-Pitch params from stem analysis + retry count."""
    p = dict(BASE_PARAMS[stem])

    # Narrow frequency range to detected pitch (skip drums)
    if stem != "drums" and analysis.get("pitch_min_hz", 0) > 0:
        p["minimum_frequency"] = max(20, analysis["pitch_min_hz"] * 0.8)
        p["maximum_frequency"] = min(8000, analysis["pitch_max_hz"] * 1.25)

    # Dense onsets → more sensitive
    if analysis.get("onset_density", 0) > 4.0:
        p["onset_threshold"] = max(0.15, p["onset_threshold"] - 0.15)

    # Quiet stem → more sensitive frames
    if analysis.get("energy_db", 0) < -30:
        p["frame_threshold"] = max(0.1, p["frame_threshold"] - 0.1)

    # Retries → progressively more permissive
    if attempt > 0:
        p["onset_threshold"] = max(0.15, p["onset_threshold"] - 0.1 * attempt)
        p["frame_threshold"] = max(0.1, p["frame_threshold"] - 0.05 * attempt)

    return p


def transcribe_stem(stem: str, wav_path: str, out_path: str,
                    model, analysis: dict, attempt: int = 0) -> str:
    """Transcribe a single stem to MIDI. Returns the MIDI path."""
    from basic_pitch.inference import predict

    params = adapt_params(stem, analysis, attempt)
    _, midi_data, _ = predict(wav_path, model, **params)
    midi_data.write(out_path)
    return out_path
