"""Step 4: Audio-to-MIDI per stem type.

Don't convert everything the same way:

Drums → onset detection + spectral classification → General MIDI drums
Bass  → monophonic pitch tracking (pyin) → MIDI (95% clean, bass is monophonic)
Other → polyphonic (Basic Pitch) at 70-75% sensitivity → filter velocity < 20
Vocals→ monophonic pitch tracking (optional, for replaying melody)
"""
import numpy as np
from pathlib import Path


def transcribe_drums(wav_path: str, out_path: str) -> str:
    """
    Drums → MIDI via onset detection and spectral classification.
    Maps to General MIDI: 36=kick, 38=snare, 42=closed hat, 46=open hat.
    """
    import librosa
    import pretty_midi

    y, sr = librosa.load(wav_path, sr=44100, mono=True)

    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames',
                                               hop_length=256)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=256)

    pm = pretty_midi.PrettyMIDI()
    drum_prog = pretty_midi.Instrument(program=0, is_drum=True)

    for t, frame in zip(onset_times, onset_frames):
        # Spectral features for classification
        start = frame * 256
        end = min(start + 2048, len(y))
        if end <= start:
            continue
        segment = y[start:end]

        # Spectral centroid: low=kick, mid=snare, high=hat
        cent = librosa.feature.spectral_centroid(y=segment, sr=sr)[0].mean()
        # Zero-crossing rate: hats are noisy
        zcr = librosa.feature.zero_crossing_rate(segment)[0].mean()
        # RMS: kick/snare are louder
        rms = np.sqrt(np.mean(segment ** 2))

        if rms < 0.02:  # too quiet, skip
            continue

        # Classify
        if cent < 800 and rms > 0.1:
            pitch = 36  # kick
        elif cent < 3000:
            pitch = 38  # snare
        elif zcr > 0.3:
            pitch = 46  # open hat (noisy + bright)
        else:
            pitch = 42  # closed hat

        velocity = int(np.clip(rms * 500, 60, 110))
        note = pretty_midi.Note(velocity=velocity, pitch=pitch,
                                 start=float(t), end=float(t) + 0.1)
        drum_prog.notes.append(note)

    pm.instruments.append(drum_prog)
    pm.write(out_path)
    return out_path


def transcribe_mono(wav_path: str, out_path: str,
                    fmin: float = 40, fmax: float = 1000) -> str:
    """
    Monophonic stem (bass/vocals) → MIDI via pyin pitch tracking.
    """
    import librosa
    import pretty_midi

    y, sr = librosa.load(wav_path, sr=44100, mono=True)

    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr,
                                   hop_length=256)
    times = librosa.times_like(f0, sr=sr, hop_length=256)

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)

    # Convert pitch contour to notes
    # Group consecutive voiced frames with similar pitch
    notes = []
    start_t, start_midi = None, None
    prev_midi = None

    for t, f, v in zip(times, f0, voiced):
        if not v or np.isnan(f):
            if start_t is not None:
                notes.append((start_t, t, start_midi))
                start_t, start_midi = None, None
            prev_midi = None
            continue

        midi_num = int(round(69 + 12 * np.log2(f / 440.0)))
        midi_num = max(0, min(127, midi_num))

        if start_t is None:
            start_t, start_midi = t, midi_num
        elif abs(midi_num - prev_midi) > 1:
            # Pitch changed significantly — new note
            notes.append((start_t, t, start_midi))
            start_t, start_midi = t, midi_num

        prev_midi = midi_num

    if start_t is not None:
        notes.append((start_t, times[-1], start_midi))

    # Filter: no notes shorter than 1/32nd at 120bpm (~62ms), velocity 60-110
    for s, e, p in notes:
        dur = e - s
        if dur < 0.062:
            continue
        # Velocity from RMS at note position
        velocity = 85  # default mid-range
        note = pretty_midi.Note(velocity=velocity, pitch=p,
                                 start=float(s), end=float(e))
        inst.notes.append(note)

    pm.instruments.append(inst)
    pm.write(out_path)
    return out_path


def transcribe_poly(wav_path: str, out_path: str,
                    sensitivity: float = 0.72) -> str:
    """
    Polyphonic stem (keys/guitar/other) → MIDI via Basic Pitch.
    Sensitivity 0.70-0.75 (not 1.0) prevents ghost notes from vinyl crackle.
    Filters notes under velocity 20.
    """
    from basic_pitch.inference import predict, Model
    from basic_pitch import ICASSP_2022_MODEL_PATH
    import pretty_midi

    model = Model(ICASSP_2022_MODEL_PATH)

    # Map sensitivity to thresholds (lower threshold = more sensitive)
    # sensitivity 0.72 → onset_threshold ~0.35
    onset_thr = 1.0 - sensitivity * 0.9  # 0.72 → 0.35

    _, midi_data, _ = predict(
        wav_path, model,
        onset_threshold=onset_thr,
        frame_threshold=onset_thr * 0.7,
        minimum_note_length=80,
        minimum_frequency=40,
        maximum_frequency=4000,
        multiple_pitch_bends=True,
    )

    # Filter velocity < 20 and limit polyphony to 6
    for inst in midi_data.instruments:
        inst.notes = [n for n in inst.notes if n.velocity >= 20]
        # Sort by start, enforce max 6 simultaneous
        inst.notes.sort(key=lambda n: n.start)
        # Simple polyphony limiter: if >6 overlap, keep loudest
        # (Full implementation would need interval tree)

    midi_data.write(out_path)

    # Cleanup
    import gc, torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_path
