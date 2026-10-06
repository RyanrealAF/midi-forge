# 🔨 MIDI Forge

**Turn any MP3 into clean MIDI stems for sampling — completely offline.**

No cloud uploads. No Hugging Face. Runs on your machine. Keeps your swing intact.

## The Workflow

```
MP3/WAV
  │
  ▼
1. PREP — normalize to -3dB, convert to 44.1kHz/16-bit WAV
  │
  ▼
2. SEPARATE (two-pass Demucs)
   Pass 1: vocals / instrumental (htdemucs_6s)
   Pass 2: instrumental → drums / bass / other
  │
  ▼
3. CLEAN per stem
   Bass:  low-pass 250Hz, gate silence
   Drums: high-pass 80Hz
   Other: 300Hz–5kHz band (keep the grit)
  │
  ▼
4. TRANSCRIBE per stem type
   Drums → onset detection + spectral classification → GM drums
   Bass  → pyin monophonic pitch tracking → MIDI
   Other → Basic Pitch polyphonic @ 72% sensitivity, velocity ≥ 20
  │
  ▼
5. VERIFY — synthesize MIDI, compare vs stem (chroma + onsets)
  │
  ▼
6. FLIP-READY — per-stem MIDI + all-stems MIDI + ZIP + report
```

## Usage

```bash
pip install -r requirements.txt
python -m forge.cli your_track.mp3 -o ./output
```

## After: Make It Flip-Ready

1. Import all MIDIs at the **original BPM** (shown in report)
2. Quantize to **0%** first — see where it drags, that's your swing
3. Bass/drums → **85%** with MPC-16 swing template
4. Keys → **50-60%**
5. Delete doubled notes, fix octave errors (bass often detects high)

## Clean MIDI Checklist (auto-checked)

- No overlapping notes in bass
- Velocity range 60-110, not flat
- Drum notes no shorter than 1/32nd
- Polyphonic < 6 simultaneous notes

Issues are flagged in the flip report.
