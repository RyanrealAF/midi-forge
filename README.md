---
title: MIDI Forge
emoji: 🔨
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🔨 MIDI Forge

High-accuracy audio-to-MIDI converter with stem separation and **audio verification**.

## How it works

```
Input audio
    │
    ▼
Demucs — separate into vocals / drums / bass / other
    │
    ▼ (per stem)
ANALYZE — BPM, key, pitch range, energy, onset density
    │
    ▼
TRANSCRIBE — Basic-Pitch with parameters adapted from analysis
    │
    ▼
VERIFY — synthesize MIDI → compare chroma + onsets vs original stem
    │
    ├── score ≥ 0.65 → keep
    └── score < 0.65 → retry with more sensitive params (max 3 attempts)
    │
    ▼
Best MIDI kept. Verification report generated.
```

## Why verify?

Most audio-to-MIDI tools transcribe blindly — you get notes but no idea if
they're right. MIDI Forge closes the loop: it renders the MIDI back to audio
and measures how well it matches the source. If the match is weak, it tries
again with different settings instead of handing you a bad file.

## Outputs

- `vocals.mid`, `drums.mid`, `bass.mid`, `other.mid`
- ZIP bundle of all MIDIs
- Verification report with per-stem scores and analysis
