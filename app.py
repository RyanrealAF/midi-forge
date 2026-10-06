"""MIDI Forge — Gradio UI."""
import os
import shutil
import tempfile
import threading
import uuid

import gradio as gr

from forge.pipeline import run_pipeline, THRESHOLD
from forge.separate import STEMS

JOBS: dict = {}
POLL = 3

CSS = """
.gradio-container { max-width: 880px; margin: 0 auto; }
.status { background: #111; border: 1px solid #2a2a2a; border-radius: 6px;
          padding: 0.8rem 1rem; font-size: 0.9rem; }
footer { display: none !important; }
"""


def submit(audio):
    if audio is None:
        raise gr.Error("Upload an audio file first.")
    jid = str(uuid.uuid4())
    wd = tempfile.mkdtemp(prefix="forged_")
    JOBS[jid] = {"status": "running", "stage": "Queued…",
                 "result": None, "error": None, "work_dir": wd}

    def _run():
        job = JOBS[jid]
        try:
            job["result"] = run_pipeline(
                audio, wd, on_stage=lambda s: job.update(stage=s))
            job["status"] = "done"
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)[:800]
            shutil.rmtree(wd, ignore_errors=True)

    threading.Thread(target=_run, daemon=True).start()
    return jid, "⏳ Forging…", gr.update(interactive=False)


def poll(jid):
    blank = [gr.update()] * 6
    if not jid or jid not in JOBS:
        return ("", *blank, gr.update(interactive=True))
    job = JOBS[jid]
    if job["status"] == "running":
        return (f"⏳ {job['stage']}", *blank, gr.update(interactive=False))
    if job["status"] == "error":
        return (f"❌ {job['error']}", *blank, gr.update(interactive=True))

    r = job["result"]
    lines = []
    for s in STEMS:
        v, a = r["verification"][s], r["analysis"][s]
        icon = "✓" if v["score"] >= THRESHOLD else "⚠"
        lines.append(f"{icon} **{s}** — {v['score']} ({v['detail']}) · "
                     f"{a['bpm']} BPM · {a['key']}")
    summary = f"✓ **{r['track']}**\n\n" + "\n".join(lines)
    return (summary,
            r["midi"]["vocals"], r["midi"]["drums"],
            r["midi"]["bass"], r["midi"]["other"],
            r["zip"], gr.update(interactive=True))


with gr.Blocks(css=CSS, title="MIDI Forge") as demo:
    gr.Markdown("""
# 🔨 MIDI Forge
**Audio → verified MIDI.** Separate → analyze → transcribe → verify.

Each stem is analyzed first (BPM, key, pitch range), transcribed with adapted
parameters, then the MIDI is synthesized back and compared to the original stem.
Low scores trigger automatic retry with more sensitive settings.
""")
    state = gr.State(None)
    audio_in = gr.Audio(label="Input audio", type="filepath", sources=["upload"])
    btn = gr.Button("▶ Forge MIDI", variant="primary", size="lg")
    status = gr.Markdown(elem_classes=["status"])

    gr.Markdown("### MIDI outputs")
    with gr.Row():
        o_vox = gr.File(label="vocals.mid", interactive=False)
        o_drm = gr.File(label="drums.mid", interactive=False)
    with gr.Row():
        o_bas = gr.File(label="bass.mid", interactive=False)
        o_oth = gr.File(label="other.mid", interactive=False)
    o_zip = gr.File(label="📦 All MIDIs (zip)", interactive=False)

    timer = gr.Timer(value=POLL, active=False)
    btn.click(fn=submit, inputs=[audio_in],
              outputs=[state, status, btn]
              ).then(fn=lambda: gr.Timer(active=True), outputs=[timer])
    timer.tick(fn=poll, inputs=[state],
               outputs=[status, o_vox, o_drm, o_bas, o_oth, o_zip, btn])

if __name__ == "__main__":
    demo.launch()
