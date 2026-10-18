# Reel-MMA

Turns raw training footage from Ignition MMA into 9:16 Instagram highlight reels. Built for the gym in July 2026. It isn't running now because the vision API cost per reel was too high.

## How it works

Google Drive → R2 `incoming/<session>/` → `batch_concat.py` → `analyzer.py` → `editor_agent.py` → `reel_editor.py` → R2 `outgoing/<session>/reel.mp4`

- **`batch_concat.py`** normalises clips with mixed resolution, fps and codec, then joins them so the best moments across all clips compete for a slot.
- **`analyzer.py`** works in two stages. Stage 1 is free and local: motion and audio energy pick candidate windows. Stage 2 sends a few downscaled frames per window to GPT-4o, which confirms the action, scores it and names the technique with a confidence.
- **`editor_agent.py`** is a text-only LLM call that plans the edit (clip order, pacing, transitions) as an edit decision list. Its output is validated and repaired, with a score-ranked fallback.
- **`reel_editor.py`** uses ffmpeg only: it cuts, crops to 9:16 around the action, cross-fades and writes the post caption.
- **`.github/workflows/reel.yml`** processes every pending session every 6 hours or on demand. Reels land in `outgoing/` for a person to review; nothing posts automatically.

## Run locally

Needs Python 3.12 and ffmpeg on `PATH`.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
python analyzer.py --video roll.mp4 --out moments.json
python editor_agent.py --analysis moments.json --out edl.json
python reel_editor.py --video roll.mp4 --analysis moments.json --edl edl.json --out reel.mp4
```

`--dry-run` on the analyzer and the editor agent skips the API, for testing the pipeline.

The workflow needs these secrets: `OPENAI_API_KEY`, `R2_ACCOUNT_ID`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`, `R2_BUCKET`, and optionally `GOOGLE_SERVICE_ACCOUNT_JSON` with `DRIVE_PARENT_FOLDER_ID`.
