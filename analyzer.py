#!/usr/bin/env python3
"""
analyzer.py  -  Fight-footage highlight analyzer (GPT-4o vision)

Turns a raw training/sparring video into moments.json (the schema reel_editor.py
consumes). Two stages so you don't pay to look at dead footage:

  STAGE 1  (free, local): sample the video at 1 fps, measure per-second MOTION
           (frame-to-frame pixel change) and AUDIO energy (RMS). Combine them,
           threshold, and merge the hot spots into a handful of candidate
           windows. A 20-minute roll collapses to ~10-20 short windows.

  STAGE 2  (GPT-4o, paid): for each candidate window, send a small set of
           timestamped, downscaled frames to GPT-4o and ask it to confirm the
           action, score it, name the technique with a confidence, and give the
           tight in/out timestamps + captions. Only the windows survive to here,
           so token spend stays tiny.

Output matches the analyst schema exactly, so it drops straight into the editor.

Usage:
    export OPENAI_API_KEY=sk-...
    python3 analyzer.py --video roll.mp4 --out moments.json

Key options:
    --model gpt-4o            vision model (default gpt-4o)
    --sample-fps 1.0          Stage-1 sampling rate
    --frames-per-window 12    max frames sent to GPT-4o per window
    --frame-width 768         downscale width sent to the model (cost lever)
    --detail high             image detail: high | low | auto
    --sensitivity 0.5         lower = more candidate windows (more spend)
    --max-windows 20          hard cap on windows sent to GPT-4o
    --dry-run                 skip the API; emit stub moments (pipeline test)
"""

import argparse
import base64
import glob
import io
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

# The analyst instructions - grappling AND striking, with the confidence gate.
SYSTEM_PROMPT = """\
You are a grappling and MMA highlight analyst reviewing frames from ONE short
candidate window of training, sparring, or match footage (BJJ, wrestling, judo,
Muay Thai, boxing, or MMA). The frames are given in time order, each labelled
with its timestamp in seconds.

Find every distinct highlight-worthy action moment in this window. Three kinds
of moment qualify:

FIGHT ACTION: takedowns, throws, sweeps, submissions, submission escapes,
scrambles, dynamic position changes, head/high kicks, knees, elbows, spinning
attacks, clean combinations, knockdowns, and clearly landed strikes.

DEFENCE AND ATTEMPTS COUNT TOO. A takedown does NOT have to succeed to be
highlight-worthy. Actively include: sprawls, stuffed or defended takedowns,
failed but explosive shots, guard retention, scrambles where neither side
clearly wins, reversals, and hard-fought exchanges. A well-defended takedown
attempt is as good a highlight as a completed one. Do not discard a moment
merely because the attack did not finish.

CONDITIONING / ATHLETICISM: visually intense training effort - fast rope climbs,
explosive sprints, plyometrics and box jumps, heavy bag work with visible
impact, pad work with visible impact, tyre flips, sled pushes, battle ropes,
heavy lifts, burpees and other high-intensity conditioning. Judge these on
EFFORT AND INTENSITY made visible: speed, power, explosiveness, strain, full
range of motion. A fast committed rope climb is highlight-worthy; someone
resting on the rope or moving slowly is not.

Ignore: resets, standing around, coaching, stretching, walking, water breaks,
and low-effort or half-hearted movement of any kind.

For striking, prioritise CLEAN, VISIBLE strikes that clearly land or nearly land
with visible impact or reaction. A wild swing at empty air is NOT highlight-worthy
(shadowboxing and air-punching should be ignored) unless visually spectacular
(e.g. a spinning attack). Bag work and pad work DO count, because the impact is
visible and audible.

For EACH moment return an object with:
- start_time, end_time: tightest window in seconds (numbers), inside the frame
  timestamps you were given, from just before the action to just after it resolves.
- visual_score: integer 1-10 for how visually impressive / reel-worthy it is.
- action_type: one of takedown, throw, sweep, submission, escape, scramble,
  strike, kick, knee, combination, knockdown, bagwork, padwork,
  rope_climb, conditioning, strength, other.
- technique_guess: best specific guess. For fight action name the technique
  (e.g. "double leg takedown", "head kick", "armbar"). For conditioning name
  the movement (e.g. "rope climb", "box jump", "tyre flip", "heavy bag work").
  Use "uncertain" if you cannot tell.
- technique_confidence: 0.0-1.0, honest and conservative. Tangled bodies, blur,
  or occlusion lower it. Only 0.8+ when the technique is unambiguous and clear.
- caption_specific: short punchy caption that NAMES the technique (for high conf).
- caption_generic: short punchy caption that does NOT name a technique (for low conf).
- description: one plain sentence of what physically happens.
- subject_x: decimal 0.0-1.0 giving where the ACTION sits horizontally in the
  frame across this moment. 0.0 = hard left edge, 0.5 = centre, 1.0 = hard
  right edge. This is used to decide where to crop for a vertical 9:16 reel,
  so report where the fighters actually are, not where the frame centre is.
  If the action moves, give the position where the key moment happens.

Rules:
- Judge on merit. If this window has nothing worthy, return an empty list.
- Never inflate technique_confidence. Base it only on what is clearly visible.
- Respond ONLY with a JSON object: {"moments": [ ... ]}  (empty list if none).
"""


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, **kw)
    if p.returncode != 0:
        err = p.stderr.decode()[-1500:] if isinstance(p.stderr, bytes) else str(p.stderr)[-1500:]
        sys.stderr.write("\n[cmd error] " + " ".join(cmd[:4]) + "\n" + err + "\n")
        raise SystemExit(1)
    return p


def video_duration(video):
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", video])
    return float(p.stdout.decode().strip())


def extract_frames(video, fps, tmpdir):
    """Sample the video at `fps`; return sorted [(timestamp_seconds, path)]."""
    out = os.path.join(tmpdir, "f_%05d.jpg")
    run(["ffmpeg", "-y", "-i", video, "-vf", f"fps={fps}", "-q:v", "4",
         out, "-loglevel", "error"])
    frames = sorted(glob.glob(os.path.join(tmpdir, "f_*.jpg")))
    # frame n (1-indexed by ffmpeg) is at (n-1)/fps seconds
    return [((i) / fps, f) for i, f in enumerate(frames)]


def motion_series(frames, width=160):
    """Mean absolute pixel change vs the previous sampled frame, per timestamp."""
    prev = None
    scores = []
    for ts, path in frames:
        img = Image.open(path).convert("L").resize((width, width))
        arr = np.asarray(img, dtype=np.float32)
        if prev is None:
            scores.append(0.0)
        else:
            scores.append(float(np.mean(np.abs(arr - prev))))
        prev = arr
    return np.array(scores)


def audio_series(video, n_bins, duration):
    """Per-second audio RMS energy. Returns zeros if the video has no audio."""
    try:
        p = subprocess.run(
            ["ffmpeg", "-i", video, "-ac", "1", "-ar", "8000",
             "-f", "s16le", "-", "-loglevel", "error"],
            capture_output=True,
        )
        if p.returncode != 0 or not p.stdout:
            return np.zeros(n_bins)
        pcm = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32)
        if pcm.size == 0:
            return np.zeros(n_bins)
        samples_per_bin = max(1, int(len(pcm) / n_bins))
        energy = []
        for i in range(n_bins):
            chunk = pcm[i * samples_per_bin:(i + 1) * samples_per_bin]
            energy.append(float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0)
        return np.array(energy)
    except Exception:
        return np.zeros(n_bins)


def norm(x):
    x = np.array(x, dtype=np.float32)
    if x.max() - x.min() < 1e-6:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


def find_windows(frames, video, duration, sensitivity, max_windows,
                 motion_w=0.5, audio_w=0.5, combine="max",
                 pad=1.0, min_dur=2.0, merge_gap=1.5):
    """Combine motion + audio, threshold, and merge hot spots into windows.

    combine="sum": weighted blend (takedown-friendly; big motion dominates).
    combine="max": hot if EITHER channel is hot on its own - this is what lets
                   striking through, since a landed strike is loud (audio-hot)
                   without being a big whole-body motion spike.
    """
    ts = np.array([t for t, _ in frames])
    motion = norm(motion_series(frames))
    audio = norm(audio_series(video, len(frames), duration))
    if combine == "max":
        combined = np.maximum(motion, audio)
    else:
        combined = motion_w * motion + audio_w * audio

    thresh = combined.mean() + sensitivity * combined.std()
    hot = combined >= thresh

    # group contiguous hot timestamps into raw windows
    windows = []
    start = None
    for i, h in enumerate(hot):
        if h and start is None:
            start = ts[i]
        elif not h and start is not None:
            windows.append([start, ts[i - 1], combined[max(0, i - 1)]])
            start = None
    if start is not None:
        windows.append([start, ts[-1], combined[-1]])

    if not windows:  # nothing crossed threshold -> take the single hottest second
        peak = int(np.argmax(combined))
        windows = [[ts[peak], ts[peak], combined[peak]]]

    # pad, clamp, merge windows that are close together
    padded = []
    for s, e, score in windows:
        padded.append([max(0, s - pad), min(duration, e + pad), score])
    padded.sort()
    merged = [padded[0]]
    for s, e, score in padded[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
            merged[-1][2] = max(merged[-1][2], score)
        else:
            merged.append([s, e, score])

    merged = [w for w in merged if (w[1] - w[0]) >= min_dur or w[0] == w[1]]
    # keep the strongest windows if there are too many
    merged.sort(key=lambda w: w[2], reverse=True)
    merged = merged[:max_windows]
    merged.sort()
    return [(round(s, 2), round(e, 2)) for s, e, _ in merged]


def chunk_windows(windows, max_dur, overlap=1.0):
    """Split long windows into shorter chunks.

    A long window sampled with a fixed frame budget gets terrible time
    resolution - a 26s window at 12 frames is one frame every ~2.2s, and a
    takedown that takes 1.5s falls entirely between frames. Chunking gives
    every chunk the FULL frame budget, so the model actually sees the action.
    Chunks overlap slightly so an action straddling a boundary isn't lost.
    """
    out = []
    for s, e in windows:
        dur = e - s
        if dur <= max_dur:
            out.append((s, e))
            continue
        cur = s
        while cur < e:
            nxt = min(e, cur + max_dur)
            out.append((round(cur, 2), round(nxt, 2)))
            if nxt >= e:
                break
            cur = nxt - overlap
    return out


def frames_in_window(frames, start, end, cap, width):
    """Pick up to `cap` frames inside [start,end], evenly spaced, downscaled."""
    inside = [(t, p) for t, p in frames if start <= t <= end]
    if not inside:
        return []
    if len(inside) > cap:
        idx = np.linspace(0, len(inside) - 1, cap).round().astype(int)
        inside = [inside[i] for i in idx]
    out = []
    for t, path in inside:
        img = Image.open(path).convert("RGB")
        if img.width > width:
            img = img.resize((width, round(img.height * width / img.width)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        out.append((round(t, 2), base64.b64encode(buf.getvalue()).decode()))
    return out


def analyze_window(client, model, detail, wframes, wstart, wend, context=None):
    """One GPT-4o call: timestamped frames in, list of moments out."""
    system = SYSTEM_PROMPT
    if context:
        system += (
            "\n\nSESSION CONTEXT FROM THE USER (this describes what THIS "
            "particular set of footage contains - trust it and adapt what you "
            "treat as highlight-worthy accordingly):\n" + context.strip() + "\n"
        )
    content = [{
        "type": "text",
        "text": (f"Frames from a candidate window spanning {wstart:.1f}s to "
                 f"{wend:.1f}s. Each frame is labelled with its timestamp.")
    }]
    for ts, b64 in wframes:
        content.append({"type": "text", "text": f"[frame @ {ts:.1f}s]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail},
        })

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    )
    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
        return data.get("moments", []) if isinstance(data, dict) else []
    except json.JSONDecodeError:
        sys.stderr.write(f"[warn] non-JSON reply for window {wstart}-{wend}\n")
        return []


def fmt_ts(seconds):
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def main():
    ap = argparse.ArgumentParser(description="Analyze fight footage -> moments.json")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="moments.json")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--sample-fps", type=float, default=1.0)
    ap.add_argument("--frames-per-window", type=int, default=20)
    ap.add_argument("--max-window-dur", type=float, default=8.0,
                    help="split candidate windows longer than this into chunks "
                         "so each chunk gets the full frame budget")
    ap.add_argument("--frame-width", type=int, default=768)
    ap.add_argument("--detail", default="high", choices=["high", "low", "auto"])
    ap.add_argument("--sensitivity", type=float, default=0.5)
    ap.add_argument("--combine", choices=["max", "sum"], default="max",
                    help="max = fire if motion OR audio is hot (catches striking); "
                         "sum = weighted blend (motion-dominant, takedown-friendly)")
    ap.add_argument("--motion-weight", type=float, default=0.5)
    ap.add_argument("--audio-weight", type=float, default=0.5)
    ap.add_argument("--max-windows", type=int, default=20)
    ap.add_argument("--context", default=None,
                    help="describe THIS set of footage so the model adapts, e.g. "
                         "\"medal ceremony and drill games, no sparring\"")
    ap.add_argument("--context-file", default=None,
                    help="read the session context from a text file instead")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip GPT-4o; emit one stub moment per window")
    args = ap.parse_args()

    context = args.context
    if args.context_file:
        with open(args.context_file) as fh:
            context = fh.read()
    if context:
        print(f"Session context: {context.strip()[:120]}")

    duration = video_duration(args.video)
    print(f"Video: {duration:.1f}s")

    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_frames(args.video, args.sample_fps, tmp)
        print(f"Stage 1: sampled {len(frames)} frames @ {args.sample_fps} fps")

        windows = find_windows(frames, args.video, duration,
                               args.sensitivity, args.max_windows,
                               motion_w=args.motion_weight,
                               audio_w=args.audio_weight,
                               combine=args.combine)
        print(f"Stage 1: {len(windows)} candidate window(s): "
              + ", ".join(f"{s:.0f}-{e:.0f}s" for s, e in windows))

        windows = chunk_windows(windows, args.max_window_dur)
        print(f"Stage 1: split into {len(windows)} chunk(s) of "
              f"<= {args.max_window_dur:.0f}s for analysis")

        client = None
        if not args.dry_run:
            from openai import OpenAI
            client = OpenAI()  # reads OPENAI_API_KEY (and project scope if key-scoped)

        moments = []
        for i, (ws, we) in enumerate(windows):
            wframes = frames_in_window(frames, ws, we,
                                       args.frames_per_window, args.frame_width)
            if not wframes:
                continue
            if args.dry_run:
                found = [{
                    "start_time": ws, "end_time": min(we, ws + 5),
                    "visual_score": 5, "action_type": "other",
                    "technique_guess": "uncertain", "technique_confidence": 0.0,
                    "caption_specific": "", "caption_generic": "Highlight of the day",
                    "description": f"[dry-run stub for window {ws}-{we}s]",
                }]
            else:
                print(f"  GPT-4o window {i+1}/{len(windows)} "
                      f"({ws:.0f}-{we:.0f}s, {len(wframes)} frames, "
                      f"1 frame per {(we-ws)/max(1,len(wframes)):.1f}s)")
                found = analyze_window(client, args.model, args.detail,
                                       wframes, ws, we, context=context)
                print(f"      -> {len(found)} moment(s)"
                      + ("" if found else "   [nothing found here]"))
            moments.extend(found)

    # normalise timestamps to mm:ss.ms strings for the editor, sort by score
    for m in moments:
        m["start_time"] = fmt_ts(float(m["start_time"]))
        m["end_time"] = fmt_ts(float(m["end_time"]))
    moments.sort(key=lambda m: float(m.get("visual_score", 0) or 0), reverse=True)

    with open(args.out, "w") as fh:
        json.dump({"moments": moments}, fh, indent=2, ensure_ascii=False)

    print(f"\n{len(moments)} moment(s) -> {args.out}")
    for m in moments[:10]:
        print(f"  score {m.get('visual_score')}  {m['start_time']}-{m['end_time']}  "
              f"conf {m.get('technique_confidence')}  {m.get('technique_guess')}")


if __name__ == "__main__":
    main()