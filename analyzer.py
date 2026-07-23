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
SHARED_SCHEMA = """\
For EACH moment return an object with:
- start_time, end_time: tightest window in seconds (numbers), inside the frame
  timestamps you were given, from just before the action to just after it resolves.
- visual_score: integer 1-10 for how visually impressive / reel-worthy it is.
  USE THE FULL RANGE AND SPREAD YOUR SCORES. Do not give everything a 7. Most
  ordinary moments belong at 3-5. Reserve 8-10 for genuinely exceptional
  moments, the ones you would lead a reel with. A routine drilling repetition
  scores low; a clean high-amplitude throw or a finished submission scores
  high. If you rate several moments in this window, they should not all get the
  same number. Scoring everything the same makes the score useless.
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


ACTIVITY_PROFILES = {
    "mma": {
        "label": "MMA / grappling / martial arts training",
        "ranked": True,
        "brief": 'You are a grappling and MMA highlight analyst reviewing frames from ONE short\ncandidate window of training, sparring, or match footage (BJJ, wrestling, judo,\nMuay Thai, boxing, or MMA). The frames are given in time order, each labelled\nwith its timestamp in seconds.\n\nFind every distinct highlight-worthy action moment in this window. Three kinds\nof moment qualify:\n\nFIGHT ACTION: takedowns, throws, sweeps, submissions, submission escapes,\nscrambles, dynamic position changes, head/high kicks, knees, elbows, spinning\nattacks, clean combinations, knockdowns, and clearly landed strikes.\n\nDEFENCE AND ATTEMPTS COUNT TOO. A takedown does NOT have to succeed to be\nhighlight-worthy. Actively include: sprawls, stuffed or defended takedowns,\nfailed but explosive shots, guard retention, scrambles where neither side\nclearly wins, reversals, and hard-fought exchanges. A well-defended takedown\nattempt is as good a highlight as a completed one. Do not discard a moment\nmerely because the attack did not finish.\n\nCONDITIONING / ATHLETICISM: visually intense training effort - fast rope climbs,\nexplosive sprints, plyometrics and box jumps, heavy bag work with visible\nimpact, pad work with visible impact, tyre flips, sled pushes, battle ropes,\nheavy lifts, burpees and other high-intensity conditioning. Judge these on\nEFFORT AND INTENSITY made visible: speed, power, explosiveness, strain, full\nrange of motion. A fast committed rope climb is highlight-worthy; someone\nresting on the rope or moving slowly is not.\n\nIgnore: resets, standing around, coaching, stretching, walking, water breaks,\nand low-effort or half-hearted movement of any kind.\n\nFor striking, prioritise CLEAN, VISIBLE strikes that clearly land or nearly land\nwith visible impact or reaction. A wild swing at empty air is NOT highlight-worthy\n(shadowboxing and air-punching should be ignored) unless visually spectacular\n(e.g. a spinning attack). Bag work and pad work DO count, because the impact is\nvisible and audible.',
    },
    "football": {
        "label": "football / soccer",
        "ranked": True,
        "brief": """\
You are a football (soccer) highlight analyst reviewing frames from ONE short
window of a match or a kickabout. The frames are in time order, each labelled
with its timestamp in seconds.

Find every distinct highlight-worthy moment in this window. What qualifies:
goals, shots on target, near misses that hit the woodwork, saves by the keeper,
skill moves and successful dribbles past an opponent, strong tackles and
interceptions, defence-splitting passes, headers, volleys, and celebrations
straight after a goal.

Judge on VISIBLE OUTCOME AND SKILL. A goal is the strongest moment there is. A
save or a clean skill move is strong. Ordinary passing in midfield, throw-ins,
players jogging back, and stoppages are NOT highlights. A shot that misses
badly is not a highlight unless the strike itself was spectacular.

Celebrations count, but only the moment right after a goal, and keep them short.
""",
    },
    "kabaddi": {
        "label": "kabaddi",
        "ranked": True,
        "brief": """\
You are a kabaddi highlight analyst reviewing frames from ONE short window of a
match. The frames are in time order, each labelled with its timestamp in seconds.

Find every distinct highlight-worthy moment. What qualifies: successful raids
where the raider touches defenders and escapes back across the line, multi-point
raids, strong tackles where defenders stop a raider, ankle holds, dashes,
last-man struggles, escapes from a group tackle, and bonus point attempts.

Judge on STRUGGLE AND OUTCOME MADE VISIBLE. The best moments are physical
contests where several players are involved and the result is clear on camera.
Players walking back to position, waiting between raids, and dead time between
plays are NOT highlights.

Kabaddi terminology may be hard to read from frames. If you cannot name the
specific move, say "uncertain" and keep the confidence low, but still report
the moment if the action is visually strong.
""",
    },
    "ceremony": {
        "label": "medal or belt ceremony, awards, presentations",
        "ranked": False,
        "brief": """\
You are documenting a CEREMONY from frames of ONE short window: a medal
presentation, belt promotion, award, or prize giving. The frames are in time
order, each labelled with its timestamp in seconds.

This is NOT a highlight reel. Your job is to capture EVERY presentation so that
nobody is left out. What qualifies: a medal or belt or trophy being handed over
or placed on someone, a handshake or bow between instructor and student, a
recipient raising or showing their award, group and team photos, applause and
celebration around a recipient, and short speech moments.

Judge on CLARITY OF THE MOMENT, not excitement. A calm, clearly framed handover
is a GOOD moment and should score well. Do not penalise a moment for being
still or quiet, ceremonies are meant to be calm. Score low only when the shot is
unclear, obstructed, or nothing is actually happening.

Report EACH separate recipient as its own moment, even when several look alike.
Missing a person is the worst possible failure here. When in doubt, include it.

For action_type use: medal_presentation, belt_promotion, handshake, group_photo,
celebration, speech, or other.
""",
    },
    "generic": {
        "label": "general activity",
        "ranked": True,
        "brief": """\
You are a highlight analyst reviewing frames from ONE short window of footage
from a sports or training session. The frames are in time order, each labelled
with its timestamp in seconds.

Find every distinct moment worth putting in a short social video. What
qualifies: any moment of clear physical skill, effort, speed, power, or
successful execution; any clear outcome such as scoring, winning an exchange, or
completing something difficult; and genuine celebration or reaction.

Judge on VISIBLE SKILL, EFFORT, OR OUTCOME. Ignore standing around, walking,
waiting, talking, and dead time. If the frames show nothing notable, return an
empty list rather than inventing something.
""",
    },
}


def build_prompt(primary, also_present=None):
    """Compose the analyst prompt for the detected activity or activities."""
    profile = ACTIVITY_PROFILES.get(primary, ACTIVITY_PROFILES["generic"])
    prompt = profile["brief"]

    extras = [a for a in (also_present or [])
              if a != primary and a in ACTIVITY_PROFILES]
    if extras:
        names = ", ".join(ACTIVITY_PROFILES[a]["label"] for a in extras)
        prompt += (
            f"\nTHIS SESSION ALSO CONTAINS: {names}. If the frames in this "
            f"window show that instead of the main activity, judge them by "
            f"their own standards and report them normally. Sessions often mix "
            f"activities, for example training followed by a presentation.\n"
        )
        for a in extras:
            prompt += "\n" + ACTIVITY_PROFILES[a]["brief"]

    return prompt + "\n" + SHARED_SCHEMA


DETECT_PROMPT = """\
You are shown frames sampled evenly across an entire training or sports session
video. Identify what activities appear in it.

Choose from exactly these labels:
  mma        - martial arts: BJJ, grappling, wrestling, judo, boxing, Muay Thai, MMA
  football   - football / soccer
  kabaddi    - kabaddi
  ceremony   - medal or belt presentations, awards, prize giving
  generic    - a physical activity that is none of the above

Return ONLY JSON:
{"primary": "<label>", "also_present": ["<label>", ...], "reasoning": "one sentence"}

"primary" is the activity filling most of the video. "also_present" lists any
OTHER labels that clearly appear, even briefly, and is empty if there are none.
A session that is mostly training but ends with medals being given out should
return primary "mma" and also_present ["ceremony"].
"""


def detect_activity(client, model, frames, n=12, width=512):
    """One cheap call: sample across the WHOLE video and ask what it is."""
    import numpy as _np
    if not frames:
        return {"primary": "generic", "also_present": [], "reasoning": "no frames"}

    idx = _np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    picked = [frames[i] for i in idx]

    content = [{"type": "text",
                "text": "Frames sampled evenly across the whole video, in order."}]
    for ts, path in picked:
        img = Image.open(path).convert("RGB")
        if img.width > width:
            img = img.resize((width, round(img.height * width / img.width)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,"
                       + base64.b64encode(buf.getvalue()).decode(),
                "detail": "low",
            },
        })

    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": DETECT_PROMPT},
                      {"role": "user", "content": content}],
        )
        data = json.loads(resp.choices[0].message.content)
        primary = str(data.get("primary", "generic")).lower().strip()
        if primary not in ACTIVITY_PROFILES:
            primary = "generic"
        also = [str(a).lower().strip() for a in data.get("also_present", [])]
        also = [a for a in also if a in ACTIVITY_PROFILES and a != primary]
        return {"primary": primary, "also_present": also,
                "reasoning": str(data.get("reasoning", ""))[:200]}
    except Exception as exc:
        sys.stderr.write(f"[warn] activity detection failed ({exc}), using generic\n")
        return {"primary": "generic", "also_present": [], "reasoning": "detection failed"}


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


def analyze_window(client, model, detail, wframes, wstart, wend, context=None,
                   system_prompt=None):
    """One GPT-4o call: timestamped frames in, list of moments out."""
    system = system_prompt or build_prompt("mma")
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


def merge_moments(moments, min_overlap=0.3, max_span=8.0):
    """Rejoin ONE action that got split across a chunk boundary.

    Chunks overlap by a second, so when an action straddles a seam it is
    analysed twice and comes back as two halves that genuinely OVERLAP in time.
    That overlap is the signal. Two different actions happening back to back
    merely sit near each other; they do not overlap.

    So merging requires all of:
      - real overlap of at least min_overlap seconds, not mere proximity
      - the same action_type
      - the same technique, or one side admitting it was uncertain
      - a combined span no longer than max_span

    The span cap is the backstop. A single technique resolves in a few seconds.
    Anything longer means unrelated actions are being chained together, which
    produces meandering clips instead of highlights.
    """
    if not moments:
        return moments

    def s(m):
        return _parse_seconds(m["start_time"])

    def e(m):
        return _parse_seconds(m["end_time"])

    def norm_tech(m):
        t = str(m.get("technique_guess", "")).lower().strip()
        return t.replace("-", " ").replace("attempt", "").strip()

    ordered = sorted(moments, key=s)
    out = [dict(ordered[0])]

    for m in ordered[1:]:
        prev = out[-1]

        overlap = min(e(prev), e(m)) - max(s(prev), s(m))
        if overlap < min_overlap:
            out.append(dict(m))
            continue

        if str(m.get("action_type", "")).lower() != str(prev.get("action_type", "")).lower():
            out.append(dict(m))
            continue

        ta, tb = norm_tech(prev), norm_tech(m)
        same_technique = ta == tb or "uncertain" in (ta, tb) or ta in tb or tb in ta
        if not same_technique:
            out.append(dict(m))
            continue

        span = max(e(prev), e(m)) - min(s(prev), s(m))
        if span > max_span:
            out.append(dict(m))
            continue

        prev["start_time"] = _fmt_seconds(min(s(prev), s(m)))
        prev["end_time"] = _fmt_seconds(max(e(prev), e(m)))
        if float(m.get("visual_score", 0) or 0) > float(prev.get("visual_score", 0) or 0):
            prev["visual_score"] = m["visual_score"]
        if float(m.get("technique_confidence", 0) or 0) > float(
                prev.get("technique_confidence", 0) or 0):
            for k in ("technique_guess", "technique_confidence",
                      "caption_specific", "caption_generic", "description"):
                if k in m:
                    prev[k] = m[k]
        prev["merged"] = prev.get("merged", 1) + 1

    return out


def _parse_seconds(ts):
    if isinstance(ts, (int, float)):
        return float(ts)
    parts = [float(p) for p in str(ts).split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def _fmt_seconds(seconds):
    m = int(seconds // 60)
    return f"{m:02d}:{seconds - m * 60:05.2f}"


def fmt_ts(seconds):
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def _analyze(args):
    """Everything that costs money: Stage 1 detection plus the GPT-4o calls."""
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

        if args.full_scan:
            windows = [(0.0, duration)]
            print("Stage 1: FULL SCAN, heuristic bypassed, whole video examined")
        else:
            windows = find_windows(frames, args.video, duration,
                               args.sensitivity, args.max_windows,
                               motion_w=args.motion_weight,
                               audio_w=args.audio_weight,
                               combine=args.combine)
        print(f"Stage 1: {len(windows)} candidate window(s): "
              + ", ".join(f"{s:.0f}-{e:.0f}s" for s, e in windows))

        windows = chunk_windows(windows, args.max_window_dur)
        # merge overlaps so chunk overlap is not double counted
        merged, covered = [], 0.0
        for s, e in sorted(windows):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        covered = sum(e - s for s, e in merged)
        pct = 100.0 * covered / duration if duration else 0
        print(f"Stage 1: split into {len(windows)} chunk(s) of "
              f"<= {args.max_window_dur:.0f}s for analysis")
        print(f"Stage 1: COVERAGE {covered:.0f}s of {duration:.0f}s "
              f"({pct:.0f}% of the video will be examined)")
        if pct < 40 and not args.full_scan:
            print("         low coverage: if action is being missed, lower "
                  "--sensitivity or use --full-scan")

        client = None
        if not args.dry_run:
            from openai import OpenAI
            client = OpenAI()  # reads OPENAI_API_KEY (and project scope if key-scoped)

        # Decide what this footage actually is before judging any of it.
        if args.profile == "auto":
            if args.dry_run:
                activity = {"primary": "generic", "also_present": [],
                            "reasoning": "dry run"}
            else:
                activity = detect_activity(client, args.model, frames)
                print(f"\nActivity detected: "
                      f"{ACTIVITY_PROFILES[activity['primary']]['label']}"
                      + (f"  (also: {', '.join(activity['also_present'])})"
                         if activity["also_present"] else ""))
                if activity.get("reasoning"):
                    print(f"  {activity['reasoning']}")
        else:
            activity = {"primary": args.profile, "also_present": [],
                        "reasoning": "set by --profile"}
            print(f"\nActivity: {ACTIVITY_PROFILES[args.profile]['label']} (forced)")

        system_prompt = build_prompt(activity["primary"], activity["also_present"])

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
                                       wframes, ws, we, context=context,
                                       system_prompt=system_prompt)
                print(f"      -> {len(found)} moment(s)"
                      + ("" if found else "   [nothing found here]"))
            moments.extend(found)

        _analyze.activity = activity

    # The GPT-4o calls above are the only part of this that costs money. Dump
    # them straight to disk before any post-processing, so a bug in merging or
    # formatting can never throw away a paid run.
    if not args.dry_run:
        raw_path = args.out + ".raw.json"
        try:
            with open(raw_path, "w") as fh:
                json.dump({"activity": getattr(_analyze, "activity", {}), "moments": moments}, fh, indent=2, ensure_ascii=False)
            print(f"\nRaw detections saved to {raw_path} ({len(moments)} found)")
        except Exception as exc:
            sys.stderr.write(f"[warn] could not write raw backup: {exc}\n")

    return moments


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
    ap.add_argument("--full-scan", action="store_true",
                    help="bypass the heuristic and examine the ENTIRE video. "
                         "Guarantees nothing is skipped; costs more.")
    ap.add_argument("--profile", default="auto",
                    choices=["auto", "mma", "football", "kabaddi",
                             "ceremony", "generic"],
                    help="what the footage is. 'auto' detects it with one "
                         "cheap call before analysis starts.")
    ap.add_argument("--from-raw", default=None,
                    help="skip all analysis and re-process a saved "
                         "moments.json.raw.json file. Free, no API calls.")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip GPT-4o; emit one stub moment per window")
    args = ap.parse_args()

    if args.from_raw:
        with open(args.from_raw) as fh:
            rawdata = json.load(fh)
        moments = rawdata["moments"] if isinstance(rawdata, dict) else rawdata
        print(f"Re-processing {len(moments)} saved detection(s) from "
              f"{args.from_raw}, no API calls")
    else:
        moments = _analyze(args)

    # Repair actions split across chunk boundaries before anything else uses them.
    before = len(moments)
    moments = merge_moments(moments)
    repaired = sum(m.get("merged", 1) - 1 for m in moments)
    if repaired:
        print(f"\nMerged {before} raw detections into {len(moments)} moments "
              f"({repaired} split scene(s) stitched back together)")

    # normalise timestamps to mm:ss.ms strings for the editor, sort by score.
    # _parse_seconds tolerates both raw seconds from the model and the
    # mm:ss.ms strings that merge_moments writes.
    for m in moments:
        m["start_time"] = fmt_ts(_parse_seconds(m["start_time"]))
        m["end_time"] = fmt_ts(_parse_seconds(m["end_time"]))
    moments.sort(key=lambda m: float(m.get("visual_score", 0) or 0), reverse=True)

    activity = getattr(_analyze, "activity", None) or {
        "primary": args.profile if args.profile != "auto" else "generic",
        "also_present": [],
    }
    if args.from_raw:
        activity = rawdata.get("activity", activity) if isinstance(rawdata, dict) else activity

    with open(args.out, "w") as fh:
        json.dump({"activity": activity, "moments": moments}, fh,
                  indent=2, ensure_ascii=False)

    print(f"\n{len(moments)} moment(s) -> {args.out}")
    for m in moments[:10]:
        print(f"  score {m.get('visual_score')}  {m['start_time']}-{m['end_time']}  "
              f"conf {m.get('technique_confidence')}  {m.get('technique_guess')}")


if __name__ == "__main__":
    main()