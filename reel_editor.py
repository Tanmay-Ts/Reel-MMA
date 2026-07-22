#!/usr/bin/env python3
"""
reel_editor.py  -  Fight-highlight reel builder

Takes ONE source video plus the analysis JSON (the schema produced by the
Gemini/OpenAI highlight analyst) and produces a vertical 9:16 Instagram reel:

  1. Sort moments by visual_score, keep the best ones.
  2. Auto-drop the weakest clips if the montage would exceed the Reels 90s cap.
  3. Trim each moment to its start/end window.
  4. Burn a technique label on-screen  ->  specific name if the model was
     confident (technique_confidence >= gate), otherwise the generic label.
     Emoji are stripped from the BURNED text (ffmpeg can't render colour emoji)
     but kept in the post caption written to the sidecar .txt.
  5. Normalise every clip to identical format so they can be joined cleanly.
  6. Stitch with an animated cross-transition (video xfade + audio acrossfade).
  7. Write the assembled Instagram post caption to <output>.caption.txt.

No paid API is used here - this stage is 100% ffmpeg.

Usage:
    python3 reel_editor.py --video roll.mp4 --analysis moments.json --out reel.mp4

Common options:
    --max-clips 6          hard cap on number of highlights (default 6)
    --gate 0.8             confidence needed to burn the SPECIFIC name (default 0.8)
    --transition smoothleft  xfade style (default smoothleft)
    --trans-dur 0.4        transition length in seconds (default 0.4)
    --no-captions          skip on-screen labels entirely
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

# 9:16 vertical, 1080x1920, the Instagram Reels native canvas.
W, H = 1080, 1920
FPS = 30
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
REELS_MAX = 90.0   # seconds - hard Instagram cap for the Reels tab
REELS_MIN = 5.0    # seconds - below this it won't land in the Reels tab

EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF" "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF" "\uFE0F" "]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    return EMOJI_RE.sub("", text or "").strip()


def parse_ts(ts) -> float:
    """Accept 'mm:ss.ms', 'hh:mm:ss.ms', or a raw number of seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    ts = str(ts).strip()
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0.0, parts[0], parts[1]
    else:
        h, m, s = 0.0, 0.0, parts[0]
    return h * 3600 + m * 60 + s


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write("\n[ffmpeg error]\n" + p.stderr[-2500:] + "\n")
        raise SystemExit(1)
    return p


def burned_label(moment: dict, gate: float) -> str:
    """Confidence gate: specific technique name only when the model is sure."""
    conf = float(moment.get("technique_confidence", 0) or 0)
    if conf >= gate:
        cap = moment.get("caption_specific") or moment.get("technique_guess") or ""
    else:
        cap = moment.get("caption_generic") or moment.get("action_type") or ""
    return strip_emoji(cap).upper()


def post_caption(moment: dict, gate: float) -> str:
    """Emoji kept here - this goes in the Instagram post text, not on-screen."""
    conf = float(moment.get("technique_confidence", 0) or 0)
    if conf >= gate:
        return moment.get("caption_specific") or moment.get("caption_generic") or ""
    return moment.get("caption_generic") or ""


def select_moments(moments, max_clips, trans_dur):
    """Rank by visual_score, then trim the tail so the montage fits under 90s."""
    ranked = sorted(
        moments, key=lambda m: float(m.get("visual_score", 0) or 0), reverse=True
    )
    ranked = ranked[:max_clips]

    def total(ms):
        raw = sum(parse_ts(m["end_time"]) - parse_ts(m["start_time"]) for m in ms)
        # each transition overlaps two clips, shaving trans_dur off the total
        return raw - max(0, len(ms) - 1) * trans_dur

    while len(ranked) > 1 and total(ranked) > REELS_MAX:
        ranked.pop()  # drop the lowest-scored remaining clip
    return ranked, total(ranked)


def normalize_clip(video, moment, idx, gate, tmpdir, captions):
    """Cut one moment and force it to the canonical format for stitching."""
    start = parse_ts(moment["start_time"])
    end = parse_ts(moment["end_time"])
    dur = max(0.1, end - start)
    out = os.path.join(tmpdir, f"clip_{idx:02d}.mp4")

    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},fps={FPS},format=yuv420p,setsar=1"
    )

    if captions:
        label = burned_label(moment, gate)
        if label:
            tf = os.path.join(tmpdir, f"label_{idx:02d}.txt")
            with open(tf, "w") as fh:
                fh.write(label)
            vf += (
                f",drawtext=fontfile={FONT}:textfile={tf}:"
                f"fontcolor=white:fontsize=58:line_spacing=8:"
                f"box=1:boxcolor=black@0.55:boxborderw=26:"
                f"x=(w-text_w)/2:y=h-360"
            )
    vf += ",setpts=PTS-STARTPTS"

    run([
        "ffmpeg", "-y", "-i", video,
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-vf", vf,
        "-af", "aresample=44100,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS",
        "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", "-video_track_timescale", "30000",
        out,
    ])
    return out, dur


def stitch(clips, durs, transition, trans_dur, out):
    """Chain xfade (video) + acrossfade (audio) across all clips."""
    if len(clips) == 1:
        run(["ffmpeg", "-y", "-i", clips[0], "-c", "copy", out])
        return

    inputs = []
    for c in clips:
        inputs += ["-i", c]

    v_parts, a_parts = [], []
    prev_v, prev_a = "[0:v]", "[0:a]"
    cum = durs[0]
    for i in range(1, len(clips)):
        offset = cum - trans_dur          # start the overlap trans_dur early
        vlab = f"[v{i}]"
        alab = f"[a{i}]"
        v_parts.append(
            f"{prev_v}[{i}:v]xfade=transition={transition}:"
            f"duration={trans_dur}:offset={offset:.3f}{vlab}"
        )
        a_parts.append(f"{prev_a}[{i}:a]acrossfade=d={trans_dur}{alab}")
        prev_v, prev_a = vlab, alab
        cum += durs[i] - trans_dur        # xfade consumes trans_dur of overlap

    filtergraph = ";".join(v_parts + a_parts)
    run([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filtergraph,
        "-map", prev_v, "-map", prev_a,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        out,
    ])


def main():
    ap = argparse.ArgumentParser(description="Build a 9:16 fight-highlight reel.")
    ap.add_argument("--video", required=True, help="source video file")
    ap.add_argument("--analysis", required=True, help="analysis JSON file")
    ap.add_argument("--out", default="reel.mp4", help="output reel path")
    ap.add_argument("--max-clips", type=int, default=6)
    ap.add_argument("--gate", type=float, default=0.8,
                    help="confidence needed to burn the specific technique name")
    ap.add_argument("--transition", default="smoothleft",
                    help="xfade style: smoothleft, slideleft, fade, circleopen, wiperight ...")
    ap.add_argument("--trans-dur", type=float, default=0.4)
    ap.add_argument("--no-captions", action="store_true")
    args = ap.parse_args()

    with open(args.analysis) as fh:
        data = json.load(fh)
    moments = data["moments"] if isinstance(data, dict) else data
    if not moments:
        raise SystemExit("No moments in analysis JSON.")

    picked, total = select_moments(moments, args.max_clips, args.trans_dur)
    print(f"Selected {len(picked)} clip(s), ~{total:.1f}s reel:")
    for m in picked:
        print(f"  score {m.get('visual_score')}  "
              f"{parse_ts(m['start_time']):.1f}-{parse_ts(m['end_time']):.1f}s  "
              f"conf {m.get('technique_confidence')}  -> "
              f"{burned_label(m, args.gate) or '(no label)'}")

    if total > REELS_MAX:
        print(f"  ! still {total:.1f}s (>90s); shorten clips or lower --max-clips")
    if total < REELS_MIN:
        print(f"  ! only {total:.1f}s (<5s); may not land in the Reels tab")

    with tempfile.TemporaryDirectory() as tmp:
        clips, durs = [], []
        for i, m in enumerate(picked):
            c, d = normalize_clip(args.video, m, i, args.gate, tmp,
                                  not args.no_captions)
            clips.append(c)
            durs.append(d)
        stitch(clips, durs, args.transition, args.trans_dur, args.out)

    # Instagram post caption (emoji kept) -> sidecar for the publish stage.
    lines = [post_caption(m, args.gate) for m in picked]
    lines = [l for l in lines if l]
    cap_path = args.out + ".caption.txt"
    with open(cap_path, "w") as fh:
        fh.write((lines[0] if lines else "Fight of the day") + "\n\n")
        fh.write("#mma #bjj #grappling #muaythai #ignitionmma\n")

    print(f"\nReel:    {args.out}")
    print(f"Caption: {cap_path}")


if __name__ == "__main__":
    main()
