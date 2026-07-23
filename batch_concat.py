#!/usr/bin/env python3
"""
batch_concat.py  -  Join many raw clips into one combined.mp4

Option 1 of the "one reel from many clips" plan: concatenate first, then run the
single combined video through analyzer.py + reel_editor.py. This way the BEST
moments across ALL clips compete for reel slots, instead of forcing one pick per
clip.

Clips from different days / cameras / orientations usually differ in resolution,
fps, or codec, and ffmpeg's fast concat can't join those cleanly. So this script
NORMALISES every clip to one common format first (letterboxed onto a shared
canvas so nothing is cropped), then concatenates. Slower than a raw concat, but
it never breaks on mixed input.

Usage:
    # join every video in a folder (sorted by name)
    python3 batch_concat.py --dir clips/ --out combined.mp4

    # or list specific files in the order you want them
    python3 batch_concat.py --files a.mp4 b.mp4 c.mp4 --out combined.mp4

Then run the normal pipeline on the result:
    python3 analyzer.py --video combined.mp4 --out moments.json --model gpt-4o
    python3 reel_editor.py --video combined.mp4 --analysis moments.json --out reel.mp4

Options:
    --width 1280 --height 720   shared canvas (default 1280x720 landscape)
    --fps 30                    shared framerate
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write("\n[ffmpeg error]\n" + p.stderr[-1800:] + "\n")
        raise SystemExit(1)
    return p


def collect_clips(args):
    if args.files:
        clips = args.files
    else:
        found = []
        for f in sorted(os.listdir(args.dir)):
            if f.lower().endswith(VIDEO_EXTS):
                found.append(os.path.join(args.dir, f))
        clips = found
    clips = [c for c in clips if os.path.isfile(c)]
    if not clips:
        raise SystemExit("No video clips found. Check --dir or --files.")
    return clips


def normalize(clip, idx, tmpdir, W, H, fps):
    """Scale-to-fit + letterbox onto a WxH canvas, unify fps/codec/audio.

    Letterbox (pad) rather than crop so no action is lost from clips whose
    aspect ratio differs from the canvas. Adds a silent track if a clip has
    no audio, so the concat doesn't desync."""
    out = os.path.join(tmpdir, f"norm_{idx:03d}.mp4")
    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={fps},format=yuv420p,setsar=1"
    )
    run([
        "ffmpeg", "-y", "-i", clip,
        "-f", "lavfi", "-t", "0.1", "-i", "anullsrc=r=44100:cl=stereo",
        "-vf", vf,
        "-map", "0:v:0",
        # use the clip's own audio if present, else the silent source
        "-map", "0:a:0?",
        "-shortest" if False else "-avoid_negative_ts", "make_zero",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        "-video_track_timescale", "30000",
        out, "-loglevel", "error",
    ])
    # guarantee an audio track exists (some clips truly have none)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", out],
        capture_output=True, text=True,
    )
    if not probe.stdout.strip():
        fixed = os.path.join(tmpdir, f"norm_{idx:03d}_a.mp4")
        run([
            "ffmpeg", "-y", "-i", out,
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-map", "0:v:0", "-map", "1:a:0", "-shortest",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            fixed, "-loglevel", "error",
        ])
        return fixed
    return out


def main():
    ap = argparse.ArgumentParser(description="Concatenate many clips into one video.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dir", help="folder of clips (joined in filename order)")
    g.add_argument("--files", nargs="+", help="explicit clip list, in order")
    ap.add_argument("--out", default="combined.mp4")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    clips = collect_clips(args)
    print(f"Joining {len(clips)} clip(s) onto a {args.width}x{args.height} canvas:")
    for c in clips:
        print(f"  {c}")

    with tempfile.TemporaryDirectory() as tmp:
        normed = []
        for i, c in enumerate(clips):
            print(f"  normalising {i+1}/{len(clips)} ...")
            normed.append(normalize(c, i, tmp, args.width, args.height, args.fps))

        listfile = os.path.join(tmp, "list.txt")
        with open(listfile, "w") as fh:
            for n in normed:
                fh.write(f"file '{os.path.abspath(n)}'\n")

        run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
            "-c", "copy", "-movflags", "+faststart",
            args.out, "-loglevel", "error",
        ])

    # report the joined duration
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", args.out],
        capture_output=True, text=True,
    )
    dur = float(p.stdout.strip()) if p.stdout.strip() else 0
    print(f"\nCombined -> {args.out}  ({dur:.1f}s total)")
    print("Next:")
    print(f"  python3 analyzer.py --video {args.out} --out moments.json --model gpt-4o")
    print(f"  python3 reel_editor.py --video {args.out} --analysis moments.json --out reel.mp4")


if __name__ == "__main__":
    main()