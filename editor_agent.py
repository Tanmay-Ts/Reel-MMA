#!/usr/bin/env python3
"""
editor_agent.py  -  the master reel editor

The plain editor is mechanical: rank by visual_score, take the top N, apply the
same padding and the same transition everywhere. It has no sense of the reel as
a whole, which is why a session full of double legs produces three near-identical
clips in a row.

This stage reads ALL detected moments and decides the edit: which clips, in what
order, how long each should breathe, and which transition to cut on. It thinks
about the things a human editor would:

  - open on a hook, not necessarily the highest-scored clip
  - vary technique types so similar moves do not stack back to back
  - give a submission a long tail and a quick takedown a short one
  - pace the cuts, quick transitions build energy, slower ones let a finish land
  - end on something that resolves

Output is an edit decision list (edl.json) that reel_editor.py consumes with
--edl. Clips are referenced by their index in the input moments array, so the
agent never has to retype timestamps and cannot corrupt them.

This call is TEXT ONLY. It reads the descriptions the vision stage already
wrote, so it sends no images and costs a fraction of the analyzer.

Usage:
    export OPENAI_API_KEY=sk-...
    python3 editor_agent.py --analysis moments.json --out edl.json
    python3 reel_editor.py --video combined.mp4 --analysis moments.json \
        --edl edl.json --out reel.mp4

Options:
    --target-clips 6      roughly how many clips the reel should hold
    --max-duration 60     upper bound in seconds
    --style hype          hype | technical | cinematic
    --context "..."       session description, same one given to the analyzer
    --dry-run             skip the API, emit a sensible score-ranked EDL
"""

import argparse
import json
import os
import sys

TRANSITIONS = ["smoothleft", "smoothright", "slideleft", "slideup",
               "fade", "circleopen", "wiperight", "dissolve"]

# Moments that belong at the END of a reel. A prize giving is the resolution of
# an event, so it reads as an ending regardless of how it scored.
CLOSING_TYPES = {"gift_presentation", "award", "ceremony", "medal",
                 "team_photo", "trophy"}

# Calm moments that carry feeling rather than action. They read over several
# seconds and look broken when cut tight, so they get generous padding.
ATMOSPHERE_TYPES = {"celebration", "crowd_reaction", "announcement",
                    "handshake", "group_moment"} | CLOSING_TYPES


def is_closing(m):
    return str(m.get("action_type", "")).lower() in CLOSING_TYPES


def is_atmosphere(m):
    return str(m.get("action_type", "")).lower() in ATMOSPHERE_TYPES


def arrange_narrative(clips, moments):
    """Action first, then the occasion's ending.

    Closing moments (prize giving, trophies, team photos) are moved to the end
    and kept in the order they happened among themselves. Everything else keeps
    the order the editor chose. This is what makes a reel feel like it finishes
    rather than just stopping.
    """
    action = [c for c in clips if not is_closing(moments[c["index"]])]
    closing = [c for c in clips if is_closing(moments[c["index"]])]
    closing.sort(key=lambda c: _parse_ts(moments[c["index"]]["start_time"]))
    if closing:
        # the first clip after the action should settle, not whip-cut
        closing[0]["transition"] = "fade"
    return action + closing


def top_up_to_minimum(clips, moments, min_duration, trans_dur=0.4):
    """Add unused moments until the reel reaches min_duration.

    The agent optimises for punchiness and will happily hand back 20 seconds.
    When a minimum is asked for, this fills it deterministically rather than
    hoping the model counts seconds correctly, which it does not.
    """
    def total(cs):
        return sum(_dur(moments[c["index"]]) + c["pad_start"] + c["pad_end"]
                   for cs_ in [cs] for c in cs_) - max(0, len(cs) - 1) * trans_dur

    used = {c["index"] for c in clips}
    spare = [i for i in range(len(moments)) if i not in used]
    # best first, but atmosphere counts extra since it adds texture not repetition
    spare.sort(key=lambda i: (float(moments[i].get("visual_score", 0) or 0)
                              + (1.5 if is_atmosphere(moments[i]) else 0)),
               reverse=True)

    added = 0
    for i in spare:
        if total(clips) >= min_duration:
            break
        m = moments[i]
        atmos = is_atmosphere(m)
        clips.append({
            "index": i,
            "pad_start": 2.0 if atmos else 1.5,
            "pad_end": 3.0 if atmos else 2.0,
            "transition": "fade" if atmos else "smoothleft",
            "reason": "added to reach the requested reel length",
        })
        added += 1
    return clips, added, total(clips)


CEREMONY_PROMPT = """\
You are editing a CEREMONY reel: a medal presentation, belt promotion, awards, or
prize giving. This is NOT a highlights reel and the usual rules are inverted.

1. CHRONOLOGICAL ORDER IS MANDATORY. Present the moments in the order they
   happened. Never reorder for impact. A ceremony reordered makes no sense.

2. INCLUDE EVERYONE. Every person receiving something must appear. Do not drop
   a moment because it scored low. Missing someone out is the worst possible
   failure here, far worse than a slightly long reel. Only drop a moment if it
   shows no recipient at all.

3. LET MOMENTS BREATHE. A handover, a handshake, a bow, a photo. These need time
   to read. Use generous padding, 2 to 3 seconds either side, so faces and
   reactions are visible. Never clip a presentation short.

4. TRANSITIONS SHOULD BE CALM. Use fade or dissolve. Hard whip transitions are
   wrong for a ceremony.

Return the same JSON structure. Order the "clips" array chronologically by the
moment's start time.
"""

STYLE_NOTES = {
    "hype": "Fast and punchy. Favour short clips, quick cuts, and hard "
            "transitions. Open on the single most explosive moment.",
    "technical": "Let technique read clearly. Favour slightly longer clips and "
                 "softer transitions so the mechanics are visible. Group "
                 "related techniques so a viewer can compare them.",
    "cinematic": "Build. Open strong but not the strongest, vary the rhythm, "
                 "and save the best finish for last. Favour smooth transitions.",
}

SYSTEM_PROMPT = """\
You are a master highlight reel editor for a mixed martial arts and Brazilian
jiu-jitsu gym. You are given a list of detected moments from one training
session, each with an index, a score, a technique, a confidence, a duration, and
a description.

Your job is to design the EDIT. You are not ranking clips, you are building a
reel that people actually want to watch to the end.

Judge on these, in order:

1. OPENING HOOK. The first clip decides whether anyone keeps watching. Pick
   something immediately legible and explosive. A clip that needs context to
   appreciate is a bad opener even if it scored highly.

2. VARIETY. Do not place similar techniques back to back. Three double leg
   takedowns in a row is boring even if all three scored well. Alternate
   between different action types, standing and ground, fight action and
   conditioning. If the session only contains one kind of action, use fewer
   clips rather than repeating the same look.

3. PACING. Vary clip length deliberately. A run of quick cuts builds energy; a
   longer clip after them lands a big finish. Do not make every clip the same
   length.

4. BREATHING ROOM PER CLIP. Clips must NEVER feel clipped. Always start BEFORE
   the action begins so the viewer sees it coming, and always end AFTER it
   resolves. Around 2 seconds either side is the baseline. Then adjust:
   - Submissions and chokes need a LONGER tail, 3 to 5 seconds, the finish and
     the tap are the whole payoff
   - Throws and takedowns need about 2 to 3 seconds so the landing lands
   - Strikes and quick exchanges can use about 1.5 to 2 seconds, they resolve fast
   - Conditioning and rope climbs need 2 to 3 seconds to show completion
   Never set pad_start below 1.0 or pad_end below 1.5. A clip that starts on the
   action already in progress looks broken.

5. THE OCCASION'S ENDING. If the session contains a prize giving, trophy,
   medals, or a team photo, those belong at the END of the reel, in the order
   they happened. They are the resolution of the event. Never open with them.

6. ATMOSPHERE IS NOT FILLER. Celebrations, cheering, announcements, huddles and
   handshakes carry the feeling of the day. Include them alongside the action,
   and give them longer clips than you would give a fast technique. They read
   over several seconds and look broken when cut tight.

7. ENDING. Finish on something conclusive. A clean finish, the best technique,
   or the occasion's closing moment.

You may DROP clips. A shorter reel of strong varied moments beats a longer one
padded with repetition. You may also reorder freely; source order does not matter.

Return ONLY a JSON object:

{
  "reasoning": "two or three sentences on the shape you chose and why",
  "clips": [
    {
      "index": 0,
      "pad_start": 0.5,
      "pad_end": 1.5,
      "transition": "smoothleft",
      "reason": "short note on why this clip sits here"
    }
  ]
}

Rules:
- "index" must be an index from the input list. Never invent one. Never repeat one.
- pad_start between 1.0 and 4.0. pad_end between 1.5 and 6.0.
- "transition" is the cut INTO this clip and must be one of:
  smoothleft, smoothright, slideleft, slideup, fade, circleopen, wiperight, dissolve.
  The first clip's transition is ignored.
- Order the "clips" array in the order they should appear in the reel.
"""


def summarise(moments):
    """Compact text view of the moments for the agent. Indexes are the contract."""
    lines = []
    for i, m in enumerate(moments):
        dur = _dur(m)
        lines.append(
            f"[{i}] score={m.get('visual_score')} "
            f"type={m.get('action_type')} "
            f"technique={m.get('technique_guess')} "
            f"conf={m.get('technique_confidence')} "
            f"dur={dur:.1f}s :: {m.get('description', '')}"
        )
    return "\n".join(lines)


def _parse_ts(ts):
    if isinstance(ts, (int, float)):
        return float(ts)
    parts = [float(p) for p in str(ts).split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def _dur(m):
    return max(0.0, _parse_ts(m["end_time"]) - _parse_ts(m["start_time"]))


def validate(edl, n_moments, max_duration, moments):
    """Never trust the model's structure. Repair or drop anything invalid."""
    clips, seen, problems = [], set(), []

    for c in edl.get("clips", []):
        try:
            idx = int(c.get("index"))
        except (TypeError, ValueError):
            problems.append(f"non-integer index {c.get('index')!r}, dropped")
            continue
        if not (0 <= idx < n_moments):
            problems.append(f"index {idx} out of range, dropped")
            continue
        if idx in seen:
            problems.append(f"duplicate index {idx}, dropped")
            continue
        seen.add(idx)

        def num(key, lo, hi, default):
            try:
                return min(hi, max(lo, float(c.get(key, default))))
            except (TypeError, ValueError):
                return default

        tr = c.get("transition", "smoothleft")
        if tr not in TRANSITIONS:
            problems.append(f"unknown transition {tr!r} on clip {idx}, using smoothleft")
            tr = "smoothleft"

        clips.append({
            "index": idx,
            "pad_start": num("pad_start", 0.5, 4.0, 2.0),
            "pad_end": num("pad_end", 0.5, 6.0, 2.0),
            "transition": tr,
            "reason": str(c.get("reason", ""))[:200],
        })

    # enforce the duration ceiling by dropping from the end
    def total(cs):
        return sum(_dur(moments[c["index"]]) + c["pad_start"] + c["pad_end"]
                   for c in cs)

    while len(clips) > 1 and total(clips) > max_duration:
        dropped = clips.pop()
        problems.append(f"over {max_duration:.0f}s, dropped trailing clip "
                        f"{dropped['index']}")

    return clips, problems


def chronological_edl(moments):
    """Ceremony fallback: every moment, in the order it happened, calm cuts.

    Deliberately keeps ALL moments. Dropping a recipient from a medal reel is
    the one unforgivable failure, so length yields to completeness here.
    """
    order = sorted(range(len(moments)), key=lambda i: _parse_ts(moments[i]["start_time"]))
    return {
        "reasoning": "Ceremony: chronological, every recipient kept.",
        "clips": [{
            "index": i,
            "pad_start": 2.0,
            "pad_end": 2.5,
            "transition": "fade" if n % 2 == 0 else "dissolve",
            "reason": "ceremony, in sequence",
        } for n, i in enumerate(order)],
    }


def fallback_edl(moments, target, max_duration):
    """Score-ranked EDL with type-aware padding. Used for --dry-run and if the
    API fails, so the pipeline degrades instead of breaking."""
    long_tail = {"submission", "escape", "knockdown"}
    short_tail = {"strike", "kick", "knee", "combination"}
    order = sorted(range(len(moments)),
                   key=lambda i: float(moments[i].get("visual_score", 0) or 0),
                   reverse=True)[:target]
    clips = []
    for n, i in enumerate(order):
        at = str(moments[i].get("action_type", "")).lower()
        pad_end = 4.0 if at in long_tail else (1.8 if at in short_tail else 2.5)
        clips.append({
            "index": i,
            "pad_start": 2.0,
            "pad_end": pad_end,
            "transition": TRANSITIONS[n % 4],
            "reason": "score-ranked fallback",
        })
    return {"reasoning": "Fallback: ranked by score with type-aware padding.",
            "clips": clips}


def main():
    ap = argparse.ArgumentParser(description="Design the reel edit from detected moments.")
    ap.add_argument("--analysis", required=True, help="moments.json from analyzer.py")
    ap.add_argument("--out", default="edl.json")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--target-clips", type=int, default=6)
    ap.add_argument("--max-duration", type=float, default=60.0)
    ap.add_argument("--min-duration", type=float, default=0.0,
                    help="force the reel to at least this many seconds by "
                         "adding more moments. 0 disables.")
    ap.add_argument("--style", default="hype", choices=list(STYLE_NOTES))
    ap.add_argument("--context", default=None)
    ap.add_argument("--context-file", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.analysis) as fh:
        data = json.load(fh)
    moments = data["moments"] if isinstance(data, dict) else data
    if not moments:
        sys.stderr.write("No moments to edit.\n")
        raise SystemExit(1)

    activity = data.get("activity", {}) if isinstance(data, dict) else {}
    primary = str(activity.get("primary", "")).lower()
    ceremony_mode = primary == "ceremony"
    if ceremony_mode:
        print("Ceremony detected: chronological order, everyone included, "
              "generous padding")

    context = args.context
    if args.context_file and os.path.isfile(args.context_file):
        with open(args.context_file) as fh:
            context = fh.read()

    print(f"{len(moments)} moment(s) available, target ~{args.target_clips} clips, "
          f"style={args.style}")

    if args.dry_run:
        edl = (chronological_edl(moments) if ceremony_mode
               else fallback_edl(moments, args.target_clips, args.max_duration))
    else:
        from openai import OpenAI
        user = (
            f"Session moments:\n{summarise(moments)}\n\n"
            f"Target roughly {args.target_clips} clips, hard maximum "
            f"{args.max_duration:.0f} seconds total.\n"
            + (f"The reel MUST be at least {args.min_duration:.0f} seconds long. "
               f"Include enough moments, and long enough clips, to reach that. "
               f"Do not hand back a short punchy reel when a longer one is asked for.\n"
               if args.min_duration else "")
            + f"Editing style: {args.style}. {STYLE_NOTES[args.style]}"
        )
        if ceremony_mode:
            user = (
                f"Session moments:\n{summarise(moments)}\n\n"
                f"Include every recipient. Hard maximum "
                f"{max(args.max_duration, 90):.0f} seconds."
            )
        if context:
            user += f"\n\nSession context from the user: {context.strip()}"

        try:
            resp = OpenAI().chat.completions.create(
                model=args.model,
                temperature=0.4,
                response_format={"type": "json_object"},
                messages=[{"role": "system",
                           "content": CEREMONY_PROMPT if ceremony_mode
                           else SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
            )
            edl = json.loads(resp.choices[0].message.content)
        except Exception as exc:
            sys.stderr.write(f"[warn] editor agent failed ({exc}); using fallback\n")
            edl = (chronological_edl(moments) if ceremony_mode
                   else fallback_edl(moments, args.target_clips, args.max_duration))

    # A ceremony must never lose a recipient to a length cap.
    cap = 1e9 if ceremony_mode else args.max_duration
    clips, problems = validate(edl, len(moments), cap, moments)
    if not clips:
        sys.stderr.write("[warn] nothing usable from the agent; using fallback\n")
        edl = (chronological_edl(moments) if ceremony_mode
               else fallback_edl(moments, args.target_clips, args.max_duration))
        clips, problems = validate(edl, len(moments), cap, moments)

    # Prize giving and team photos belong at the end, whatever the agent chose.
    if not ceremony_mode:
        clips = arrange_narrative(clips, moments)

    added = 0
    if args.min_duration and not ceremony_mode:
        clips, added, _ = top_up_to_minimum(clips, moments, args.min_duration,
                                            trans_dur=0.4)
        if added:
            clips = arrange_narrative(clips, moments)

    out = {"reasoning": str(edl.get("reasoning", ""))[:600], "clips": clips}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    if out["reasoning"]:
        print(f"\nEditor: {out['reasoning']}")
    print(f"\n{len(clips)} clip(s) in the edit:")
    total = 0.0
    for n, c in enumerate(clips):
        m = moments[c["index"]]
        length = _dur(m) + c["pad_start"] + c["pad_end"]
        total += length
        cut = "open" if n == 0 else c["transition"]
        print(f"  {n+1}. [{c['index']}] {m.get('technique_guess')} "
              f"({m.get('action_type')}) {length:.1f}s  cut={cut}")
        if c["reason"]:
            print(f"       {c['reason']}")
    if added:
        print(f"  (+{added} moment(s) added to reach the {args.min_duration:.0f}s minimum)")
    print(f"\nApprox reel length: {total:.1f}s  ->  {args.out}")
    if args.min_duration and total < args.min_duration:
        print(f"  ! only {total:.1f}s; the session did not contain enough "
              f"usable moments to reach {args.min_duration:.0f}s")
    for p in problems:
        print(f"  [repaired] {p}")


if __name__ == "__main__":
    main()