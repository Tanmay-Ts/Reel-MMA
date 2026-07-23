#!/usr/bin/env python3
"""
r2_sync.py  -  Cloudflare R2 as the drop folder for the reel pipeline

Bucket layout. The date folder is the unit of work, and the presence of a
matching outgoing/ folder is how a job knows it is already done, so no database
is needed to track state:

    incoming/2026-07-23/IMG_2402.MOV
    incoming/2026-07-23/IMG_2403.MOV
    incoming/2026-07-23/context.txt      (optional, session description)
    outgoing/2026-07-23/reel.mp4         (written by this pipeline)

R2 is S3-compatible, so this speaks S3 via boto3.

Environment (same names as the repo secrets):
    R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET

Usage:
    python3 r2_sync.py pending
        -> prints one folder name per line that still needs a reel

    python3 r2_sync.py download 2026-07-23 --dest work/
        -> pulls that folder's clips into work/clips/ and context.txt into work/

    python3 r2_sync.py upload 2026-07-23 --reel reel.mp4 --caption reel.mp4.caption.txt
        -> writes the finished reel back under outgoing/2026-07-23/
"""

import argparse
import os
import sys

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
INCOMING = "incoming/"
OUTGOING = "outgoing/"
CONTEXT_NAME = "context.txt"
REEL_NAME = "reel.mp4"


def client():
    """R2 speaks S3. region_name must be 'auto' for R2."""
    import boto3
    from botocore.config import Config

    for var in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY", "R2_SECRET_KEY", "R2_BUCKET"):
        if not os.environ.get(var):
            sys.stderr.write(f"Missing environment variable: {var}\n")
            raise SystemExit(1)

    # R2_ENDPOINT lets tests point at a mock, and allows any other
    # S3-compatible host without touching this code.
    endpoint = os.environ.get("R2_ENDPOINT") or (
        f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def list_keys(s3, bucket, prefix):
    """All object keys under a prefix, handling pagination."""
    keys = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            keys.append(obj["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def folder_of(key, prefix):
    """incoming/2026-07-23/a.MOV -> 2026-07-23 ; returns None if not nested."""
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix):]
    if "/" not in rest:
        return None
    return rest.split("/", 1)[0]


def find_pending(incoming_keys, outgoing_keys):
    """Folders that have at least one video but no finished reel yet.

    Pure function so it can be tested without touching the network.
    """
    have_video = set()
    for k in incoming_keys:
        f = folder_of(k, INCOMING)
        if f and k.lower().endswith(VIDEO_EXTS):
            have_video.add(f)

    done = set()
    for k in outgoing_keys:
        f = folder_of(k, OUTGOING)
        if f and k.endswith(REEL_NAME):
            done.add(f)

    return sorted(have_video - done)


def cmd_pending(args):
    s3 = client()
    bucket = os.environ["R2_BUCKET"]
    pending = find_pending(
        list_keys(s3, bucket, INCOMING),
        list_keys(s3, bucket, OUTGOING),
    )
    for f in pending:
        print(f)
    if not pending:
        sys.stderr.write("Nothing pending.\n")


def cmd_download(args):
    s3 = client()
    bucket = os.environ["R2_BUCKET"]
    prefix = f"{INCOMING}{args.folder}/"
    keys = list_keys(s3, bucket, prefix)
    if not keys:
        sys.stderr.write(f"No objects under {prefix}\n")
        raise SystemExit(1)

    clipdir = os.path.join(args.dest, "clips")
    os.makedirs(clipdir, exist_ok=True)

    n_clips = 0
    got_context = False
    for k in keys:
        name = k[len(prefix):]
        if not name or name.endswith("/"):
            continue
        if k.lower().endswith(VIDEO_EXTS):
            dest = os.path.join(clipdir, os.path.basename(name))
            s3.download_file(bucket, k, dest)
            n_clips += 1
            print(f"  clip    {name}")
        elif os.path.basename(name) == CONTEXT_NAME:
            dest = os.path.join(args.dest, CONTEXT_NAME)
            s3.download_file(bucket, k, dest)
            got_context = True
            print(f"  context {name}")

    print(f"Downloaded {n_clips} clip(s)"
          + (" plus session context" if got_context else " (no context.txt)"))
    if n_clips == 0:
        sys.stderr.write("No video files found in that folder.\n")
        raise SystemExit(1)


def cmd_upload(args):
    s3 = client()
    bucket = os.environ["R2_BUCKET"]
    prefix = f"{OUTGOING}{args.folder}/"

    uploads = [(args.reel, REEL_NAME, "video/mp4")]
    if args.caption and os.path.isfile(args.caption):
        uploads.append((args.caption, "caption.txt", "text/plain"))

    for local, name, ctype in uploads:
        if not os.path.isfile(local):
            sys.stderr.write(f"Missing local file: {local}\n")
            raise SystemExit(1)
        s3.upload_file(local, bucket, prefix + name,
                       ExtraArgs={"ContentType": ctype})
        print(f"  uploaded {prefix}{name}")


def main():
    ap = argparse.ArgumentParser(description="R2 drop-folder sync for the reel pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("pending", help="list folders that still need a reel")

    d = sub.add_parser("download", help="fetch one folder's clips and context")
    d.add_argument("folder")
    d.add_argument("--dest", default=".")

    u = sub.add_parser("upload", help="write a finished reel back to the bucket")
    u.add_argument("folder")
    u.add_argument("--reel", required=True)
    u.add_argument("--caption", default=None)

    args = ap.parse_args()
    {"pending": cmd_pending, "download": cmd_download, "upload": cmd_upload}[args.cmd](args)


if __name__ == "__main__":
    main()
