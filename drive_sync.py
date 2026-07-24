#!/usr/bin/env python3
"""
drive_sync.py  -  Google Drive is where the sensei uploads. This mirrors it to R2.

The sensei drops clips into a dated folder inside one shared Drive folder:

    Ignition Reels/
        2026-07-24 belt promotion/
            IMG_2402.MOV
            IMG_2403.MOV
        2026-07-26 sparring/
            IMG_2410.MOV

The folder NAME carries everything the pipeline needs. He is already typing it,
so it costs him nothing:

    "2026-07-24 belt promotion"
      -> session id : 2026-07-24-belt-promotion
      -> context    : "belt promotion"

The context is written into R2 as context.txt, which the analyzer already reads
via --context-file. So no new plumbing downstream: this script only mirrors
Drive into R2's incoming/ area, and the existing pipeline takes over unchanged.

A folder is skipped if it has already been mirrored, so re-running is safe and
cheap.

Setup, once:
  1. Google Cloud console -> create a service account -> download its JSON key
  2. Enable the Google Drive API for that project
  3. In Drive, share the parent folder with the service account's email
     (Viewer is enough) so it can read the sessions
  4. Put the JSON key in the repo secret GOOGLE_SERVICE_ACCOUNT_JSON
     and the parent folder's id in DRIVE_PARENT_FOLDER_ID

Environment:
    GOOGLE_SERVICE_ACCOUNT_JSON   the service account key, as raw JSON
    DRIVE_PARENT_FOLDER_ID        id of the shared parent folder
    R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET

Usage:
    python3 drive_sync.py list          # what is in Drive, and what is new
    python3 drive_sync.py sync          # mirror every new session into R2
    python3 drive_sync.py sync --folder "2026-07-24 belt promotion"
"""

import argparse
import io
import json
import os
import re
import sys

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
INCOMING = "incoming/"
OUTGOING = "outgoing/"
FOLDER_MIME = "application/vnd.google-apps.folder"

# leading date in the folder name, e.g. "2026-07-24" or "24-07-2026" or "2026_07_24"
DATE_RE = re.compile(r"^\s*(\d{4}[-_/.]\d{1,2}[-_/.]\d{1,2}|\d{1,2}[-_/.]\d{1,2}[-_/.]\d{4})\s*")


def parse_folder_name(name):
    """Split a Drive folder name into a session id and a context sentence.

    The date prefix is optional. Whatever follows it is treated as the sensei's
    description of the session and becomes the analyzer's context.

        "2026-07-24 belt promotion"  -> ("2026-07-24-belt-promotion", "belt promotion")
        "2026-07-26 sparring"        -> ("2026-07-26-sparring", "sparring")
        "kabaddi friendly"           -> ("kabaddi-friendly", "kabaddi friendly")
        "2026-07-24"                 -> ("2026-07-24", "")
    """
    raw = (name or "").strip()
    m = DATE_RE.match(raw)
    context = raw[m.end():].strip() if m else raw

    # session id: safe for object keys, no spaces or awkward characters
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-").lower()
    slug = re.sub(r"-{2,}", "-", slug) or "session"
    return slug, context


def drive_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        sys.stderr.write("Missing GOOGLE_SERVICE_ACCOUNT_JSON\n")
        raise SystemExit(1)
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        # allow pointing at a file path as well as inlining the JSON
        with open(raw) as fh:
            info = json.load(fh)

    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive.readonly"])
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def r2_client():
    import boto3
    from botocore.config import Config

    for var in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY", "R2_SECRET_KEY", "R2_BUCKET"):
        if not os.environ.get(var):
            sys.stderr.write(f"Missing environment variable: {var}\n")
            raise SystemExit(1)

    endpoint = os.environ.get("R2_ENDPOINT") or (
        f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com")
    return boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
        region_name="auto", config=Config(signature_version="s3v4"))


def list_r2_keys(s3, bucket, prefix):
    keys, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def already_synced(sessions, incoming_keys, outgoing_keys):
    """A session is done if it is already mirrored in, or already has a reel out."""
    have_in = {k[len(INCOMING):].split("/", 1)[0]
               for k in incoming_keys if k.startswith(INCOMING) and "/" in k[len(INCOMING):]}
    have_out = {k[len(OUTGOING):].split("/", 1)[0]
                for k in outgoing_keys if k.startswith(OUTGOING) and "/" in k[len(OUTGOING):]}
    done = have_in | have_out
    return [s for s in sessions if s["session_id"] not in done]


def list_drive_sessions(svc, parent_id):
    """Every sub-folder of the shared parent, with its video files."""
    sessions = []
    page = None
    while True:
        resp = svc.files().list(
            q=f"'{parent_id}' in parents and mimeType='{FOLDER_MIME}' and trashed=false",
            fields="nextPageToken, files(id, name)",
            pageSize=100, pageToken=page,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        for f in resp.get("files", []):
            session_id, context = parse_folder_name(f["name"])
            sessions.append({
                "drive_id": f["id"], "name": f["name"],
                "session_id": session_id, "context": context, "files": [],
            })
        page = resp.get("nextPageToken")
        if not page:
            break

    for s in sessions:
        page = None
        while True:
            resp = svc.files().list(
                q=f"'{s['drive_id']}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, size, mimeType)",
                pageSize=200, pageToken=page,
                supportsAllDrives=True, includeItemsFromAllDrives=True,
            ).execute()
            for f in resp.get("files", []):
                if f["name"].lower().endswith(VIDEO_EXTS):
                    s["files"].append(f)
            page = resp.get("nextPageToken")
            if not page:
                break
    return sessions


def stream_to_r2(svc, s3, bucket, file_obj, key):
    """Download from Drive into memory, then put to R2."""
    from googleapiclient.http import MediaIoBaseDownload

    request = svc.files().get_media(fileId=file_obj["id"])
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request, chunksize=8 * 1024 * 1024)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    s3.upload_fileobj(buf, bucket, key)


def cmd_list(args):
    svc = drive_service()
    s3 = r2_client()
    bucket = os.environ["R2_BUCKET"]
    sessions = list_drive_sessions(svc, os.environ["DRIVE_PARENT_FOLDER_ID"])
    if not sessions:
        print("No session folders found in the shared Drive folder.")
        return
    new = {s["session_id"] for s in already_synced(
        sessions, list_r2_keys(s3, bucket, INCOMING), list_r2_keys(s3, bucket, OUTGOING))}
    print(f"{len(sessions)} session folder(s) in Drive:\n")
    for s in sessions:
        mark = "NEW " if s["session_id"] in new else "done"
        print(f"  [{mark}] {s['name']}")
        print(f"          id      : {s['session_id']}")
        print(f"          context : {s['context'] or '(none)'}")
        print(f"          clips   : {len(s['files'])}")


def cmd_sync(args):
    svc = drive_service()
    s3 = r2_client()
    bucket = os.environ["R2_BUCKET"]

    sessions = list_drive_sessions(svc, os.environ["DRIVE_PARENT_FOLDER_ID"])
    if args.folder:
        sessions = [s for s in sessions
                    if args.folder in (s["name"], s["session_id"])]
        if not sessions:
            sys.stderr.write(f"No Drive folder matching {args.folder!r}\n")
            raise SystemExit(1)
    else:
        sessions = already_synced(
            sessions, list_r2_keys(s3, bucket, INCOMING),
            list_r2_keys(s3, bucket, OUTGOING))

    sessions = [s for s in sessions if s["files"]]
    if not sessions:
        print("Nothing new to sync.")
        return

    for s in sessions:
        print(f"\nSyncing {s['name']}  ->  incoming/{s['session_id']}/")
        prefix = f"{INCOMING}{s['session_id']}/"
        for f in s["files"]:
            size = int(f.get("size", 0) or 0)
            print(f"  {f['name']}  ({size / 1048576:.0f} MB)")
            stream_to_r2(svc, s3, bucket, f, prefix + f["name"])

        # the folder name becomes the analyzer's session context
        if s["context"]:
            s3.put_object(Bucket=bucket, Key=prefix + "context.txt",
                          Body=s["context"].encode(), ContentType="text/plain")
            print(f"  context.txt  <- \"{s['context']}\"")

    print(f"\nSynced {len(sessions)} session(s).")


def main():
    ap = argparse.ArgumentParser(description="Mirror Google Drive sessions into R2.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show Drive folders and which are new")
    sy = sub.add_parser("sync", help="mirror new sessions into R2")
    sy.add_argument("--folder", default=None, help="sync only this folder")
    args = ap.parse_args()
    {"list": cmd_list, "sync": cmd_sync}[args.cmd](args)


if __name__ == "__main__":
    main()