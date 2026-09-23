#!/usr/bin/env python3
"""
Builds a narrated slow-zoom slideshow video with ffmpeg and uploads it to YouTube.

Input (env var PAYLOAD, JSON):
  title, description, tags, categoryId, privacyStatus,
  imageUrls: [..]   (in display order)
  audioUrls: [..]   (narration chunks, in playback order)
  extraTailMinutes: optional, default 60 - minutes of silent video that keep
    playing (slow zoom continues, no more narration) after the voiceover ends

Secrets (env): YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import requests

W, H, FPS = 1280, 720, 15      # 720p, 15 fps keeps the file small and the render fast
ZOOM = 0.15                    # total zoom over each image (1.00 -> 1.15)
WORK = pathlib.Path("work")


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def download(url, dest, tries=4):
    for attempt in range(1, tries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            return
        except Exception as e:  # noqa: BLE001
            print(f"download failed ({attempt}/{tries}): {e}", flush=True)
            if attempt == tries:
                raise
            time.sleep(5 * attempt)


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(out.strip())


def split_sentences(text):
    """Same sentence-splitting rule the n8n workflow uses to build TTS chunks,
    so captions line up with how the narration was actually spoken."""
    sentences = re.findall(r"[^.!?]+[.!?]+(?:\s+|$)", text or "")
    if not sentences:
        sentences = [text] if text else []
    return [s.strip() for s in sentences if s.strip()]


def format_srt_time(seconds):
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_subtitles(audio_paths, chunk_texts):
    """No transcription needed: each audio file's real duration (ffprobe) is
    split across its sentences in proportion to sentence length, and chunks
    are laid end to end in narration order to get absolute timestamps."""
    cues = []
    offset = 0.0
    for path, text in zip(audio_paths, chunk_texts):
        dur = probe_duration(path)
        sentences = split_sentences(text)
        if sentences:
            total_chars = sum(len(s) for s in sentences) or 1
            t = offset
            for s in sentences:
                share = len(s) / total_chars * dur
                cues.append((t, t + share, s))
                t += share
        offset += dur
    srt_path = WORK / "captions.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(cues, 1):
            f.write(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n")
    return srt_path


def build_narration(audio_paths):
    """Concatenate all narration chunks into one AAC track."""
    listfile = WORK / "audio_list.txt"
    listfile.write_text("".join(f"file '{p.name}'\n" for p in audio_paths))
    out = WORK / "narration.m4a"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", listfile, "-vn", "-c:a", "aac", "-b:a", "64k", "-ac", "1",
         "-ar", "44100", out])
    return out


def render_zoom_clips(image_paths, total_seconds, prefix="clip", start_parity=0):
    """One slow-zoom clip per image (alternating zoom in / zoom out) covering
    total_seconds. start_parity keeps the in/out alternation continuing
    smoothly when this is called a second time for the extra tail."""
    n = len(image_paths)
    total_frames = max(n, round(total_seconds * FPS))
    bounds = [round(i * total_frames / n) for i in range(n + 1)]
    clips = []
    for i, img in enumerate(image_paths):
        frames = max(1, bounds[i + 1] - bounds[i])
        if (start_parity + i) % 2 == 0:
            z = f"1+{ZOOM}*on/{frames}"
        else:
            z = f"{1 + ZOOM}-{ZOOM}*on/{frames}"
        vf = (
            f"scale={W * 2}:-2:flags=lanczos,crop={W * 2}:{H * 2},"
            f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={W}x{H}:fps={FPS},format=yuv420p"
        )
        clip = WORK / f"{prefix}_{i:02d}.mp4"
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", img, "-vf", vf,
             "-frames:v", frames, "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "30", "-pix_fmt", "yuv420p", "-r", FPS, clip])
        clips.append(clip)
    return clips


def concat_clips(clips, dest):
    listfile = WORK / f"{dest.stem}_list.txt"
    listfile.write_text("".join(f"file '{c.name}'\n" for c in clips))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", listfile, "-c", "copy", dest])
    return dest


def build_silence(duration_seconds, dest):
    """A silent AAC track matching narration.m4a's format, so it can be
    stream-copy concatenated onto the end of the narration."""
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         "anullsrc=channel_layout=mono:sample_rate=44100",
         "-t", f"{duration_seconds:.3f}", "-c:a", "aac", "-b:a", "64k", dest])
    return dest


def pad_audio_with_silence(narration_path, narration_seconds, extra_seconds):
    """Appends extra_seconds of silence after the narration so the audio
    track lasts as long as the video's extra tail. Returns (path, new_total)."""
    if extra_seconds <= 0:
        return narration_path, narration_seconds
    silence = build_silence(extra_seconds, WORK / "silence.m4a")
    listfile = WORK / "padded_audio_list.txt"
    listfile.write_text(
        f"file '{narration_path.name}'\nfile '{silence.name}'\n"
    )
    out = WORK / "narration_padded.m4a"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", listfile, "-c", "copy", out])
    return out, narration_seconds + extra_seconds


def mux(video, audio, dest, subtitles_path=None):
    if subtitles_path:
        # Burning captions in requires re-encoding the video stream (can no
        # longer just stream-copy), so this pass is slower than the plain path.
        vf = (
            f"subtitles={subtitles_path}:force_style="
            "'FontSize=20,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,"
            "BorderStyle=1,Outline=2,Shadow=0'"
        )
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", audio,
             "-map", "0:v:0", "-map", "1:a:0", "-vf", vf,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k",
             "-movflags", "+faststart", dest])
    else:
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", audio,
             "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
             "-movflags", "+faststart", dest])


def clean_tags(tags):
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    out, used = [], 0
    for t in tags or []:
        t = str(t).replace("<", "").replace(">", "").strip()
        if not t or used + len(t) + 1 > 450:
            continue
        out.append(t)
        used += len(t) + 1
    return out


def upload_to_youtube(path, payload):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    creds = Credentials(
        None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    creds.refresh(Request())
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)

    body = {
        "snippet": {
            "title": str(payload["title"]).replace("<", "").replace(">", "")[:100],
            "description": str(payload.get("description", "")).replace("<", "").replace(">", "")[:4900],
            "tags": clean_tags(payload.get("tags")),
            "categoryId": str(payload.get("categoryId", "27")),
        },
        "status": {
            "privacyStatus": payload.get("privacyStatus", "private"),
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(path), mimetype="video/mp4",
                            chunksize=16 * 1024 * 1024, resumable=True)
    request = yt.videos().insert(part="snippet,status", body=body, media_body=media)

    response, failures = None, 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                print(f"Uploaded {int(status.progress() * 100)}%", flush=True)
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and failures < 5:
                failures += 1
                time.sleep(2 ** failures)
                continue
            raise
    print("Uploaded video id:", response["id"], flush=True)
    return response["id"]


def main():
    payload = json.loads(os.environ["PAYLOAD"])
    image_urls, audio_urls = payload["imageUrls"], payload["audioUrls"]
    if not image_urls or not audio_urls:
        sys.exit("payload needs imageUrls and audioUrls")

    WORK.mkdir(exist_ok=True)
    tags = clean_tags(payload.get("tags"))
    (WORK / "metadata.txt").write_text(
        f"TITLE:\n{payload.get('title', '')}\n\nDESCRIPTION:\n{payload.get('description', '')}\n\n"
        f"TAGS:\n{', '.join(tags)}\n"
    )

    audio_paths = []
    for i, url in enumerate(audio_urls):
        p = WORK / f"audio_{i:03d}.mp3"
        download(url, p)
        audio_paths.append(p)
    image_paths = []
    for i, url in enumerate(image_urls):
        p = WORK / f"image_{i:02d}.jpg"
        download(url, p)
        image_paths.append(p)

    narration = build_narration(audio_paths)
    total = probe_duration(narration)
    print(f"Narration length: {total / 60:.1f} minutes", flush=True)

    subtitles_path = None
    chunk_texts = payload.get("chunkTexts")
    if payload.get("subtitles") and chunk_texts and len(chunk_texts) == len(audio_paths):
        try:
            subtitles_path = build_subtitles(audio_paths, chunk_texts)
            print(f"Captions: {subtitles_path}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"Subtitle generation failed, continuing without captions: {e}", flush=True)
            subtitles_path = None
    elif payload.get("subtitles"):
        print("subtitles requested but chunkTexts missing/mismatched: skipping captions", flush=True)

    extra_seconds = max(0.0, float(payload.get("extraTailMinutes", 60)) * 60)

    main_clips = render_zoom_clips(image_paths, total, prefix="clip")
    extra_clips = []
    if extra_seconds > 0:
        print(f"Extra silent tail: {extra_seconds / 60:.0f} minutes", flush=True)
        extra_clips = render_zoom_clips(
            image_paths, extra_seconds, prefix="clip_extra",
            start_parity=len(image_paths),
        )
    video = concat_clips(main_clips + extra_clips, WORK / "video.mp4")

    audio_for_mux, audio_total = pad_audio_with_silence(narration, total, extra_seconds)

    final = WORK / "final.mp4"
    mux(video, audio_for_mux, final, subtitles_path=subtitles_path)
    print(f"Final file: {final.stat().st_size / 1e6:.0f} MB "
          f"({audio_total / 60:.1f} min total)", flush=True)

    if os.environ.get("YT_REFRESH_TOKEN"):
        upload_to_youtube(final, payload)
    else:
        print("No YT_REFRESH_TOKEN set: skipping YouTube upload. "
              "Download the video from this run's Artifacts.", flush=True)


if __name__ == "__main__":
    main()
