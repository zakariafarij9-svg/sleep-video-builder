#!/usr/bin/env python3
"""
Builds a narrated slow-zoom slideshow video with ffmpeg and uploads it to YouTube.

Input (env var PAYLOAD, JSON):
  title, description, tags, categoryId, privacyStatus,
  imageUrls: [..]   (in display order)
  audioUrls: [..]   (narration chunks, in playback order)
  subtitles: bool   (default true)  burn subtitles into the video + save subtitles.srt
  language:  "en-GB" etc.           used for speech recognition

Secrets (env): YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN
"""
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import time
import traceback

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


def build_narration(audio_paths):
    """Concatenate all narration chunks into one AAC track."""
    listfile = WORK / "audio_list.txt"
    listfile.write_text("".join(f"file '{p.name}'\n" for p in audio_paths))
    out = WORK / "narration.m4a"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", listfile, "-vn", "-c:a", "aac", "-b:a", "64k", "-ac", "1",
         "-ar", "44100", out])
    return out


def build_video(image_paths, total_seconds):
    """One slow-zoom clip per image (alternating zoom in / zoom out), then concat."""
    n = len(image_paths)
    total_frames = max(n, round(total_seconds * FPS))
    bounds = [round(i * total_frames / n) for i in range(n + 1)]
    clips = []
    for i, img in enumerate(image_paths):
        frames = max(1, bounds[i + 1] - bounds[i])
        if i % 2 == 0:
            z = f"1+{ZOOM}*on/{frames}"
        else:
            z = f"{1 + ZOOM}-{ZOOM}*on/{frames}"
        vf = (
            f"scale={W * 2}:-2:flags=lanczos,crop={W * 2}:{H * 2},"
            f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={W}x{H}:fps={FPS},format=yuv420p"
        )
        clip = WORK / f"clip_{i:02d}.mp4"
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", img, "-vf", vf,
             "-frames:v", frames, "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "30", "-pix_fmt", "yuv420p", "-r", FPS, clip])
        clips.append(clip)
    listfile = WORK / "clips.txt"
    listfile.write_text("".join(f"file '{c.name}'\n" for c in clips))
    out = WORK / "video.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", listfile, "-c", "copy", out])
    return out


def mux(video, audio, dest):
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", audio,
         "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
         "-movflags", "+faststart", dest])


# ---------------------------------------------------------------- subtitles
def words_to_cues(words, max_chars=80, max_secs=6.5):
    """words: [(start, end, text)] -> [[start, end, text]] readable subtitle cues."""
    cues, cur, cur_start, cur_end = [], [], None, None

    def flush():
        nonlocal cur, cur_start, cur_end
        if cur:
            cues.append([cur_start, cur_end, " ".join(cur)])
        cur, cur_start, cur_end = [], None, None

    for start, end, text in words:
        text = text.strip()
        if not text:
            continue
        if cur and (len(" ".join(cur)) + 1 + len(text) > max_chars or end - cur_start > max_secs):
            flush()
        if not cur:
            cur_start = start
        cur.append(text)
        cur_end = end
        joined = " ".join(cur)
        if text[-1] in ".!?" and len(joined) >= 30:
            flush()
        elif text[-1] in ",;:" and len(joined) >= 60:
            flush()
    flush()

    for i, c in enumerate(cues):          # small hold after speech, never overlap the next cue
        end = c[1] + 0.25
        if i + 1 < len(cues):
            end = min(end, cues[i + 1][0] - 0.02)
        c[1] = max(end, c[0] + 0.2)
    return cues


def fmt_ts(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(cues, path):
    blocks = []
    for i, (start, end, text) in enumerate(cues, 1):
        blocks.append(f"{i}\n{fmt_ts(start)} --> {fmt_ts(end)}\n" + "\n".join(textwrap.wrap(text, 42)) + "\n")
    pathlib.Path(path).write_text("\n".join(blocks), encoding="utf-8")


def transcribe_to_srt(audio_path, language, srt_path):
    """Speech-to-text with faster-whisper (free, runs on the GitHub runner)."""
    from faster_whisper import WhisperModel

    lang = (language or "en").split("-")[0].lower()
    model = WhisperModel("base.en" if lang == "en" else "base", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(audio_path), language=lang, beam_size=1, word_timestamps=True,
        vad_filter=False, condition_on_previous_text=False,
    )
    words, last_report = [], 0
    for seg in segments:
        for w in seg.words or []:
            words.append((w.start, w.end, w.word))
        if seg.end - last_report >= 300:
            last_report = seg.end
            print(f"  transcribed {seg.end / 60:.0f} min", flush=True)
    cues = words_to_cues(words)
    if not cues:
        raise RuntimeError("speech recognition returned no words")
    write_srt(cues, srt_path)
    print(f"Wrote {len(cues)} subtitle cues", flush=True)


SUB_STYLE = ("FontName=DejaVu Sans,FontSize=13,PrimaryColour=&H00FFFFFF&,OutlineColour=&H00000000&,"
             "BorderStyle=1,Outline=2,Shadow=0,MarginV=36,Alignment=2")


def mux_with_subtitles(video, audio, srt, dest):
    """One pass: burn subtitles into the picture and add the narration."""
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", audio,
         "-vf", f"subtitles=filename={srt}:force_style='{SUB_STYLE}'",
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p", "-r", FPS,
         "-c:a", "copy", "-movflags", "+faststart", dest])


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

    srt = None
    if payload.get("subtitles", True):
        try:
            print("Creating subtitles...", flush=True)
            srt = WORK / "subtitles.srt"
            transcribe_to_srt(narration, payload.get("language", "en"), srt)
        except Exception:  # noqa: BLE001  (never lose the video because subtitles failed)
            traceback.print_exc()
            print("WARNING: subtitles failed, continuing without them.", flush=True)
            srt = None

    video = build_video(image_paths, total)
    final = WORK / "final.mp4"
    if srt:
        mux_with_subtitles(video, narration, srt, final)
    else:
        mux(video, narration, final)
    print(f"Final file: {final.stat().st_size / 1e6:.0f} MB", flush=True)

    if os.environ.get("YT_REFRESH_TOKEN"):
        upload_to_youtube(final, payload)
    else:
        print("No YT_REFRESH_TOKEN set: skipping YouTube upload. "
              "Download the video from this run's Artifacts.", flush=True)


if __name__ == "__main__":
    main()
