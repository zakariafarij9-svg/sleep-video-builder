#!/usr/bin/env python3
"""
Builds a narrated slow-zoom slideshow video with ffmpeg. The finished files are
copied into dist/, which the GitHub workflow publishes as a GitHub Release.

Input (env var PAYLOAD, JSON):
  title, description, tags,
  imageUrls: [..]   (in display order)
  audioUrls: [..]   (narration chunks, in playback order)
  subtitles: bool   (default true)  burn subtitles into the video + save subtitles.srt
  language:  "en-GB" etc.           used for speech recognition
  outroVideoPath: optional, default "assets/black_screen.mp4" - a video
    committed to the repo that gets appended after the main video with a
    fade transition. Skipped entirely if the file isn't there.
  outroFadeSeconds: optional, default 3 - crossfade duration in seconds
  outroMinutes: optional, default OUTRO_MINUTES_DEFAULT below - the outro clip is
    looped (or trimmed) to this length, so a short clip is enough. Use 0 to play
    it once as-is.

Output: dist/<title>.mp4, dist/thumbnail.jpg, dist/subtitles.srt, dist/metadata.txt
"""
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap
import time
import traceback

import requests

W, H, FPS = 1280, 720, 15      # 720p, 15 fps keeps the file small and the render fast
ZOOM = 0.15                    # total zoom over each image (1.00 -> 1.15)
WORK = pathlib.Path("work")
DIST = pathlib.Path("dist")
OUTRO_MINUTES_DEFAULT = 90     # length of the black fade-out tail; set 0 for a single short fade


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


# ---------------------------------------------------------------- outro
def normalize_outro(outro_path, target_seconds=0):
    """Match the outro's resolution/fps/pixel format and audio format to the
    main video, so the crossfade filter has two compatible inputs. If
    target_seconds > 0 the clip is looped (or trimmed) to exactly that length,
    so a short clip can fill a long outro."""
    normalized = WORK / "outro_normalized.mp4"
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p")
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if target_seconds > 0:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-i", outro_path]
    if target_seconds > 0:
        cmd += ["-t", f"{target_seconds:.3f}"]
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-ar", "44100",
            normalized]
    run(cmd)
    return normalized


def add_outro_with_crossfade(main_path, outro_path, fade_seconds, dest, target_seconds=0):
    """Appends outro_path after main_path with a fade-to-black video
    crossfade and an audio crossfade (keeps the outro's own audio track)."""
    outro_norm = normalize_outro(outro_path, target_seconds)
    main_dur = probe_duration(main_path)
    fade = min(fade_seconds, max(0.5, main_dur - 0.5))
    offset = max(0.0, main_dur - fade)
    filter_complex = (
        f"[0:v][1:v]xfade=transition=fade:duration={fade:.3f}:offset={offset:.3f}[v];"
        f"[0:a][1:a]acrossfade=d={fade:.3f}[a]"
    )
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", main_path, "-i", outro_norm,
         "-filter_complex", filter_complex, "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", dest])
    return dest


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


def ass_color(hex_color, default="FFFFFF"):
    """'#RRGGBB' -> libass '&H00BBGGRR&' (falls back to white if the value is invalid)."""
    h = str(hex_color or default).strip().lstrip("#")
    if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
        h = default
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}&".upper()


def mux_with_subtitles(video, audio, srt, dest, color="#FFFFFF"):
    """One pass: burn subtitles into the picture and add the narration."""
    style = (f"FontName=DejaVu Sans,FontSize=13,PrimaryColour={ass_color(color)},"
             "OutlineColour=&H00000000&,BorderStyle=1,Outline=2,Shadow=0,MarginV=36,Alignment=2")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", audio,
         "-vf", f"subtitles=filename={srt}:force_style='{style}'",
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p", "-r", FPS,
         "-c:a", "copy", "-movflags", "+faststart", dest])


# ---------------------------------------------------------------- thumbnail
def make_thumbnail(image_path, text, dest, dim=0.45):
    """1920x1080 thumbnail: image darkened, big white text with a black outline (kept under YouTube's 2 MB limit)."""
    from PIL import Image, ImageDraw, ImageFont

    tw, th = 1920, 1080
    k = tw / 1280                                    # scale factor vs. a 1280-wide layout
    img = Image.open(image_path).convert("RGB")
    img = img.resize((tw, round(img.height * tw / img.width)), Image.LANCZOS)
    top = max(0, (img.height - th) // 2)
    img = img.crop((0, top, tw, top + th))
    img = Image.blend(img, Image.new("RGB", (tw, th), (0, 0, 0)), dim)   # darken so text pops

    text = (text or "BORING\nHISTORY\nFOR SLEEP").strip()
    lines = [l.strip() for l in (text.split("\n") if "\n" in text else textwrap.wrap(text, 12)) if l.strip()]

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    def load(size):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            return ImageFont.load_default(size)

    draw = ImageDraw.Draw(img)
    size = int(200 * k)
    while size > 60:
        font = load(size)
        stroke = max(6, size // 16)
        widths = [draw.textlength(l, font=font) for l in lines]
        line_h = int(size * 1.15)
        if max(widths) + 2 * stroke <= tw - int(140 * k) and line_h * len(lines) <= th - int(100 * k):
            break
        size -= 9
    y = (th - line_h * len(lines)) // 2
    for line, w in zip(lines, widths):
        draw.text(((tw - w) / 2, y), line, font=font, fill=(255, 255, 255),
                  stroke_width=stroke, stroke_fill=(0, 0, 0))
        y += line_h

    # YouTube rejects thumbnails over 2 MB: lower JPEG quality until it fits
    for quality in (90, 85, 80, 75, 70, 60):
        img.save(dest, "JPEG", quality=quality, optimize=True)
        if pathlib.Path(dest).stat().st_size < 1_900_000:
            break


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


def package_outputs(final, thumb, srt, payload):
    """Copy the deliverables into dist/ (the workflow publishes that folder as a GitHub Release)."""
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(payload.get("title", ""))).strip("-")[:80] or "video"
    shutil.move(str(final), DIST / f"{slug}.mp4")
    for src, name in ((thumb, "thumbnail.jpg"), (srt, "subtitles.srt"), (WORK / "metadata.txt", "metadata.txt")):
        if src and pathlib.Path(src).exists():
            shutil.copy(src, DIST / name)
    for f in sorted(DIST.iterdir()):
        print(f"  dist/{f.name}  ({f.stat().st_size / 1e6:.1f} MB)", flush=True)


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

    thumb = None
    if payload.get("thumbnail", True):
        try:
            thumb = WORK / "thumbnail.jpg"
            bg, dim = image_paths[0], 0.45          # fallback: first video image
            if payload.get("thumbnailUrl"):          # preferred: the separately generated thumbnail image
                try:
                    bg = WORK / "thumbnail_bg.jpg"
                    download(payload["thumbnailUrl"], bg)
                    dim = 0.30
                except Exception as e:  # noqa: BLE001
                    print("WARNING: could not download thumbnail image, using first image:", e, flush=True)
                    bg, dim = image_paths[0], 0.45
            make_thumbnail(bg, payload.get("thumbnailText"), thumb, dim)
            print("Thumbnail created", flush=True)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print("WARNING: thumbnail failed, continuing without it.", flush=True)
            thumb = None

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
        mux_with_subtitles(video, narration, srt, final, payload.get("subtitleColor"))
    else:
        mux(video, narration, final)
    print(f"Final file: {final.stat().st_size / 1e6:.0f} MB", flush=True)

    outro_path = pathlib.Path(payload.get("outroVideoPath", "assets/black_screen.mp4"))
    fade_seconds = float(payload.get("outroFadeSeconds", 3))
    outro_seconds = float(payload.get("outroMinutes", OUTRO_MINUTES_DEFAULT)) * 60
    if outro_path.exists():
        try:
            print(f"Adding fade-to-black outro: {outro_path} "
                  f"(looped to {outro_seconds / 60:.0f} min, {fade_seconds}s crossfade)", flush=True)
            with_outro = WORK / "final_with_outro.mp4"
            add_outro_with_crossfade(final, outro_path, fade_seconds, with_outro, outro_seconds)
            final = with_outro
            print(f"Final file with outro: {final.stat().st_size / 1e6:.0f} MB "
                  f"({probe_duration(final) / 60:.1f} min total)", flush=True)
        except Exception:  # noqa: BLE001 (never lose the video because the outro failed)
            traceback.print_exc()
            print("WARNING: outro crossfade failed, continuing without it.", flush=True)
    else:
        print(f"No outro video found at {outro_path} — skipping fade-to-black outro.", flush=True)

    print("Packaging output files...", flush=True)
    package_outputs(final, thumb, srt, payload)


if __name__ == "__main__":
    main()
