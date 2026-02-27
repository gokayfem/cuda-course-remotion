"""Generate TTS audio clips and overlay them on course slideshow videos."""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import fal_client
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent / "out"
SLIDESHOW_DIR = Path(__file__).resolve().parent.parent / "slideshow"
VIDEO_DIR = SLIDESHOW_DIR / "out"

VOICE_ID = "5yPNUy2ZGgvEkNjAmzo1"
TTS_MODEL = "fal-ai/elevenlabs/tts/eleven-v3"
TTS_STABILITY = 0.5

MAX_RETRIES = 3
RETRY_DELAYS = [5, 10, 20]

TRANSITION_OVERLAP_SECONDS = 0.5
DURATION_PADDING_SECONDS = 2

CHUNK_THRESHOLD_SECONDS = 40
CHUNK_MAX_SECONDS = 35


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def load_env():
    """Load FAL_KEY from .env and set it for fal_client."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    fal_key = os.getenv("FAL_KEY")
    if not fal_key or fal_key.startswith("<"):
        print(f"ERROR: Set a valid FAL_KEY in {env_path}")
        sys.exit(1)
    os.environ["FAL_KEY"] = fal_key
    return fal_key


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def timestamp_to_ms(ts):
    """Convert 'MM:SS' timestamp string to milliseconds."""
    if not re.match(r"^\d{1,2}:\d{2}$", ts):
        raise ValueError(f"Invalid timestamp format: {ts!r}. Expected MM:SS.")
    parts = ts.split(":")
    minutes = int(parts[0])
    seconds = int(parts[1])
    if seconds >= 60:
        raise ValueError(f"Invalid seconds value in timestamp: {ts!r}")
    return (minutes * 60 + seconds) * 1000


def seconds_to_timestamp(total_seconds):
    """Convert seconds (int) to 'MM:SS' timestamp string."""
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


# ---------------------------------------------------------------------------
# TTS generation
# ---------------------------------------------------------------------------


def on_queue_update(update):
    """Log fal queue status updates."""
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(f"    [fal] {log['message']}")


def generate_tts(text, voice_id=VOICE_ID):
    """Call fal ElevenLabs TTS, return audio URL."""
    result = fal_client.subscribe(
        TTS_MODEL,
        arguments={
            "text": text,
            "voice": voice_id,
            "stability": TTS_STABILITY,
            "apply_text_normalization": "auto",
        },
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    return result["audio"]["url"]


def download_audio(url, path):
    """Download audio file from URL to local path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, path)


def generate_all_clips(segments, audio_dir, voice_id=VOICE_ID):
    """Generate and download TTS clips for all segments.

    Skips clips that already exist on disk (resume-friendly).
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    total = len(segments)

    for i, seg in enumerate(segments):
        filename = f"slide_{i:02d}.mp3"
        filepath = audio_dir / filename

        if filepath.exists() or (audio_dir / f"slide_{i:02d}_p0.mp3").exists():
            print(f"  [{i+1}/{total}] {seg['slide_name']} -- already exists, skipping")
            continue

        print(f"  [{i+1}/{total}] {seg['slide_name']} ({seg['duration_seconds']}s)")

        for attempt in range(MAX_RETRIES):
            try:
                audio_url = generate_tts(seg["speech"], voice_id=voice_id)
                download_audio(audio_url, filepath)
                print(f"    Saved: {filepath.name}")
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    print(f"    Failed (attempt {attempt+1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"    ERROR: Failed after {MAX_RETRIES} attempts: {e}")
                    sys.exit(1)


# ---------------------------------------------------------------------------
# Phase 2: Measure durations & re-render video
# ---------------------------------------------------------------------------


def get_clip_duration(path):
    """Run ffprobe on an audio file, return duration in seconds (float)."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: ffprobe failed on {path}")
        print(f"  stderr: {result.stderr}")
        sys.exit(1)
    return float(result.stdout.strip())


def find_silence_near(filepath, target_seconds, search_window=10):
    """Find the center of a silence gap nearest to target_seconds.

    Uses ffmpeg silencedetect to find pauses in speech, then picks the one
    closest to the target. Falls back to target_seconds if no silence found.
    """
    cmd = [
        "ffmpeg", "-i", str(filepath),
        "-af", "silencedetect=noise=-30dB:d=0.3",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse silence_start/silence_end pairs from stderr
    silence_starts = re.findall(r"silence_start:\s*([\d.]+)", result.stderr)
    silence_ends = re.findall(r"silence_end:\s*([\d.]+)", result.stderr)

    candidates = []
    for s, e in zip(silence_starts, silence_ends):
        center = (float(s) + float(e)) / 2
        if abs(center - target_seconds) <= search_window:
            candidates.append(center)

    if not candidates:
        return target_seconds

    return min(candidates, key=lambda c: abs(c - target_seconds))


def split_long_clips(audio_dir, clip_indices):
    """Split clips >= CHUNK_THRESHOLD at silence points near CHUNK_MAX boundaries.

    A 66s clip gets split at the silence nearest to 50s -> two parts.
    A 120s clip gets split near 50s and 100s -> three parts.
    Original file is replaced with slide_NN_p0.mp3, slide_NN_p1.mp3, etc.
    Already-split clips (part files exist) are skipped.
    """
    for idx in clip_indices:
        # Skip if already split
        if (audio_dir / f"slide_{idx:02d}_p0.mp3").exists():
            continue

        filepath = audio_dir / f"slide_{idx:02d}.mp3"
        if not filepath.exists():
            print(f"ERROR: Missing audio clip: {filepath}")
            sys.exit(1)

        dur = get_clip_duration(filepath)
        if dur < CHUNK_THRESHOLD_SECONDS:
            continue

        num_parts = math.ceil(dur / CHUNK_MAX_SECONDS)
        split_points = []
        for p in range(1, num_parts):
            target = p * CHUNK_MAX_SECONDS
            actual = find_silence_near(filepath, target)
            split_points.append(actual)

        # Build cut boundaries: [0, split1, split2, ..., end]
        boundaries = [0] + split_points + [dur]
        stem = filepath.stem
        ext = filepath.suffix

        parts = []
        for i in range(len(boundaries) - 1):
            part_path = audio_dir / f"{stem}_p{i}{ext}"
            cmd = ["ffmpeg", "-y", "-i", str(filepath)]
            if boundaries[i] > 0:
                cmd.extend(["-ss", str(boundaries[i])])
            cmd.extend(["-t", str(boundaries[i + 1] - boundaries[i])])
            cmd.extend(["-c", "copy", str(part_path)])
            subprocess.run(cmd, capture_output=True, text=True)
            part_dur = get_clip_duration(part_path)
            parts.append((part_path.name, part_dur))

        filepath.unlink()

        parts_str = " + ".join(f"{name} ({d:.1f}s)" for name, d in parts)
        print(f"  slide_{idx:02d}.mp3 ({dur:.1f}s) -> {parts_str}")


def get_clip_parts(audio_dir, clip_index):
    """Return ordered list of audio file paths for a clip (may be split into parts)."""
    single = audio_dir / f"slide_{clip_index:02d}.mp3"
    if single.exists():
        return [single]

    parts = sorted(audio_dir.glob(f"slide_{clip_index:02d}_p*.mp3"))
    if parts:
        return parts

    print(f"ERROR: No audio found for slide_{clip_index:02d}")
    sys.exit(1)


def measure_all_durations(audio_dir, clip_indices):
    """Measure total TTS duration for each clip (summing parts if split).

    Args:
        audio_dir: Directory containing slide_NN.mp3 or slide_NN_pN.mp3 files.
        clip_indices: List of original segment indices to measure.
    Returns list of floats (total duration per clip).
    """
    durations = []
    for i in clip_indices:
        parts = get_clip_parts(audio_dir, i)
        total = sum(get_clip_duration(p) for p in parts)
        durations.append(total)
    return durations


def compute_slide_durations(actual_durations, padding=DURATION_PADDING_SECONDS):
    """Convert actual TTS durations to integer slide durations.

    Each slide duration = ceil(actual_tts_duration) + padding seconds.
    Returns list of ints (Remotion requires integer durations).
    """
    return [int(math.ceil(d)) + int(padding) for d in actual_durations]


def compute_start_times(slide_durations, transition_overlap=TRANSITION_OVERLAP_SECONDS):
    """Compute MM:SS start timestamps from slide durations.

    Each slide starts when the previous one ends, minus transition overlap.
    Returns list of MM:SS strings.
    """
    start_times = []
    cumulative = 0.0
    for i, dur in enumerate(slide_durations):
        start_seconds = int(round(cumulative))
        start_times.append(seconds_to_timestamp(start_seconds))
        cumulative += dur - transition_overlap
    return start_times


def update_remotion_source(module_num, new_durations):
    """Update duration arrays in CudaModule{N}.tsx and Root.tsx.

    Performs regex replacement on the well-structured duration arrays.
    """
    src_dir = SLIDESHOW_DIR / "src"

    # --- Update CudaModule{N}.tsx ---
    module_tsx = src_dir / "courses" / "cuda-mastery" / f"CudaModule{module_num}.tsx"
    if not module_tsx.exists():
        print(f"ERROR: Module source not found: {module_tsx}")
        sys.exit(1)

    content = module_tsx.read_text()

    # Replace each durationSeconds value in the slides array.
    # Matches both module 1 (S01_TitleSlide) and modules 2-10 (M2S01_Title)
    duration_pattern = re.compile(
        r"(component:\s+\w+,\s+durationSeconds:\s+)\d+"
    )

    matches = list(duration_pattern.finditer(content))
    if len(matches) != len(new_durations):
        print(
            f"ERROR: Found {len(matches)} duration entries in {module_tsx.name}, "
            f"expected {len(new_durations)}"
        )
        sys.exit(1)

    # Replace in reverse order to preserve character positions
    for match, new_dur in reversed(list(zip(matches, new_durations))):
        start = match.start(0)
        end = match.end(0)
        replacement = f"{match.group(1)}{new_dur}"
        content = content[:start] + replacement + content[end:]

    module_tsx.write_text(content)
    print(f"  Updated {module_tsx.name} with {len(new_durations)} new durations")

    # --- Update Root.tsx (module 1) or MODULE{N}_SLIDE_DURATIONS export (modules 2-10) ---
    durations_str = ", ".join(str(d) for d in new_durations)

    if module_num == 1:
        # Module 1 has an inline array in Root.tsx
        root_tsx = src_dir / "Root.tsx"
        if not root_tsx.exists():
            print(f"ERROR: Root.tsx not found: {root_tsx}")
            sys.exit(1)

        root_content = root_tsx.read_text()
        root_pattern = re.compile(
            r"(const\s+module1Durations\s*=\s*\[)[^\]]+(\])"
        )

        if not root_pattern.search(root_content):
            print(f"ERROR: Could not find module1Durations array in Root.tsx")
            sys.exit(1)

        root_content = root_pattern.sub(rf"\g<1>{durations_str}\2", root_content)
        root_tsx.write_text(root_content)
        print(f"  Updated Root.tsx module1Durations array")
    else:
        # Modules 2-10 export MODULE{N}_SLIDE_DURATIONS from their own file
        module_content = module_tsx.read_text()
        export_pattern = re.compile(
            rf"(export\s+const\s+MODULE{module_num}_SLIDE_DURATIONS\s*=\s*\[)[^\]]+(\])"
        )

        if not export_pattern.search(module_content):
            print(
                f"ERROR: Could not find MODULE{module_num}_SLIDE_DURATIONS "
                f"in {module_tsx.name}"
            )
            sys.exit(1)

        module_content = export_pattern.sub(
            rf"\g<1>{durations_str}\2", module_content
        )
        module_tsx.write_text(module_content)
        print(
            f"  Updated MODULE{module_num}_SLIDE_DURATIONS in {module_tsx.name}"
        )


def render_video(module_num):
    """Run remotion render for the specified module."""
    composition_id = f"CudaModule{module_num}"
    output_path = VIDEO_DIR / f"module{module_num}.mp4"

    print(f"  Rendering {composition_id} -> {output_path}")
    print(f"  (this may take several minutes...)")

    cmd = [
        "npx", "remotion", "render",
        composition_id,
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(SLIDESHOW_DIR),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"\n  Remotion render FAILED (exit code {result.returncode})")
        print(f"  stdout:\n{result.stdout[-2000:]}")
        print(f"  stderr:\n{result.stderr[-2000:]}")
        sys.exit(1)

    print(f"  Render complete: {output_path}")


# ---------------------------------------------------------------------------
# FFmpeg merge
# ---------------------------------------------------------------------------


def merge_audio_on_video(start_times, audio_dir, video_path, output_path, clip_indices):
    """Build and run ffmpeg command to overlay audio clips on video.

    Handles multi-part clips: each part is placed sequentially within its slide.
    Uses adelay to offset each part, amix to combine, copies video stream.

    Args:
        start_times: List of 'MM:SS' strings — one per slide.
        audio_dir: Directory containing slide_NN.mp3 or slide_NN_pN.mp3 files.
        video_path: Path to the source video.
        output_path: Path for the narrated output video.
        clip_indices: Original segment indices for clip file lookup.
    """
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Collect all audio parts and their absolute start times (ms)
    all_parts = []
    for slide_idx, ts in enumerate(start_times):
        slide_start_ms = timestamp_to_ms(ts)
        parts = get_clip_parts(audio_dir, clip_indices[slide_idx])
        offset_ms = 0
        for part_path in parts:
            all_parts.append((part_path, slide_start_ms + offset_ms))
            part_dur = get_clip_duration(part_path)
            offset_ms += int(part_dur * 1000)

    num_inputs = len(all_parts)

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    for part_path, _ in all_parts:
        cmd.extend(["-i", str(part_path)])

    # Build filter_complex
    filter_parts = []
    mix_inputs = []

    for i, (_, delay_ms) in enumerate(all_parts):
        input_idx = i + 1
        label = f"a{i}"
        filter_parts.append(
            f"[{input_idx}:a]"
            f"aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=mono,"
            f"adelay={delay_ms}|{delay_ms}"
            f"[{label}]"
        )
        mix_inputs.append(f"[{label}]")

    mix_line = (
        "".join(mix_inputs)
        + f"amix=inputs={num_inputs}:duration=longest:normalize=0,"
        + "alimiter=level_in=1:level_out=0.95:limit=1.0"
        + "[audio]"
    )
    filter_parts.append(mix_line)

    filter_complex = ";".join(filter_parts)

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[audio]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "256k",
        "-ar", "44100",
        str(output_path),
    ])

    print(f"\n  Running ffmpeg ({num_inputs} audio parts across {len(start_times)} slides)...")
    print(f"  Output: {output_path}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  FFmpeg FAILED (exit code {result.returncode})")
        print(f"  stderr:\n{result.stderr}")
        sys.exit(1)

    print(f"  Done! Narrated video: {output_path}")


# ---------------------------------------------------------------------------
# Phase 4: Avatar overlay
# ---------------------------------------------------------------------------

AVATAR_SIZE = 250
AVATAR_RADIUS = 120
AVATAR_CENTER = AVATAR_SIZE // 2
AVATAR_FADE_SECONDS = 0.5
AVATAR_FPS = 25


def overlay_avatars_on_video(
    narrated_path, avatar_dir, audio_dir, clip_indices, start_times, output_path
):
    """Overlay circular avatar videos on the narrated video with fade transitions.

    Builds a single ffmpeg command that:
    1. Processes each avatar into a circle with alpha mask
    2. Adds fade-in/fade-out on the alpha channel
    3. Concatenates all avatars with transparent gaps
    4. Overlays the combined stream on the base video
    """
    if not narrated_path.exists():
        print(f"ERROR: Narrated video not found: {narrated_path}")
        sys.exit(1)

    # Collect all avatar clips with their absolute start times (seconds)
    avatar_clips = []
    for slide_idx, ts in enumerate(start_times):
        slide_start_sec = timestamp_to_ms(ts) / 1000.0
        parts = get_clip_parts(audio_dir, clip_indices[slide_idx])
        offset_sec = 0.0
        for part_path in parts:
            avatar_name = part_path.stem + ".mp4"
            avatar_path = avatar_dir / avatar_name
            audio_dur = get_clip_duration(part_path)
            if avatar_path.exists():
                avatar_clips.append((avatar_path, slide_start_sec + offset_sec))
            else:
                print(f"  WARNING: Missing avatar {avatar_name}, skipping")
            offset_sec += audio_dur

    if not avatar_clips:
        print("  No avatar clips found!")
        return

    total = len(avatar_clips)
    print(f"  Overlaying {total} avatar clips with circular crop + fade transitions...")

    # Get actual avatar video durations
    avatar_durations = [get_clip_duration(p) for p, _ in avatar_clips]

    # Build ffmpeg inputs: base video + all avatars
    cmd = ["ffmpeg", "-y", "-i", str(narrated_path)]
    for avatar_path, _ in avatar_clips:
        cmd.extend(["-i", str(avatar_path)])

    # Build filter_complex
    filter_parts = []
    concat_segments = []

    for i, ((avatar_path, abs_start), dur) in enumerate(
        zip(avatar_clips, avatar_durations)
    ):
        input_idx = i + 1
        fade_out_start = max(0, dur - AVATAR_FADE_SECONDS)

        # Scale to circle size, add alpha mask, fade in/out
        filter_parts.append(
            f"[{input_idx}:v]scale={AVATAR_SIZE}:{AVATAR_SIZE},"
            f"format=yuva420p,"
            f"geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':"
            f"a='if(lt(pow(X-{AVATAR_CENTER},2)+pow(Y-{AVATAR_CENTER},2),"
            f"pow({AVATAR_RADIUS},2)),255,0)',"
            f"fade=t=in:st=0:d={AVATAR_FADE_SECONDS}:alpha=1,"
            f"fade=t=out:st={fade_out_start:.3f}:d={AVATAR_FADE_SECONDS}:alpha=1"
            f"[av{i}]"
        )
        concat_segments.append(f"[av{i}]")

        # Add transparent gap before next clip
        if i < total - 1:
            next_start = avatar_clips[i + 1][1]
            gap = next_start - (abs_start + dur)
            if gap > 0.04:
                filter_parts.append(
                    f"color=c=black@0.0:s={AVATAR_SIZE}x{AVATAR_SIZE}"
                    f":d={gap:.3f}:r={AVATAR_FPS},"
                    f"format=yuva420p[gap{i}]"
                )
                concat_segments.append(f"[gap{i}]")

    # Concat all avatar + gap segments into one stream
    concat_str = "".join(concat_segments)
    n_segments = len(concat_segments)
    filter_parts.append(f"{concat_str}concat=n={n_segments}:v=1:a=0[avatars]")

    # Overlay on base video
    margin = AVATAR_SIZE + 30
    filter_parts.append(
        f"[0:v][avatars]overlay=x=W-{margin}:y=H-{margin}:eof_action=pass[outv]"
    )

    filter_complex = ";\n".join(filter_parts)

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "0:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        str(output_path),
    ])

    print(f"  Running ffmpeg ({total} avatar clips, {n_segments} concat segments)...")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  FFmpeg avatar overlay FAILED (exit code {result.returncode})")
        print(f"  stderr:\n{result.stderr[-3000:]}")
        sys.exit(1)

    print(f"  Done! Final video: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio and overlay on CUDA Mastery videos"
    )
    parser.add_argument(
        "--module",
        type=int,
        required=True,
        choices=range(1, 11),
        metavar="N",
        help="Module number (1-10)",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip TTS generation (use existing clips)",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Skip video re-render (use existing video)",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip ffmpeg merge (only generate TTS clips + render)",
    )
    parser.add_argument(
        "--skip-avatar",
        action="store_true",
        help="Skip avatar overlay (only generate narrated video)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=VOICE_ID,
        help=f"ElevenLabs voice ID (default: {VOICE_ID})",
    )
    args = parser.parse_args()

    module_num = args.module

    # Load speech JSON
    speech_path = OUTPUT_DIR / f"module{module_num}_speech.json"
    if not speech_path.exists():
        print(f"ERROR: Speech script not found: {speech_path}")
        print(f"  Run generate_speech.py --modules {module_num} first.")
        sys.exit(1)

    with open(speech_path) as f:
        speech_data = json.load(f)

    all_segments = speech_data["segments"]
    audio_dir = OUTPUT_DIR / "audio" / f"module{module_num}"
    video_path = VIDEO_DIR / f"module{module_num}.mp4"
    output_path = OUTPUT_DIR / f"module{module_num}_narrated.mp4"

    # Filter out exercise and quiz segments — shown separately, not in video
    EXCLUDED_SLIDES = {"Exercise", "Quiz"}
    filtered = [
        (orig_idx, seg)
        for orig_idx, seg in enumerate(all_segments)
        if not any(tag in seg["slide_name"] for tag in EXCLUDED_SLIDES)
    ]
    clip_indices = [orig_idx for orig_idx, _ in filtered]
    segments = [seg for _, seg in filtered]

    skipped = len(all_segments) - len(segments)
    print(f"Module {module_num}: {speech_data['title']}")
    print(f"  Total segments: {len(all_segments)} ({skipped} exercise(s) excluded)")
    print(f"  Active segments: {len(segments)}")
    print(f"  Audio dir: {audio_dir}")
    print(f"  Video: {video_path}")
    print(f"  Output: {output_path}")

    # --- Phase 1: Generate TTS clips ---
    # TTS generates clips for ALL segments (including exercises) since clip files
    # are indexed by original position. We only skip exercises in phases 2-3.
    if not args.skip_tts:
        load_env()
        print(f"\n--- Phase 1: Generate TTS clips ---")
        generate_all_clips(all_segments, audio_dir, voice_id=args.voice)
        print(f"  All {len(all_segments)} clips ready.")

    # --- Phase 1.5: Split long clips at silence boundaries ---
    print(f"\n--- Splitting clips > {CHUNK_THRESHOLD_SECONDS}s at silence boundaries ---")
    split_long_clips(audio_dir, clip_indices)

    # --- Phase 2: Measure durations & re-render video ---
    if not args.skip_render:
        print(f"\n--- Phase 2: Measure TTS durations & re-render video ---")

        actual_durations = measure_all_durations(audio_dir, clip_indices)
        print(f"\n  Actual TTS durations (exercises excluded):")
        for i, (seg, actual) in enumerate(zip(segments, actual_durations)):
            original = seg["duration_seconds"]
            diff = actual - original
            print(f"    [{i:02d}] {seg['slide_name']:20s}  allocated={original:5.1f}s  actual={actual:5.1f}s  diff={diff:+.1f}s")

        slide_durations = compute_slide_durations(actual_durations)
        total_original = sum(s["duration_seconds"] for s in segments)
        total_new = sum(slide_durations)
        print(f"\n  Total duration: {total_original}s (original) -> {total_new}s (TTS-driven)")

        start_times = compute_start_times(slide_durations)
        print(f"\n  New start times:")
        for i, (seg, ts, dur) in enumerate(zip(segments, start_times, slide_durations)):
            print(f"    [{i:02d}] {ts:>6s}  {seg['slide_name']:20s}  ({dur}s)")

        update_remotion_source(module_num, slide_durations)
        render_video(module_num)
    else:
        # Even when skipping render, compute start_times from existing clips
        # so the merge phase uses correct timestamps
        print(f"\n  Measuring TTS durations for merge timestamps...")
        actual_durations = measure_all_durations(audio_dir, clip_indices)
        slide_durations = compute_slide_durations(actual_durations)
        start_times = compute_start_times(slide_durations)

    # --- Phase 3: Merge audio onto video ---
    if not args.skip_merge:
        print(f"\n--- Phase 3: Merge audio onto video ---")
        merge_audio_on_video(start_times, audio_dir, video_path, output_path, clip_indices)

    # --- Phase 4: Avatar overlay ---
    if not args.skip_avatar:
        avatar_dir = OUTPUT_DIR / "avatar" / f"module{module_num}"
        final_path = OUTPUT_DIR / f"module{module_num}_final.mp4"

        if avatar_dir.exists() and any(avatar_dir.glob("*.mp4")):
            print(f"\n--- Phase 4: Avatar overlay ---")
            overlay_avatars_on_video(
                output_path, avatar_dir, audio_dir,
                clip_indices, start_times, final_path,
            )
        else:
            print(f"\n  Skipping avatar overlay: no avatars in {avatar_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
