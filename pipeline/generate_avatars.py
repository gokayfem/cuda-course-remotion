"""Generate avatar videos for each TTS audio clip using fal.ai LTX audio-to-video."""

import argparse
import os
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
AVATAR_MODEL = "fal-ai/ltx-2-19b/audio-to-video"
DEFAULT_IMAGE_URL = "https://v3b.fal.media/files/b/0a902bde/rgFrveYIdKvUfDxy7tbMe_5DB1gQbM.png"
DEFAULT_PROMPT = "A man speaks to the camera"
FPS = 25
WAIT_BETWEEN_REQUESTS = 2
MAX_RETRIES = 3
RETRY_DELAYS = [10, 30, 60]

NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, "
    "unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, "
    "extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, "
    "camera shake, incorrect depth of field, background too sharp, background clutter, "
    "distracting reflections, harsh shadows, inconsistent lighting direction, color banding, "
    "cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, "
    "incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, "
    "background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, "
    "jittery movement, awkward pauses, incorrect timing, unnatural transitions, "
    "inconsistent framing, tilted camera, flat lighting, inconsistent tone, "
    "cinematic oversaturation, stylized filters, or AI artifacts."
)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_clip_duration(path):
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def get_audio_clips(audio_dir):
    """Get sorted list of audio clips in directory."""
    return sorted(
        f for f in audio_dir.iterdir()
        if f.suffix == ".mp3" and f.name.startswith("slide_")
    )


def on_queue_update(update):
    """Log fal queue status updates."""
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(f"    [fal] {log['message']}")


# ---------------------------------------------------------------------------
# Avatar generation
# ---------------------------------------------------------------------------


def generate_avatar_video(audio_path, output_path, image_url, prompt):
    """Generate a single avatar video from an audio clip."""
    print(f"    Uploading {audio_path.name}...")
    audio_url = fal_client.upload_file(str(audio_path))

    duration = get_clip_duration(audio_path)
    num_frames = 481

    print(f"    Generating video ({duration:.1f}s, {num_frames} frames @ {FPS}fps)...")

    result = fal_client.subscribe(
        AVATAR_MODEL,
        arguments={
            "prompt": prompt,
            "audio_url": audio_url,
            "image_url": image_url,
            "match_audio_length": True,
            "num_frames": num_frames,
            "video_size": "square_hd",
            "use_multiscale": True,
            "fps": FPS,
            "guidance_scale": 3,
            "num_inference_steps": 40,
            "acceleration": "regular",
            "camera_lora": "static",
            "camera_lora_scale": 1,
            "negative_prompt": NEGATIVE_PROMPT,
            "enable_prompt_expansion": True,
            "enable_safety_checker": True,
            "video_output_type": "X264 (.mp4)",
            "video_quality": "high",
            "video_write_mode": "balanced",
            "image_strength": 1,
            "end_image_strength": 1,
            "audio_strength": 1,
            "preprocess_audio": True,
        },
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    video_url = result["video"]["url"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(video_url, output_path)
    print(f"    Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate avatar videos from TTS audio clips using fal.ai LTX"
    )
    parser.add_argument(
        "--module",
        type=int,
        required=True,
        metavar="N",
        help="Module number (matches audio directory naming)",
    )
    parser.add_argument(
        "--image-url",
        type=str,
        default=DEFAULT_IMAGE_URL,
        help=f"Reference image URL for avatar (default: built-in)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Video generation prompt (default: '{DEFAULT_PROMPT}')",
    )
    args = parser.parse_args()

    load_env()

    module = args.module
    image_url = args.image_url
    prompt = args.prompt

    audio_dir = OUTPUT_DIR / "audio" / f"module{module}"
    avatar_dir = OUTPUT_DIR / "avatar" / f"module{module}"
    avatar_dir.mkdir(parents=True, exist_ok=True)

    clips = get_audio_clips(audio_dir)
    total = len(clips)

    print(f"Module {module}: Generating avatar videos for {total} audio clips")
    print(f"  Audio dir:  {audio_dir}")
    print(f"  Avatar dir: {avatar_dir}")
    print(f"  Image:      {image_url}")
    print(f"  Prompt:     {prompt}")
    print(f"  FPS:        {FPS}")
    print()

    succeeded = 0
    failed = 0

    for i, clip in enumerate(clips):
        output_name = clip.stem + ".mp4"
        output_path = avatar_dir / output_name

        if output_path.exists():
            print(f"  [{i+1}/{total}] {clip.name} -- already exists, skipping")
            succeeded += 1
            continue

        print(f"  [{i+1}/{total}] {clip.name}")

        generated = False
        for attempt in range(MAX_RETRIES):
            try:
                generate_avatar_video(clip, output_path, image_url, prompt)
                generated = True
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    print(f"    Failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                    print(f"    Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"    ERROR after {MAX_RETRIES} attempts: {e}")

        if generated:
            succeeded += 1
        else:
            failed += 1

        if i < total - 1:
            time.sleep(WAIT_BETWEEN_REQUESTS)

    print()
    print(f"Done! {succeeded} succeeded, {failed} failed")
    print(f"Avatar videos: {avatar_dir}")


if __name__ == "__main__":
    main()
