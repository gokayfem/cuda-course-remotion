# ai-course-video

Generate narrated course videos with AI avatar overlay. Build animated slideshows with [Remotion](https://www.remotion.dev), auto-generate speech scripts with Gemini, synthesize audio with ElevenLabs TTS, and overlay talking-head avatars -- all stitched together automatically.

```
Remotion slides -> Gemini speech script -> ElevenLabs TTS -> Avatar video -> Final composited video
```

## Example: CUDA Mastery

This repo ships with a complete 10-module **CUDA Mastery** course as a working example. Each module produces a ~10-minute narrated video with animated slides and an AI avatar presenter.

## How It Works

```
slideshow/                         pipeline/
  src/courses/cuda-mastery/          generate_speech.py   (1) Gemini watches video -> speech JSON
    CudaModule1.tsx                  generate_audio.py    (2) TTS + re-render + merge + avatar overlay
    scenes/S01_TitleSlide.tsx        generate_avatars.py  (3) AI talking-head videos
  src/components/
    SlideLayout.tsx
    CodeBlock.tsx
    AnimatedText.tsx
  src/styles/theme.ts
```

**Pipeline steps:**

1. **Render slides** -- Remotion compiles React components into an MP4
2. **Generate speech** -- Gemini 2.5 Pro watches the video and writes a narration script (JSON with per-slide text + timing)
3. **Generate audio** -- ElevenLabs TTS converts each slide's speech to MP3, measures actual durations, re-renders the slideshow with corrected timing, then merges audio onto the video
4. **Generate avatars** -- fal.ai LTX creates talking-head videos synced to each audio clip
5. **Final composite** -- Avatar videos are cropped to circles and overlaid on the narrated video

## Prerequisites

- **Node.js** >= 18
- **Python** >= 3.10
- **ffmpeg** and **ffprobe** in PATH
- **fal.ai API key** ([get one here](https://fal.ai)) -- provides access to Gemini, ElevenLabs TTS, and LTX video generation

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/ai-course-video.git
cd ai-course-video

# Slideshow dependencies
cd slideshow && npm install && cd ..

# Pipeline dependencies
cd pipeline && pip install -r requirements.txt && cd ..
```

### 2. Set up API key

```bash
cp .env.example .env
# Edit .env and add your fal.ai key
```

### 3. Preview slides

```bash
cd slideshow
npm start
# Opens Remotion Studio at http://localhost:3000
```

### 4. Render a module video

```bash
cd slideshow
npx remotion render CudaModule1 out/module1.mp4
```

### 5. Generate speech script

```bash
cd pipeline
python generate_speech.py --modules 1

# Dry run (prints prompt without API calls):
python generate_speech.py --dry-run --modules 1
```

### 6. Generate narrated video

```bash
cd pipeline
python generate_audio.py --module 1
```

This runs the full pipeline: TTS generation, clip splitting, duration measurement, Remotion re-render, ffmpeg audio merge, and avatar overlay.

### 7. Generate avatars (optional, run before step 6)

```bash
cd pipeline
python generate_avatars.py --module 1
```

Avatar videos are picked up automatically by `generate_audio.py` in its final phase.

## Pipeline Scripts

### `generate_speech.py`

Uploads rendered module video to Gemini 2.5 Pro and generates a per-slide narration script.

| Flag | Description |
|------|-------------|
| `--modules N [N ...]` | Module numbers to process (default: all 1-10) |
| `--dry-run` | Print prompts without making API calls |

**Output:** `pipeline/out/module{N}_speech.json`

### `generate_audio.py`

Converts speech scripts to audio, re-renders slides with corrected timing, and composites the final video.

| Flag | Description |
|------|-------------|
| `--module N` | Module number (required) |
| `--voice ID` | ElevenLabs voice ID |
| `--skip-tts` | Skip TTS generation (use existing clips) |
| `--skip-render` | Skip Remotion re-render |
| `--skip-merge` | Skip ffmpeg audio merge |
| `--skip-avatar` | Skip avatar overlay |

**Phases:**
1. TTS generation (ElevenLabs via fal.ai)
2. Long clip splitting at silence boundaries (>40s clips)
3. Duration measurement + Remotion source update + re-render
4. ffmpeg audio merge onto video
5. Avatar overlay (circular crop, fade transitions)

### `generate_avatars.py`

Creates talking-head avatar videos from TTS audio clips using fal.ai LTX.

| Flag | Description |
|------|-------------|
| `--module N` | Module number (required) |
| `--image-url URL` | Reference image for the avatar |
| `--prompt TEXT` | Video generation prompt (default: "A man speaks to the camera") |

**Output:** `pipeline/out/avatar/module{N}/slide_NN.mp4`

## Component Library

Reusable Remotion components in `slideshow/src/components/`:

| Component | Description |
|-----------|-------------|
| `SlideLayout` | Single-column and two-column layouts with module badge, slide counter, accent bar |
| `SlideBackground` | Background variants: dark, gradient, code, accent |
| `CodeBlock` | Syntax-highlighted code with line-by-line animation, line numbers, highlights |
| `AnimatedText` | `SlideTitle`, `FadeInText`, `BulletPoint` with spring animations |
| `Diagram` | `Box`, `Arrow`, `ThreadGrid`, `ProgressBar` for technical diagrams |

Theme and fonts are configured in `slideshow/src/styles/`.

## Creating Your Own Course

See [docs/CREATING_A_COURSE.md](docs/CREATING_A_COURSE.md) for a step-by-step guide.

## Project Structure

```
ai-course-video/
  .env.example              # API key template
  pipeline/
    generate_speech.py       # Step 1: Gemini -> speech JSON
    generate_audio.py        # Step 2: TTS + render + merge + avatar
    generate_avatars.py      # Step 3: AI avatar videos
    requirements.txt
    out/                     # Generated artifacts (gitignored)
  slideshow/
    package.json
    remotion.config.ts
    src/
      Root.tsx               # Composition registry
      components/            # Reusable slide components
      styles/                # Theme and font config
      courses/
        cuda-mastery/        # Example course (10 modules)
          CudaModule1.tsx
          scenes/
            S01_TitleSlide.tsx
            ...
            m2/ ... m10/
  docs/
    CREATING_A_COURSE.md
```

## License

MIT
