# Creating Your Own Course

This guide walks through creating a new course using the ai-course-video framework. The included **CUDA Mastery** course serves as a reference implementation.

## 1. Create a course folder

```
slideshow/src/courses/your-course/
  YourModule1.tsx
  scenes/
    S01_TitleSlide.tsx
    S02_Introduction.tsx
    ...
```

Each course lives in its own directory under `slideshow/src/courses/`.

## 2. Build slide scenes

Each slide is a React component that receives `frame` and `fps` from Remotion's `useCurrentFrame()` and `useVideoConfig()`.

```tsx
import React from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, BulletPoint } from "../../../components/AnimatedText";
import { THEME } from "../../../styles/theme";

export const S01_TitleSlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="accent" moduleNumber={1} moduleName="Your Course">
      <SlideTitle
        title="Your Course Title"
        subtitle="Module 1: Getting Started"
        frame={frame}
        fps={fps}
      />
    </SlideLayout>
  );
};
```

### Available components

Import from `../../../components/` (or `../../../../components/` for scenes nested in subdirectories):

- **SlideLayout** / **TwoColumnLayout** -- Page layouts with module badge and accent bar
- **SlideBackground** -- Background variants (`dark`, `gradient`, `code`, `accent`)
- **SlideTitle** -- Animated title with underline and optional subtitle
- **BulletPoint** -- Cascading bullet list items with icons
- **FadeInText** -- Simple fade + translate animation
- **CodeBlock** -- Syntax-highlighted code with line-by-line reveal
- **Box**, **Arrow**, **ThreadGrid**, **ProgressBar** -- Diagram primitives

### Theme

The theme is defined in `slideshow/src/styles/theme.ts`. You can modify colors, fonts, and spacing there, or create a course-specific theme file.

## 3. Create module composition files

Each module file imports its scenes and defines the slide sequence with durations:

```tsx
import React from "react";
import { AbsoluteFill } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";

import { S01_TitleSlide } from "./scenes/S01_TitleSlide";
import { S02_Introduction } from "./scenes/S02_Introduction";
// ... more scene imports

const FPS = 25;

const slides: Array<{
  component: React.FC;
  durationSeconds: number;
}> = [
  { component: S01_TitleSlide, durationSeconds: 20 },
  { component: S02_Introduction, durationSeconds: 35 },
  // ... more slides
];

const TRANSITION_FRAMES = 13;

export const MODULE1_SLIDE_DURATIONS = slides.map((s) => s.durationSeconds);

export const YourModule1: React.FC = () => {
  return (
    <AbsoluteFill>
      <TransitionSeries>
        {slides.map((s, i) => {
          const Comp = s.component;
          const transition =
            i % 3 === 0
              ? fade()
              : slide({ direction: "from-right" });

          return (
            <React.Fragment key={i}>
              {i > 0 && (
                <TransitionSeries.Transition
                  presentation={transition}
                  timing={linearTiming({ durationInFrames: TRANSITION_FRAMES })}
                />
              )}
              <TransitionSeries.Sequence
                durationInFrames={s.durationSeconds * FPS}
              >
                <Comp />
              </TransitionSeries.Sequence>
            </React.Fragment>
          );
        })}
      </TransitionSeries>
    </AbsoluteFill>
  );
};
```

## 4. Register compositions in Root.tsx

Add your course to `slideshow/src/Root.tsx`:

```tsx
import { YourModule1, MODULE1_SLIDE_DURATIONS } from "./courses/your-course/YourModule1";

// Inside RemotionRoot component, add:
<Folder name="Your-Course">
  <Composition
    id="YourModule1"
    component={YourModule1}
    durationInFrames={calcDuration(MODULE1_SLIDE_DURATIONS)}
    fps={FPS}
    width={1920}
    height={1080}
    defaultProps={{}}
  />
</Folder>
```

## 5. Add module data to generate_speech.py

Add an entry to the `MODULE_DATA` dictionary in `pipeline/generate_speech.py`:

```python
MODULE_DATA = {
    1: {
        "title": "Getting Started",
        "context": "This module introduces the fundamentals...",
        "durations": [20, 35, 40, 38, ...],  # Initial slide durations in seconds
        "slides": [
            "Title", "Introduction", "Overview", ...  # Slide names for the prompt
        ],
    },
    # ... more modules
}
```

The `durations` list should match the `durationSeconds` values in your module file. The `slides` list provides names that Gemini uses to identify each slide in the narration.

## 6. Run the pipeline

```bash
# Preview slides
cd slideshow && npm start

# Render video
npx remotion render YourModule1 out/module1.mp4

# Generate speech script
cd ../pipeline
python generate_speech.py --modules 1

# Generate avatars (optional)
python generate_avatars.py --module 1 --image-url "YOUR_IMAGE_URL" --prompt "A woman explains to the camera"

# Generate narrated video (TTS + re-render + merge + avatar overlay)
python generate_audio.py --module 1
```

## Tips

- **Start with short slides** (20-40 seconds each). TTS pacing is approximately 3 words/second.
- **Use `--dry-run`** with `generate_speech.py` to preview the prompt before making API calls.
- **Use `--skip-*` flags** with `generate_audio.py` to re-run individual phases.
- **The pipeline is resume-friendly** -- existing TTS clips and avatar videos are skipped automatically.
- **Slide durations get overwritten** during the pipeline. The initial values in your module file are starting estimates; `generate_audio.py` replaces them with actual TTS-driven durations.
- **Exercise/Quiz slides are excluded** from the rendered video by default (see `EXCLUDED_SLIDES` in `generate_audio.py`). Adjust this set for your course.
