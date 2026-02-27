import React from "react";
import { AbsoluteFill } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";

import { M10S01_Title } from "./scenes/m10/M10S01_Title";
import { M10S02_TensorCoreIntro } from "./scenes/m10/M10S02_TensorCoreIntro";
import { M10S03_WMMA } from "./scenes/m10/M10S03_WMMA";
import { M10S04_MixedPrecisionIntro } from "./scenes/m10/M10S04_MixedPrecisionIntro";
import { M10S05_LossScaling } from "./scenes/m10/M10S05_LossScaling";
import { M10S06_AMPPattern } from "./scenes/m10/M10S06_AMPPattern";
import { M10S07_CUTLASSIntro } from "./scenes/m10/M10S07_CUTLASSIntro";
import { M10S08_TritonIntro } from "./scenes/m10/M10S08_TritonIntro";
import { M10S09_TritonVsCuda } from "./scenes/m10/M10S09_TritonVsCuda";
import { M10S10_PyTorchExtIntro } from "./scenes/m10/M10S10_PyTorchExtIntro";
import { M10S11_PyTorchExtCode } from "./scenes/m10/M10S11_PyTorchExtCode";
import { M10S12_LandscapeOverview } from "./scenes/m10/M10S12_LandscapeOverview";
import { M10S13_CareerPath } from "./scenes/m10/M10S13_CareerPath";
import { M10S14_ResourcesGuide } from "./scenes/m10/M10S14_ResourcesGuide";
import { M10S15_CurriculumReview } from "./scenes/m10/M10S15_CurriculumReview";
import { M10S18_Summary } from "./scenes/m10/M10S18_Summary";

const FPS = 30;

const slides: Array<{
  component: React.FC;
  durationSeconds: number;
}> = [
  { component: M10S01_Title, durationSeconds: 20 },
  { component: M10S02_TensorCoreIntro, durationSeconds: 36 },
  { component: M10S03_WMMA, durationSeconds: 40 },
  { component: M10S04_MixedPrecisionIntro, durationSeconds: 40 },
  { component: M10S05_LossScaling, durationSeconds: 38 },
  { component: M10S06_AMPPattern, durationSeconds: 38 },
  { component: M10S07_CUTLASSIntro, durationSeconds: 40 },
  { component: M10S08_TritonIntro, durationSeconds: 40 },
  { component: M10S09_TritonVsCuda, durationSeconds: 38 },
  { component: M10S10_PyTorchExtIntro, durationSeconds: 38 },
  { component: M10S11_PyTorchExtCode, durationSeconds: 40 },
  { component: M10S12_LandscapeOverview, durationSeconds: 36 },
  { component: M10S13_CareerPath, durationSeconds: 38 },
  { component: M10S14_ResourcesGuide, durationSeconds: 36 },
  { component: M10S15_CurriculumReview, durationSeconds: 42 },
  { component: M10S18_Summary, durationSeconds: 36 },
];

const TRANSITION_FRAMES = 15;

export const CudaModule10: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: "#0a0a0a" }}>
      <TransitionSeries>
        {slides.map(({ component: SlideComponent, durationSeconds }, i) => {
          const elements: React.ReactNode[] = [];

          elements.push(
            <TransitionSeries.Sequence
              key={`slide-${i}`}
              durationInFrames={durationSeconds * FPS}
            >
              <SlideComponent />
            </TransitionSeries.Sequence>
          );

          if (i < slides.length - 1) {
            elements.push(
              <TransitionSeries.Transition
                key={`transition-${i}`}
                presentation={
                  i % 3 === 0
                    ? fade()
                    : i % 3 === 1
                    ? slide({ direction: "from-right" })
                    : fade()
                }
                timing={linearTiming({ durationInFrames: TRANSITION_FRAMES })}
              />
            );
          }

          return elements;
        })}
      </TransitionSeries>
    </AbsoluteFill>
  );
};

// Duration calculation for Root.tsx
export const MODULE10_SLIDE_DURATIONS = [20, 36, 40, 40, 38, 38, 40, 40, 38, 38, 40, 36, 38, 36, 42, 36];
