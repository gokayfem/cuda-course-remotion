import React from "react";
import { AbsoluteFill } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";

import { M4S01_Title } from "./scenes/m4/M4S01_Title";
import { M4S02_WhyPatterns } from "./scenes/m4/M4S02_WhyPatterns";
import { M4S03_ReductionIntro } from "./scenes/m4/M4S03_ReductionIntro";
import { M4S04_ReductionV1 } from "./scenes/m4/M4S04_ReductionV1";
import { M4S05_ReductionV2 } from "./scenes/m4/M4S05_ReductionV2";
import { M4S06_ReductionOptimized } from "./scenes/m4/M4S06_ReductionOptimized";
import { M4S07_ScanIntro } from "./scenes/m4/M4S07_ScanIntro";
import { M4S08_ScanAlgorithms } from "./scenes/m4/M4S08_ScanAlgorithms";
import { M4S09_ScanCode } from "./scenes/m4/M4S09_ScanCode";
import { M4S10_HistogramIntro } from "./scenes/m4/M4S10_HistogramIntro";
import { M4S11_HistogramVersions } from "./scenes/m4/M4S11_HistogramVersions";
import { M4S12_HistogramCode } from "./scenes/m4/M4S12_HistogramCode";
import { M4S13_CompactionIntro } from "./scenes/m4/M4S13_CompactionIntro";
import { M4S14_CompactionCode } from "./scenes/m4/M4S14_CompactionCode";
import { M4S15_PracticalApplications } from "./scenes/m4/M4S15_PracticalApplications";
import { M4S18_Summary } from "./scenes/m4/M4S18_Summary";

const FPS = 30;

const slides: Array<{
  component: React.FC;
  durationSeconds: number;
}> = [
  { component: M4S01_Title, durationSeconds: 20 },
  { component: M4S02_WhyPatterns, durationSeconds: 36 },
  { component: M4S03_ReductionIntro, durationSeconds: 40 },
  { component: M4S04_ReductionV1, durationSeconds: 38 },
  { component: M4S05_ReductionV2, durationSeconds: 38 },
  { component: M4S06_ReductionOptimized, durationSeconds: 42 },       // Dense content
  { component: M4S07_ScanIntro, durationSeconds: 36 },
  { component: M4S08_ScanAlgorithms, durationSeconds: 42 },           // Complex comparison
  { component: M4S09_ScanCode, durationSeconds: 40 },
  { component: M4S10_HistogramIntro, durationSeconds: 36 },
  { component: M4S11_HistogramVersions, durationSeconds: 38 },
  { component: M4S12_HistogramCode, durationSeconds: 36 },
  { component: M4S13_CompactionIntro, durationSeconds: 40 },          // Multi-step pipeline
  { component: M4S14_CompactionCode, durationSeconds: 38 },
  { component: M4S15_PracticalApplications, durationSeconds: 36 },
  { component: M4S18_Summary, durationSeconds: 34 },
];

const TRANSITION_FRAMES = 15;

export const CudaModule4: React.FC = () => {
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
export const MODULE4_SLIDE_DURATIONS = [20, 36, 40, 38, 38, 42, 36, 42, 40, 36, 38, 36, 40, 38, 36, 34];
