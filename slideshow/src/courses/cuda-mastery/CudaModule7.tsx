import React from "react";
import { AbsoluteFill } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";

import { M7S01_Title } from "./scenes/m7/M7S01_Title";
import { M7S02_WhyLibraries } from "./scenes/m7/M7S02_WhyLibraries";
import { M7S03_CublasIntro } from "./scenes/m7/M7S03_CublasIntro";
import { M7S04_ColumnMajor } from "./scenes/m7/M7S04_ColumnMajor";
import { M7S05_CublasPerformance } from "./scenes/m7/M7S05_CublasPerformance";
import { M7S06_CudnnIntro } from "./scenes/m7/M7S06_CudnnIntro";
import { M7S07_CudnnAlgorithms } from "./scenes/m7/M7S07_CudnnAlgorithms";
import { M7S08_ThrustIntro } from "./scenes/m7/M7S08_ThrustIntro";
import { M7S09_ThrustAlgorithms } from "./scenes/m7/M7S09_ThrustAlgorithms";
import { M7S10_CurandIntro } from "./scenes/m7/M7S10_CurandIntro";
import { M7S11_CurandML } from "./scenes/m7/M7S11_CurandML";
import { M7S12_LibraryEcosystem } from "./scenes/m7/M7S12_LibraryEcosystem";
import { M7S13_LibraryInterop } from "./scenes/m7/M7S13_LibraryInterop";
import { M7S14_PracticalTips } from "./scenes/m7/M7S14_PracticalTips";
import { M7S15_CaseStudy } from "./scenes/m7/M7S15_CaseStudy";
import { M7S18_Summary } from "./scenes/m7/M7S18_Summary";

const FPS = 30;

const slides: Array<{
  component: React.FC;
  durationSeconds: number;
}> = [
  { component: M7S01_Title, durationSeconds: 20 },
  { component: M7S02_WhyLibraries, durationSeconds: 36 },
  { component: M7S03_CublasIntro, durationSeconds: 40 },
  { component: M7S04_ColumnMajor, durationSeconds: 38 },
  { component: M7S05_CublasPerformance, durationSeconds: 38 },
  { component: M7S06_CudnnIntro, durationSeconds: 40 },
  { component: M7S07_CudnnAlgorithms, durationSeconds: 42 },
  { component: M7S08_ThrustIntro, durationSeconds: 38 },
  { component: M7S09_ThrustAlgorithms, durationSeconds: 40 },
  { component: M7S10_CurandIntro, durationSeconds: 36 },
  { component: M7S11_CurandML, durationSeconds: 38 },
  { component: M7S12_LibraryEcosystem, durationSeconds: 36 },
  { component: M7S13_LibraryInterop, durationSeconds: 38 },
  { component: M7S14_PracticalTips, durationSeconds: 36 },
  { component: M7S15_CaseStudy, durationSeconds: 40 },
  { component: M7S18_Summary, durationSeconds: 34 },
];

const TRANSITION_FRAMES = 15;

export const CudaModule7: React.FC = () => {
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
export const MODULE7_SLIDE_DURATIONS = [20, 36, 40, 38, 38, 40, 42, 38, 40, 36, 38, 36, 38, 36, 40, 34];
