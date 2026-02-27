import React from "react";
import { AbsoluteFill } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";

import { M8S01_Title } from "./scenes/m8/M8S01_Title";
import { M8S02_WhyMatmul } from "./scenes/m8/M8S02_WhyMatmul";
import { M8S03_NaiveImpl } from "./scenes/m8/M8S03_NaiveImpl";
import { M8S04_NaiveAnalysis } from "./scenes/m8/M8S04_NaiveAnalysis";
import { M8S05_TilingConcept } from "./scenes/m8/M8S05_TilingConcept";
import { M8S06_TiledCode } from "./scenes/m8/M8S06_TiledCode";
import { M8S07_RegisterTiling } from "./scenes/m8/M8S07_RegisterTiling";
import { M8S08_VectorizedLoads } from "./scenes/m8/M8S08_VectorizedLoads";
import { M8S09_OptimizationJourney } from "./scenes/m8/M8S09_OptimizationJourney";
import { M8S10_ArithmeticIntensity } from "./scenes/m8/M8S10_ArithmeticIntensity";
import { M8S11_BankConflicts } from "./scenes/m8/M8S11_BankConflicts";
import { M8S12_TensorCores } from "./scenes/m8/M8S12_TensorCores";
import { M8S13_WhenCustom } from "./scenes/m8/M8S13_WhenCustom";
import { M8S14_CUTLASS } from "./scenes/m8/M8S14_CUTLASS";
import { M8S15_MLApplications } from "./scenes/m8/M8S15_MLApplications";
import { M8S18_Summary } from "./scenes/m8/M8S18_Summary";

const FPS = 30;

const slides: Array<{
  component: React.FC;
  durationSeconds: number;
}> = [
  { component: M8S01_Title, durationSeconds: 20 },
  { component: M8S02_WhyMatmul, durationSeconds: 36 },
  { component: M8S03_NaiveImpl, durationSeconds: 40 },
  { component: M8S04_NaiveAnalysis, durationSeconds: 38 },
  { component: M8S05_TilingConcept, durationSeconds: 42 },
  { component: M8S06_TiledCode, durationSeconds: 40 },
  { component: M8S07_RegisterTiling, durationSeconds: 42 },
  { component: M8S08_VectorizedLoads, durationSeconds: 40 },
  { component: M8S09_OptimizationJourney, durationSeconds: 40 },
  { component: M8S10_ArithmeticIntensity, durationSeconds: 38 },
  { component: M8S11_BankConflicts, durationSeconds: 38 },
  { component: M8S12_TensorCores, durationSeconds: 40 },
  { component: M8S13_WhenCustom, durationSeconds: 36 },
  { component: M8S14_CUTLASS, durationSeconds: 38 },
  { component: M8S15_MLApplications, durationSeconds: 38 },
  { component: M8S18_Summary, durationSeconds: 34 },
];

const TRANSITION_FRAMES = 15;

export const CudaModule8: React.FC = () => {
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
export const MODULE8_SLIDE_DURATIONS = [20, 36, 40, 38, 42, 40, 42, 40, 40, 38, 38, 40, 36, 38, 38, 34];
