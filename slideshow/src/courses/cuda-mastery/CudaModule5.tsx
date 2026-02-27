import React from "react";
import { AbsoluteFill } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";

import { M5S01_Title } from "./scenes/m5/M5S01_Title";
import { M5S02_WhyOptimize } from "./scenes/m5/M5S02_WhyOptimize";
import { M5S03_OccupancyIntro } from "./scenes/m5/M5S03_OccupancyIntro";
import { M5S04_OccupancyFactors } from "./scenes/m5/M5S04_OccupancyFactors";
import { M5S05_LaunchConfig } from "./scenes/m5/M5S05_LaunchConfig";
import { M5S06_MemoryBandwidth } from "./scenes/m5/M5S06_MemoryBandwidth";
import { M5S07_BandwidthOptimization } from "./scenes/m5/M5S07_BandwidthOptimization";
import { M5S08_ILP } from "./scenes/m5/M5S08_ILP";
import { M5S09_LoopUnrolling } from "./scenes/m5/M5S09_LoopUnrolling";
import { M5S10_RooflineIntro } from "./scenes/m5/M5S10_RooflineIntro";
import { M5S11_RooflineAnalysis } from "./scenes/m5/M5S11_RooflineAnalysis";
import { M5S12_NsightIntro } from "./scenes/m5/M5S12_NsightIntro";
import { M5S13_NsightMetrics } from "./scenes/m5/M5S13_NsightMetrics";
import { M5S14_ProfilingWorkflow } from "./scenes/m5/M5S14_ProfilingWorkflow";
import { M5S15_CaseStudy } from "./scenes/m5/M5S15_CaseStudy";
import { M5S18_Summary } from "./scenes/m5/M5S18_Summary";

const FPS = 30;

const slides: Array<{
  component: React.FC;
  durationSeconds: number;
}> = [
  { component: M5S01_Title, durationSeconds: 20 },
  { component: M5S02_WhyOptimize, durationSeconds: 36 },
  { component: M5S03_OccupancyIntro, durationSeconds: 40 },
  { component: M5S04_OccupancyFactors, durationSeconds: 38 },
  { component: M5S05_LaunchConfig, durationSeconds: 38 },
  { component: M5S06_MemoryBandwidth, durationSeconds: 40 },
  { component: M5S07_BandwidthOptimization, durationSeconds: 42 },
  { component: M5S08_ILP, durationSeconds: 40 },
  { component: M5S09_LoopUnrolling, durationSeconds: 40 },
  { component: M5S10_RooflineIntro, durationSeconds: 38 },
  { component: M5S11_RooflineAnalysis, durationSeconds: 40 },
  { component: M5S12_NsightIntro, durationSeconds: 36 },
  { component: M5S13_NsightMetrics, durationSeconds: 40 },
  { component: M5S14_ProfilingWorkflow, durationSeconds: 38 },
  { component: M5S15_CaseStudy, durationSeconds: 40 },
  { component: M5S18_Summary, durationSeconds: 34 },
];

const TRANSITION_FRAMES = 15;

export const CudaModule5: React.FC = () => {
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
export const MODULE5_SLIDE_DURATIONS = [20, 36, 40, 38, 38, 40, 42, 40, 40, 38, 40, 36, 40, 38, 40, 34];
