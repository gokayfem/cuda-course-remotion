import React from "react";
import { AbsoluteFill } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";

import { M3S01_Title } from "./scenes/m3/M3S01_Title";
import { M3S02_WhatIsAWarp } from "./scenes/m3/M3S02_WhatIsAWarp";
import { M3S03_WarpScheduling } from "./scenes/m3/M3S03_WarpScheduling";
import { M3S04_WarpDivergence } from "./scenes/m3/M3S04_WarpDivergence";
import { M3S05_DivergencePatterns } from "./scenes/m3/M3S05_DivergencePatterns";
import { M3S06_Syncthreads } from "./scenes/m3/M3S06_Syncthreads";
import { M3S07_MemoryFences } from "./scenes/m3/M3S07_MemoryFences";
import { M3S08_AtomicOps } from "./scenes/m3/M3S08_AtomicOps";
import { M3S09_AtomicHistogram } from "./scenes/m3/M3S09_AtomicHistogram";
import { M3S10_WarpShuffle } from "./scenes/m3/M3S10_WarpShuffle";
import { M3S11_WarpReduce } from "./scenes/m3/M3S11_WarpReduce";
import { M3S12_WarpVote } from "./scenes/m3/M3S12_WarpVote";
import { M3S13_CooperativeGroups } from "./scenes/m3/M3S13_CooperativeGroups";
import { M3S14_PracticalPatterns } from "./scenes/m3/M3S14_PracticalPatterns";
import { M3S18_Summary } from "./scenes/m3/M3S18_Summary";

const FPS = 30;

const slides: Array<{
  component: React.FC;
  durationSeconds: number;
}> = [
  { component: M3S01_Title, durationSeconds: 20 },
  { component: M3S02_WhatIsAWarp, durationSeconds: 36 },
  { component: M3S03_WarpScheduling, durationSeconds: 38 },
  { component: M3S04_WarpDivergence, durationSeconds: 40 },       // Key concept
  { component: M3S05_DivergencePatterns, durationSeconds: 42 },   // Dense patterns
  { component: M3S06_Syncthreads, durationSeconds: 38 },
  { component: M3S07_MemoryFences, durationSeconds: 35 },
  { component: M3S08_AtomicOps, durationSeconds: 40 },
  { component: M3S09_AtomicHistogram, durationSeconds: 38 },
  { component: M3S10_WarpShuffle, durationSeconds: 40 },          // Complex concept
  { component: M3S11_WarpReduce, durationSeconds: 42 },           // Animation-heavy
  { component: M3S12_WarpVote, durationSeconds: 36 },
  { component: M3S13_CooperativeGroups, durationSeconds: 34 },
  { component: M3S14_PracticalPatterns, durationSeconds: 38 },
  { component: M3S18_Summary, durationSeconds: 33 },
];

const TRANSITION_FRAMES = 15;

export const CudaModule3: React.FC = () => {
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
export const MODULE3_SLIDE_DURATIONS = [20, 36, 38, 40, 42, 38, 35, 40, 38, 40, 42, 36, 34, 38, 33];
