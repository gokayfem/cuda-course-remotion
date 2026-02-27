import React from "react";
import { AbsoluteFill } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";

import { M6S01_Title } from "./scenes/m6/M6S01_Title";
import { M6S02_WhyStreams } from "./scenes/m6/M6S02_WhyStreams";
import { M6S03_StreamBasics } from "./scenes/m6/M6S03_StreamBasics";
import { M6S04_AsyncTransfers } from "./scenes/m6/M6S04_AsyncTransfers";
import { M6S05_PinnedMemory } from "./scenes/m6/M6S05_PinnedMemory";
import { M6S06_OverlapPattern } from "./scenes/m6/M6S06_OverlapPattern";
import { M6S07_DoubleBuffering } from "./scenes/m6/M6S07_DoubleBuffering";
import { M6S08_EventsIntro } from "./scenes/m6/M6S08_EventsIntro";
import { M6S09_EventTiming } from "./scenes/m6/M6S09_EventTiming";
import { M6S10_MultiGPUIntro } from "./scenes/m6/M6S10_MultiGPUIntro";
import { M6S11_MultiGPUCode } from "./scenes/m6/M6S11_MultiGPUCode";
import { M6S12_StreamCallbacks } from "./scenes/m6/M6S12_StreamCallbacks";
import { M6S13_PracticalPatterns } from "./scenes/m6/M6S13_PracticalPatterns";
import { M6S14_CommonMistakes } from "./scenes/m6/M6S14_CommonMistakes";
import { M6S15_CaseStudy } from "./scenes/m6/M6S15_CaseStudy";
import { M6S18_Summary } from "./scenes/m6/M6S18_Summary";

const FPS = 30;

const slides: Array<{
  component: React.FC;
  durationSeconds: number;
}> = [
  { component: M6S01_Title, durationSeconds: 20 },
  { component: M6S02_WhyStreams, durationSeconds: 36 },
  { component: M6S03_StreamBasics, durationSeconds: 38 },
  { component: M6S04_AsyncTransfers, durationSeconds: 40 },
  { component: M6S05_PinnedMemory, durationSeconds: 38 },
  { component: M6S06_OverlapPattern, durationSeconds: 42 },
  { component: M6S07_DoubleBuffering, durationSeconds: 38 },
  { component: M6S08_EventsIntro, durationSeconds: 38 },
  { component: M6S09_EventTiming, durationSeconds: 38 },
  { component: M6S10_MultiGPUIntro, durationSeconds: 36 },
  { component: M6S11_MultiGPUCode, durationSeconds: 38 },
  { component: M6S12_StreamCallbacks, durationSeconds: 36 },
  { component: M6S13_PracticalPatterns, durationSeconds: 40 },
  { component: M6S14_CommonMistakes, durationSeconds: 36 },
  { component: M6S15_CaseStudy, durationSeconds: 40 },
  { component: M6S18_Summary, durationSeconds: 34 },
];

const TRANSITION_FRAMES = 15;

export const CudaModule6: React.FC = () => {
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
export const MODULE6_SLIDE_DURATIONS = [20, 36, 38, 40, 38, 42, 38, 38, 38, 36, 38, 36, 40, 36, 40, 34];
