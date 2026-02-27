import React from "react";
import { AbsoluteFill } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";

import { M2S01_Title } from "./scenes/m2/M2S01_Title";
import { M2S02_MemoryHierarchy } from "./scenes/m2/M2S02_MemoryHierarchy";
import { M2S03_GlobalMemory } from "./scenes/m2/M2S03_GlobalMemory";
import { M2S04_Coalescing } from "./scenes/m2/M2S04_Coalescing";
import { M2S05_CoalescingCode } from "./scenes/m2/M2S05_CoalescingCode";
import { M2S06_SoAvsAoS } from "./scenes/m2/M2S06_SoAvsAoS";
import { M2S07_SharedMemoryIntro } from "./scenes/m2/M2S07_SharedMemoryIntro";
import { M2S08_SharedMemoryCode } from "./scenes/m2/M2S08_SharedMemoryCode";
import { M2S09_SharedMemTranspose } from "./scenes/m2/M2S09_SharedMemTranspose";
import { M2S10_BankConflicts } from "./scenes/m2/M2S10_BankConflicts";
import { M2S11_BankConflictFix } from "./scenes/m2/M2S11_BankConflictFix";
import { M2S12_Registers } from "./scenes/m2/M2S12_Registers";
import { M2S13_RegisterSpilling } from "./scenes/m2/M2S13_RegisterSpilling";
import { M2S14_ConstantMemory } from "./scenes/m2/M2S14_ConstantMemory";
import { M2S18_Summary } from "./scenes/m2/M2S18_Summary";

const FPS = 30;

const slides: Array<{
  component: React.FC;
  durationSeconds: number;
}> = [
  { component: M2S01_Title, durationSeconds: 20 },
  { component: M2S02_MemoryHierarchy, durationSeconds: 38 },
  { component: M2S03_GlobalMemory, durationSeconds: 32 },
  { component: M2S04_Coalescing, durationSeconds: 42 },         // Key concept — more time
  { component: M2S05_CoalescingCode, durationSeconds: 38 },
  { component: M2S06_SoAvsAoS, durationSeconds: 36 },
  { component: M2S07_SharedMemoryIntro, durationSeconds: 38 },
  { component: M2S08_SharedMemoryCode, durationSeconds: 40 },
  { component: M2S09_SharedMemTranspose, durationSeconds: 38 },
  { component: M2S10_BankConflicts, durationSeconds: 40 },      // Complex concept
  { component: M2S11_BankConflictFix, durationSeconds: 35 },
  { component: M2S12_Registers, durationSeconds: 35 },
  { component: M2S13_RegisterSpilling, durationSeconds: 32 },
  { component: M2S14_ConstantMemory, durationSeconds: 34 },
  { component: M2S18_Summary, durationSeconds: 32 },
];

const TRANSITION_FRAMES = 15;

export const CudaModule2: React.FC = () => {
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
export const MODULE2_SLIDE_DURATIONS = [20, 38, 32, 42, 38, 36, 38, 40, 38, 40, 35, 35, 32, 34, 32];
