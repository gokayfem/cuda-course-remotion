import React from "react";
import { AbsoluteFill } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";

import { S01_TitleSlide } from "./scenes/S01_TitleSlide";
import { S02_WhyGPU } from "./scenes/S02_WhyGPU";
import { S03_CPUvsGPU } from "./scenes/S03_CPUvsGPU";
import { S04_CUDAModel } from "./scenes/S04_CUDAModel";
import { S05_CUDAQualifiers } from "./scenes/S05_CUDAQualifiers";
import { S06_FirstKernel } from "./scenes/S06_FirstKernel";
import { S07_MemoryModel } from "./scenes/S07_MemoryModel";
import { S08_VectorAdd } from "./scenes/S08_VectorAdd";
import { S09_VectorAddFull } from "./scenes/S09_VectorAddFull";
import { S10_ThreadIndexing1D } from "./scenes/S10_ThreadIndexing1D";
import { S11_ThreadIndexing2D } from "./scenes/S11_ThreadIndexing2D";
import { S12_GridStrideLoop } from "./scenes/S12_GridStrideLoop";
import { S13_ErrorHandling } from "./scenes/S13_ErrorHandling";
import { S14_GPUTiming } from "./scenes/S14_GPUTiming";
import { S18_Summary } from "./scenes/S18_Summary";

// 10 minutes at 30fps = 18,000 frames
// 18 slides: ~33 seconds each = ~1000 frames each
// Some slides need more time (code, quizzes), some less (title)

const FPS = 25;

const slides: Array<{
  component: React.FC;
  durationSeconds: number;
}> = [
  { component: S01_TitleSlide, durationSeconds: 27 },     // Title - shorter
  { component: S02_WhyGPU, durationSeconds: 64 },         // Why GPU - medium
  { component: S03_CPUvsGPU, durationSeconds: 64 },       // CPU vs GPU - needs time for diagrams
  { component: S04_CUDAModel, durationSeconds: 61 },      // CUDA Model - complex diagram
  { component: S05_CUDAQualifiers, durationSeconds: 55 }, // Qualifiers - table + code
  { component: S06_FirstKernel, durationSeconds: 56 },    // First kernel - important
  { component: S07_MemoryModel, durationSeconds: 55 },    // Memory model - key concept
  { component: S08_VectorAdd, durationSeconds: 49 },      // Vector add kernel - detailed
  { component: S09_VectorAddFull, durationSeconds: 45 },  // Full host code
  { component: S10_ThreadIndexing1D, durationSeconds: 50 }, // 1D indexing
  { component: S11_ThreadIndexing2D, durationSeconds: 54 }, // 2D indexing
  { component: S12_GridStrideLoop, durationSeconds: 54 },   // Grid stride - important pattern
  { component: S13_ErrorHandling, durationSeconds: 52 },    // Error handling
  { component: S14_GPUTiming, durationSeconds: 58 },        // Timing & bandwidth
  { component: S18_Summary, durationSeconds: 52 },          // Summary
];

const TRANSITION_FRAMES = 13;

export const CudaModule1: React.FC = () => {
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

          // Add transition between slides (not after last)
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
