import React from "react";
import { AbsoluteFill } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";

import { M9S01_Title } from "./scenes/m9/M9S01_Title";
import { M9S02_TransformerAnatomy } from "./scenes/m9/M9S02_TransformerAnatomy";
import { M9S03_SoftmaxIntro } from "./scenes/m9/M9S03_SoftmaxIntro";
import { M9S04_OnlineSoftmax } from "./scenes/m9/M9S04_OnlineSoftmax";
import { M9S05_SoftmaxCode } from "./scenes/m9/M9S05_SoftmaxCode";
import { M9S06_LayerNormIntro } from "./scenes/m9/M9S06_LayerNormIntro";
import { M9S07_LayerNormCode } from "./scenes/m9/M9S07_LayerNormCode";
import { M9S08_FlashAttentionIntro } from "./scenes/m9/M9S08_FlashAttentionIntro";
import { M9S09_FlashAttentionAlgo } from "./scenes/m9/M9S09_FlashAttentionAlgo";
import { M9S10_FusionIntro } from "./scenes/m9/M9S10_FusionIntro";
import { M9S11_FusionPatterns } from "./scenes/m9/M9S11_FusionPatterns";
import { M9S12_FusedBiasGelu } from "./scenes/m9/M9S12_FusedBiasGelu";
import { M9S13_FlashAttentionV2 } from "./scenes/m9/M9S13_FlashAttentionV2";
import { M9S14_TransformerOptimizations } from "./scenes/m9/M9S14_TransformerOptimizations";
import { M9S15_CaseStudy } from "./scenes/m9/M9S15_CaseStudy";
import { M9S18_Summary } from "./scenes/m9/M9S18_Summary";

const FPS = 30;

const slides: Array<{
  component: React.FC;
  durationSeconds: number;
}> = [
  { component: M9S01_Title, durationSeconds: 20 },
  { component: M9S02_TransformerAnatomy, durationSeconds: 36 },
  { component: M9S03_SoftmaxIntro, durationSeconds: 40 },
  { component: M9S04_OnlineSoftmax, durationSeconds: 42 },
  { component: M9S05_SoftmaxCode, durationSeconds: 40 },
  { component: M9S06_LayerNormIntro, durationSeconds: 40 },
  { component: M9S07_LayerNormCode, durationSeconds: 40 },
  { component: M9S08_FlashAttentionIntro, durationSeconds: 42 },
  { component: M9S09_FlashAttentionAlgo, durationSeconds: 42 },
  { component: M9S10_FusionIntro, durationSeconds: 38 },
  { component: M9S11_FusionPatterns, durationSeconds: 40 },
  { component: M9S12_FusedBiasGelu, durationSeconds: 38 },
  { component: M9S13_FlashAttentionV2, durationSeconds: 40 },
  { component: M9S14_TransformerOptimizations, durationSeconds: 38 },
  { component: M9S15_CaseStudy, durationSeconds: 40 },
  { component: M9S18_Summary, durationSeconds: 34 },
];

const TRANSITION_FRAMES = 15;

export const CudaModule9: React.FC = () => {
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
export const MODULE9_SLIDE_DURATIONS = [20, 36, 40, 42, 40, 40, 40, 42, 42, 38, 40, 38, 40, 38, 40, 34];
