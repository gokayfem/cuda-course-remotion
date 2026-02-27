import React from "react";
import { Composition, Folder } from "remotion";
import { CudaModule1 } from "./courses/cuda-mastery/CudaModule1";
import { CudaModule2, MODULE2_SLIDE_DURATIONS } from "./courses/cuda-mastery/CudaModule2";
import { CudaModule3, MODULE3_SLIDE_DURATIONS } from "./courses/cuda-mastery/CudaModule3";
import { CudaModule4, MODULE4_SLIDE_DURATIONS } from "./courses/cuda-mastery/CudaModule4";
import { CudaModule5, MODULE5_SLIDE_DURATIONS } from "./courses/cuda-mastery/CudaModule5";
import { CudaModule6, MODULE6_SLIDE_DURATIONS } from "./courses/cuda-mastery/CudaModule6";
import { CudaModule7, MODULE7_SLIDE_DURATIONS } from "./courses/cuda-mastery/CudaModule7";
import { CudaModule8, MODULE8_SLIDE_DURATIONS } from "./courses/cuda-mastery/CudaModule8";
import { CudaModule9, MODULE9_SLIDE_DURATIONS } from "./courses/cuda-mastery/CudaModule9";
import { CudaModule10, MODULE10_SLIDE_DURATIONS } from "./courses/cuda-mastery/CudaModule10";

const FPS = 25;
const TRANSITION_FRAMES = 13;

function calcDuration(durations: number[]): number {
  const totalSeconds = durations.reduce((sum, d) => sum + d, 0);
  const numTransitions = durations.length - 1;
  return totalSeconds * FPS - numTransitions * TRANSITION_FRAMES;
}

const module1Durations = [27, 64, 64, 61, 55, 56, 55, 49, 45, 50, 54, 54, 52, 58, 52];

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Folder name="CUDA-Mastery">
        <Composition
          id="CudaModule1"
          component={CudaModule1}
          durationInFrames={calcDuration(module1Durations)}
          fps={FPS}
          width={1920}
          height={1080}
          defaultProps={{}}
        />
        <Composition
          id="CudaModule2"
          component={CudaModule2}
          durationInFrames={calcDuration(MODULE2_SLIDE_DURATIONS)}
          fps={FPS}
          width={1920}
          height={1080}
          defaultProps={{}}
        />
        <Composition
          id="CudaModule3"
          component={CudaModule3}
          durationInFrames={calcDuration(MODULE3_SLIDE_DURATIONS)}
          fps={FPS}
          width={1920}
          height={1080}
          defaultProps={{}}
        />
        <Composition
          id="CudaModule4"
          component={CudaModule4}
          durationInFrames={calcDuration(MODULE4_SLIDE_DURATIONS)}
          fps={FPS}
          width={1920}
          height={1080}
          defaultProps={{}}
        />
        <Composition
          id="CudaModule5"
          component={CudaModule5}
          durationInFrames={calcDuration(MODULE5_SLIDE_DURATIONS)}
          fps={FPS}
          width={1920}
          height={1080}
          defaultProps={{}}
        />
        <Composition
          id="CudaModule6"
          component={CudaModule6}
          durationInFrames={calcDuration(MODULE6_SLIDE_DURATIONS)}
          fps={FPS}
          width={1920}
          height={1080}
          defaultProps={{}}
        />
        <Composition
          id="CudaModule7"
          component={CudaModule7}
          durationInFrames={calcDuration(MODULE7_SLIDE_DURATIONS)}
          fps={FPS}
          width={1920}
          height={1080}
          defaultProps={{}}
        />
        <Composition
          id="CudaModule8"
          component={CudaModule8}
          durationInFrames={calcDuration(MODULE8_SLIDE_DURATIONS)}
          fps={FPS}
          width={1920}
          height={1080}
          defaultProps={{}}
        />
        <Composition
          id="CudaModule9"
          component={CudaModule9}
          durationInFrames={calcDuration(MODULE9_SLIDE_DURATIONS)}
          fps={FPS}
          width={1920}
          height={1080}
          defaultProps={{}}
        />
        <Composition
          id="CudaModule10"
          component={CudaModule10}
          durationInFrames={calcDuration(MODULE10_SLIDE_DURATIONS)}
          fps={FPS}
          width={1920}
          height={1080}
          defaultProps={{}}
        />
      </Folder>
    </>
  );
};
