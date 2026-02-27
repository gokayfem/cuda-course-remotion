import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint, FadeInText } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M3S06_Syncthreads: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const timelineDelay = 4 * fps;
  const timelineSpring = spring({
    frame: frame - timelineDelay,
    fps,
    config: { damping: 200 },
  });
  const timelineOpacity = interpolate(timelineSpring, [0, 1], [0, 1]);

  const warningDelay = 6.5 * fps;
  const warningOpacity = interpolate(
    frame - warningDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const syncCode = `__global__ void reduce(float* data, float* out) {
  __shared__ float sdata[256];
  int tid = threadIdx.x;

  // Load into shared memory
  sdata[tid] = data[blockIdx.x * blockDim.x + tid];

  __syncthreads();  // BARRIER: all threads must finish loading

  // Now safe to read any element in sdata[]
  if (tid < 128)
    sdata[tid] += sdata[tid + 128];

  __syncthreads();  // BARRIER: wait for this reduction step

  if (tid < 64)
    sdata[tid] += sdata[tid + 64];
}`;

  const deadlockCode = `// DEADLOCK! Not all threads hit __syncthreads
if (threadIdx.x < 128) {
  __syncthreads();  // Only half the block reaches this!
}`;

  // Timeline visualization: threads arriving at barrier
  const THREAD_COUNT = 8;
  const threadArrivals = [0.2, 0.5, 0.3, 0.8, 0.6, 0.4, 0.7, 0.1];

  const renderTimeline = () => {
    const barrierX = 380;
    const containerWidth = 520;
    const startY = 0;
    const threadHeight = 22;
    const threadGap = 4;

    return (
      <div
        style={{
          position: "relative",
          width: containerWidth,
          height: THREAD_COUNT * (threadHeight + threadGap) + 40,
          opacity: timelineOpacity,
        }}
      >
        {/* Barrier line */}
        <div
          style={{
            position: "absolute",
            left: barrierX,
            top: 0,
            width: 3,
            height: THREAD_COUNT * (threadHeight + threadGap),
            backgroundColor: THEME.colors.accentRed,
            borderRadius: 2,
          }}
        />
        <div
          style={{
            position: "absolute",
            left: barrierX - 40,
            top: THREAD_COUNT * (threadHeight + threadGap) + 4,
            fontSize: 12,
            color: THEME.colors.accentRed,
            fontFamily: fontFamilyCode,
            fontWeight: 700,
          }}
        >
          __syncthreads()
        </div>

        {/* Before/After labels */}
        <div
          style={{
            position: "absolute",
            left: barrierX / 2 - 30,
            top: THREAD_COUNT * (threadHeight + threadGap) + 4,
            fontSize: 11,
            color: THEME.colors.textMuted,
            fontFamily: fontFamilyBody,
          }}
        >
          Load phase
        </div>
        <div
          style={{
            position: "absolute",
            left: barrierX + 40,
            top: THREAD_COUNT * (threadHeight + threadGap) + 4,
            fontSize: 11,
            color: THEME.colors.textMuted,
            fontFamily: fontFamilyBody,
          }}
        >
          Compute phase
        </div>

        {/* Thread bars */}
        {Array.from({ length: THREAD_COUNT }).map((_, i) => {
          const arrivalProgress = interpolate(
            frame - timelineDelay,
            [0, 1.5 * fps],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );

          const threadReached =
            arrivalProgress >= threadArrivals[i];
          const allReached = arrivalProgress >= 0.85;

          const threadBarWidth = threadReached
            ? barrierX - 10
            : arrivalProgress * barrierX * (threadArrivals[i] + 0.3);

          const afterBarWidth = allReached
            ? interpolate(
                frame - (timelineDelay + 2 * fps),
                [0, 0.8 * fps],
                [0, containerWidth - barrierX - 20],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              )
            : 0;

          const yPos = startY + i * (threadHeight + threadGap);

          return (
            <React.Fragment key={i}>
              {/* Thread label */}
              <div
                style={{
                  position: "absolute",
                  left: 0,
                  top: yPos + 2,
                  fontSize: 11,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyCode,
                  width: 24,
                }}
              >
                T{i}
              </div>

              {/* Before barrier */}
              <div
                style={{
                  position: "absolute",
                  left: 28,
                  top: yPos,
                  width: Math.min(threadBarWidth, barrierX - 38),
                  height: threadHeight,
                  backgroundColor: threadReached
                    ? `${THEME.colors.accentBlue}30`
                    : `${THEME.colors.accentBlue}15`,
                  border: `1px solid ${threadReached ? THEME.colors.accentBlue : `${THEME.colors.accentBlue}40`}`,
                  borderRadius: 3,
                }}
              />

              {/* Waiting indicator */}
              {threadReached && !allReached && (
                <div
                  style={{
                    position: "absolute",
                    left: barrierX - 12,
                    top: yPos + 3,
                    fontSize: 12,
                    color: THEME.colors.accentOrange,
                  }}
                >
                  ...
                </div>
              )}

              {/* After barrier */}
              {afterBarWidth > 0 && (
                <div
                  style={{
                    position: "absolute",
                    left: barrierX + 8,
                    top: yPos,
                    width: afterBarWidth,
                    height: threadHeight,
                    backgroundColor: `${THEME.colors.nvidiaGreen}30`,
                    border: `1px solid ${THEME.colors.nvidiaGreen}60`,
                    borderRadius: 3,
                  }}
                />
              )}
            </React.Fragment>
          );
        })}
      </div>
    );
  };

  const renderLeft = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <SlideTitle
        title="__syncthreads()"
        subtitle="Block-level barrier synchronization"
      />

      <CodeBlock
        code={syncCode}
        title="reduction.cu"
        fontSize={13}
        delay={0.5 * fps}
        highlightLines={[8, 13]}
      />
    </div>
  );

  const renderRight = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <FadeInText
        text="Why Barriers Are Needed"
        fontSize={20}
        fontWeight={700}
        delay={1 * fps}
        color={THEME.colors.accentBlue}
      />

      <BulletPoint
        index={0}
        delay={1.5 * fps}
        text="Prevents data races"
        subtext="Without sync, thread A may read shared memory before thread B finishes writing."
        highlight
      />
      <BulletPoint
        index={1}
        delay={1.5 * fps}
        text="Block-scope only"
        subtext="Only synchronizes threads within the same block. Cannot sync across blocks."
      />
      <BulletPoint
        index={2}
        delay={1.5 * fps}
        text="All threads must reach it"
        subtext="Every thread in the block must call __syncthreads() or the block deadlocks."
      />

      {/* Timeline visualization */}
      <FadeInText
        text="Thread execution timeline:"
        fontSize={16}
        fontWeight={600}
        delay={3.5 * fps}
        style={{ marginTop: 4 }}
      />
      {renderTimeline()}

      {/* Deadlock warning */}
      <div style={{ opacity: warningOpacity }}>
        <CodeBlock
          code={deadlockCode}
          title="DEADLOCK BUG"
          fontSize={13}
          delay={warningDelay}
          animateLines={false}
        />
      </div>
    </div>
  );

  return (
    <TwoColumnLayout
      moduleNumber={3}
      left={renderLeft()}
      right={renderRight()}
      leftWidth="45%"
    />
  );
};
