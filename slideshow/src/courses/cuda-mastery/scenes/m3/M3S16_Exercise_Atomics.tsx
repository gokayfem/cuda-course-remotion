import React from "react";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

const HistogramBar: React.FC<{
  bin: number;
  count: number;
  maxCount: number;
  color: string;
  delay: number;
}> = ({ bin, count, maxCount, color, delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const barSpring = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });
  const barHeight = interpolate(barSpring, [0, 1], [0, (count / maxCount) * 120]);
  const opacity = interpolate(barSpring, [0, 1], [0, 1]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 4,
        opacity,
      }}
    >
      <span style={{ fontSize: 12, color, fontFamily: fontFamilyCode, fontWeight: 700 }}>
        {count}
      </span>
      <div
        style={{
          width: 32,
          height: barHeight,
          backgroundColor: `${color}40`,
          border: `1px solid ${color}`,
          borderRadius: 3,
        }}
      />
      <span style={{ fontSize: 11, color: THEME.colors.textMuted, fontFamily: fontFamilyCode }}>
        [{bin}]
      </span>
    </div>
  );
};

export const M3S16_Exercise_Atomics: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const showSolution = frame > 8 * fps;

  const hintOpacity = interpolate(
    frame - 4 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const histogramData = [12, 28, 45, 38, 22, 15, 8, 31];
  const maxCount = Math.max(...histogramData);
  const binColors = [
    THEME.colors.accentRed,
    THEME.colors.accentOrange,
    THEME.colors.accentYellow,
    THEME.colors.nvidiaGreen,
    THEME.colors.accentCyan,
    THEME.colors.accentBlue,
    THEME.colors.accentPurple,
    THEME.colors.accentPink,
  ];

  const resultOpacity = interpolate(
    frame - 12 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="accent" moduleNumber={3}>
      <SlideTitle
        title="Exercise: Histogram Kernel"
        subtitle="Implement an efficient histogram using atomic operations"
      />

      <div style={{ display: "flex", gap: 36, flex: 1 }}>
        {/* Left: Problem + skeleton */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text="The Task"
            delay={0.5 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentOrange}
            style={{ marginBottom: 8 }}
          />

          <FadeInText
            text="Given an array of values 0-7, count occurrences of each value."
            delay={0.8 * fps}
            fontSize={16}
            color={THEME.colors.textSecondary}
            style={{ marginBottom: 12 }}
          />

          <CodeBlock
            delay={1.2 * fps}
            title="histogram_skeleton.cu"
            fontSize={14}
            code={`__global__ void histogram(
    const int* input, int* bins,
    int n, int numBins
) {
    // TODO: Declare shared memory for local histogram
    // __shared__ int ???[???];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: Initialize shared memory bins to 0

    // TODO: Sync threads after initialization

    // TODO: Accumulate into LOCAL shared memory bins
    //       (which atomic operation?)

    // TODO: Sync threads before writing back

    // TODO: Add local bins to GLOBAL bins
    //       (which atomic operation?)
}`}
            highlightLines={[6, 10, 12, 14, 15, 17, 19, 20]}
          />

          {/* Hint */}
          <div
            style={{
              marginTop: 12,
              padding: "10px 16px",
              backgroundColor: "rgba(79,195,247,0.08)",
              borderRadius: 8,
              borderLeft: `4px solid ${THEME.colors.accentBlue}`,
              opacity: hintOpacity,
            }}
          >
            <span style={{ fontSize: 15, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.5 }}>
              <span style={{ color: THEME.colors.accentBlue, fontWeight: 700 }}>Hint:</span>{" "}
              Use shared memory atomics (fast) within each block, then global atomics (slow)
              only once per bin per block. This is called a two-phase approach.
            </span>
          </div>

          {/* Histogram visualization */}
          <div
            style={{
              marginTop: 12,
              display: "flex",
              alignItems: "flex-end",
              gap: 8,
              padding: "8px 12px",
              backgroundColor: "rgba(255,255,255,0.02)",
              borderRadius: 8,
              opacity: interpolate(
                frame - 3 * fps,
                [0, 0.4 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            {histogramData.map((count, i) => (
              <HistogramBar
                key={i}
                bin={i}
                count={count}
                maxCount={maxCount}
                color={binColors[i]}
                delay={3.2 * fps + i * 0.15 * fps}
              />
            ))}
          </div>
        </div>

        {/* Right: Solution */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text={showSolution ? "Solution: Two-Phase Histogram" : "Think about it first..."}
            delay={showSolution ? 8 * fps : 5 * fps}
            fontSize={20}
            fontWeight={700}
            color={showSolution ? THEME.colors.nvidiaGreen : THEME.colors.textMuted}
            style={{ marginBottom: 10 }}
          />

          {showSolution && (
            <>
              <CodeBlock
                delay={8.5 * fps}
                title="histogram_solution.cu (FAST)"
                fontSize={13}
                code={`__global__ void histogram(
    const int* input, int* bins,
    int n, int numBins
) {
    __shared__ int localBins[256]; // max bins

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;

    // Phase 0: Clear local bins
    for (int i = lane; i < numBins; i += blockDim.x)
        localBins[i] = 0;
    __syncthreads();

    // Phase 1: Accumulate into shared memory
    if (tid < n) {
        int bin = input[tid];
        atomicAdd(&localBins[bin], 1); // FAST
    }
    __syncthreads();

    // Phase 2: Merge local bins into global
    for (int i = lane; i < numBins; i += blockDim.x) {
        if (localBins[i] > 0)
            atomicAdd(&bins[i], localBins[i]); // once/block
    }
}`}
                highlightLines={[5, 11, 12, 18, 24, 25]}
              />

              <div
                style={{
                  marginTop: 12,
                  padding: "10px 14px",
                  backgroundColor: "rgba(118,185,0,0.08)",
                  borderRadius: 8,
                  opacity: resultOpacity,
                }}
              >
                <div style={{ fontSize: 14, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.6 }}>
                  <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>Why two phases?</span>
                  <br />
                  <span style={{ fontFamily: fontFamilyCode, fontSize: 13, color: THEME.colors.accentCyan }}>atomicAdd</span>{" "}
                  on shared memory is ~100x faster than global memory.
                  We do N shared atomics but only <span style={{ fontFamily: fontFamilyCode, fontSize: 13, color: THEME.colors.accentOrange }}>numBins * numBlocks</span>{" "}
                  global atomics.
                </div>
              </div>

              <div
                style={{
                  marginTop: 10,
                  padding: "10px 14px",
                  background: "rgba(255,171,64,0.08)",
                  borderRadius: 8,
                  borderLeft: `3px solid ${THEME.colors.accentOrange}`,
                  opacity: resultOpacity,
                }}
              >
                <div style={{ fontSize: 14, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.5 }}>
                  <span style={{ color: THEME.colors.accentOrange, fontWeight: 700 }}>Bonus optimization:</span>{" "}
                  For very few bins, use warp-level vote + popc to avoid even shared memory atomics.
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </SlideLayout>
  );
};
