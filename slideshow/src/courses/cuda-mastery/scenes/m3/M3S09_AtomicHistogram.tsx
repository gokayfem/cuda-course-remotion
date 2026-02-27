import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint, FadeInText } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

const BIN_COLORS = [
  THEME.colors.nvidiaGreen,
  THEME.colors.accentBlue,
  THEME.colors.accentOrange,
  THEME.colors.accentPurple,
  THEME.colors.accentCyan,
  THEME.colors.accentPink,
  THEME.colors.accentYellow,
  THEME.colors.accentRed,
];

export const M3S09_AtomicHistogram: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const histogramCode = `__global__ void histogram(const int* data, int n,
                          int* hist, int numBins) {
  // Phase 1: shared memory histogram (fast atomics)
  __shared__ int sharedHist[256];
  int tid = threadIdx.x;
  if (tid < numBins) sharedHist[tid] = 0;
  __syncthreads();

  // Phase 2: each thread processes its elements
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    atomicAdd(&sharedHist[data[i]], 1);  // shared mem atomic
  }
  __syncthreads();

  // Phase 3: one global atomic per bin (few operations)
  if (tid < numBins) {
    atomicAdd(&hist[tid], sharedHist[tid]);  // global atomic
  }
}`;

  // Histogram visualization
  const NUM_BINS = 8;
  const binHeights = [45, 72, 38, 85, 62, 28, 55, 48];
  const maxHeight = 100;
  const BIN_WIDTH = 50;
  const BIN_GAP = 8;

  const histDelay = 3 * fps;
  const histSpring = spring({
    frame: frame - histDelay,
    fps,
    config: { damping: 200 },
  });
  const histOpacity = interpolate(histSpring, [0, 1], [0, 1]);

  // Data flow animation
  const dataFlowProgress = interpolate(
    frame - 4 * fps,
    [0, 2 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Comparison timing
  const compDelay = 6 * fps;
  const compSpring = spring({
    frame: frame - compDelay,
    fps,
    config: { damping: 200 },
  });
  const compOpacity = interpolate(compSpring, [0, 1], [0, 1]);

  const renderHistogram = () => (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-start",
        gap: 8,
        opacity: histOpacity,
      }}
    >
      {/* Data input (simulated) */}
      <div style={{ display: "flex", gap: 4, alignItems: "center", marginBottom: 4 }}>
        <span
          style={{
            fontSize: 13,
            color: THEME.colors.textMuted,
            fontFamily: fontFamilyBody,
            marginRight: 6,
          }}
        >
          Input data:
        </span>
        {[3, 1, 4, 1, 5, 2, 6, 5, 3, 7, 0, 2, 4, 6, 1, 3].map((val, i) => {
          const dotOpacity = interpolate(
            dataFlowProgress,
            [i * 0.05, i * 0.05 + 0.1],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          return (
            <div
              key={i}
              style={{
                width: 22,
                height: 22,
                borderRadius: 3,
                backgroundColor: `${BIN_COLORS[val]}30`,
                border: `1px solid ${BIN_COLORS[val]}60`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 11,
                color: BIN_COLORS[val],
                fontFamily: fontFamilyCode,
                fontWeight: 600,
                opacity: dotOpacity,
              }}
            >
              {val}
            </div>
          );
        })}
      </div>

      {/* Arrow */}
      <div
        style={{
          width: 2,
          height: 16,
          backgroundColor: THEME.colors.textMuted,
          marginLeft: 40,
          opacity: dataFlowProgress > 0.3 ? 1 : 0,
        }}
      />

      {/* Histogram bars */}
      <div style={{ display: "flex", alignItems: "flex-end", gap: BIN_GAP }}>
        {Array.from({ length: NUM_BINS }).map((_, i) => {
          const barDelay = histDelay + 0.5 * fps + i * 0.1 * fps;
          const barProgress = interpolate(
            frame - barDelay,
            [0, 1 * fps],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          const barHeight =
            (binHeights[i] / maxHeight) * 80 * barProgress * dataFlowProgress;

          return (
            <div
              key={i}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 4,
              }}
            >
              {/* Count label */}
              <span
                style={{
                  fontSize: 12,
                  color: BIN_COLORS[i],
                  fontFamily: fontFamilyCode,
                  fontWeight: 700,
                  opacity: dataFlowProgress > 0.5 ? 1 : 0,
                }}
              >
                {Math.round(binHeights[i] * dataFlowProgress)}
              </span>
              {/* Bar */}
              <div
                style={{
                  width: BIN_WIDTH,
                  height: barHeight,
                  backgroundColor: `${BIN_COLORS[i]}30`,
                  border: `1.5px solid ${BIN_COLORS[i]}`,
                  borderRadius: 4,
                  minHeight: 4,
                }}
              />
              {/* Bin label */}
              <span
                style={{
                  fontSize: 12,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyCode,
                }}
              >
                [{i}]
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );

  const renderComparison = () => (
    <div
      style={{
        display: "flex",
        gap: 20,
        opacity: compOpacity,
      }}
    >
      {/* Naive approach */}
      <div
        style={{
          flex: 1,
          padding: "10px 14px",
          backgroundColor: "rgba(255,82,82,0.08)",
          border: `1px solid ${THEME.colors.accentRed}40`,
          borderRadius: 8,
        }}
      >
        <div
          style={{
            fontSize: 15,
            color: THEME.colors.accentRed,
            fontFamily: fontFamilyBody,
            fontWeight: 700,
            marginBottom: 6,
          }}
        >
          Naive: Global Atomics Only
        </div>
        <div
          style={{
            fontSize: 13,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
            lineHeight: 1.5,
          }}
        >
          Every thread does atomicAdd to global memory. Massive contention when many threads hit the same bin.
        </div>
      </div>

      {/* Optimized approach */}
      <div
        style={{
          flex: 1,
          padding: "10px 14px",
          backgroundColor: "rgba(118,185,0,0.08)",
          border: `1px solid ${THEME.colors.nvidiaGreen}40`,
          borderRadius: 8,
        }}
      >
        <div
          style={{
            fontSize: 15,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyBody,
            fontWeight: 700,
            marginBottom: 6,
          }}
        >
          Optimized: Shared + Global
        </div>
        <div
          style={{
            fontSize: 13,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
            lineHeight: 1.5,
          }}
        >
          Per-block shared mem histogram (fast atomics), then one global atomicAdd per bin. Up to 10x faster.
        </div>
      </div>
    </div>
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={3}>
      <SlideTitle
        title="Atomic Histogram"
        subtitle="A practical pattern: shared memory atomics + global merge"
      />

      <div style={{ display: "flex", gap: 36, flex: 1 }}>
        {/* Left: Code */}
        <div style={{ flex: 1 }}>
          <CodeBlock
            code={histogramCode}
            title="histogram.cu"
            fontSize={13}
            delay={0.5 * fps}
            highlightLines={[13, 19]}
          />
        </div>

        {/* Right: Visualization + comparison */}
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            gap: 12,
          }}
        >
          {renderHistogram()}
          {renderComparison()}
        </div>
      </div>

      {/* Key insight */}
      <div
        style={{
          marginTop: 8,
          padding: "10px 24px",
          backgroundColor: "rgba(118,185,0,0.10)",
          borderRadius: 10,
          border: `2px solid ${THEME.colors.nvidiaGreen}50`,
          textAlign: "center",
          opacity: interpolate(
            frame - 7.5 * fps,
            [0, 0.5 * fps],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          ),
        }}
      >
        <span
          style={{
            fontSize: 18,
            color: THEME.colors.textPrimary,
            fontFamily: fontFamilyBody,
            fontWeight: 700,
          }}
        >
          Pattern:{" "}
          <span style={{ color: THEME.colors.nvidiaGreen }}>
            Reduce locally in shared memory
          </span>
          , then merge globally with one atomic per bin.
        </span>
      </div>
    </SlideLayout>
  );
};
