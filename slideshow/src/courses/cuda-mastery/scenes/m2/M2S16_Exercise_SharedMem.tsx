import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M2S16_Exercise_SharedMem: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const showSolution = frame > 8 * fps;

  const diagramOpacity = interpolate(
    frame - 4 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const insightOpacity = interpolate(
    frame - 11 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Tree reduction visualization data
  const reductionSteps = [
    { stride: 128, activeThreads: "0..127", label: "Step 1: stride=128" },
    { stride: 64, activeThreads: "0..63", label: "Step 2: stride=64" },
    { stride: 32, activeThreads: "0..31", label: "Step 3: stride=32" },
    { stride: 1, activeThreads: "0", label: "Final: result in sdata[0]" },
  ];

  return (
    <SlideLayout variant="accent" moduleNumber={2} slideNumber={16} totalSlides={18}>
      <SlideTitle
        title="Exercise: Shared Memory Reduction"
        subtitle="Compute the sum of an array using parallel reduction within each block"
      />

      <div style={{ display: "flex", gap: 36, flex: 1 }}>
        {/* Left: Challenge skeleton */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text="Your Challenge"
            delay={0.5 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentOrange}
            style={{ marginBottom: 12 }}
          />

          <CodeBlock
            delay={1 * fps}
            title="reduction_skeleton.cu"
            fontSize={15}
            code={`__global__ void block_reduce(
    const float *input, float *output, int N
) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Step 1: Load to shared memory
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    // Step 2: Tree reduction
    // YOUR CODE HERE
    // Hint: halve the stride each iteration
    // Don't forget __syncthreads()!

    // Step 3: Write result
    if (tid == 0) output[blockIdx.x] = sdata[0];
}`}
            highlightLines={[12, 13, 14, 15]}
          />

          {/* Tree reduction diagram */}
          <div style={{ marginTop: 14, opacity: diagramOpacity }}>
            <div style={{ fontSize: 15, color: THEME.colors.accentCyan, fontFamily: fontFamilyBody, fontWeight: 700, marginBottom: 8 }}>
              Tree Reduction Pattern:
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {reductionSteps.map((step, i) => {
                const stepDelay = 4.5 * fps + i * 0.4 * fps;
                const s = spring({ frame: frame - stepDelay, fps, config: { damping: 200 } });
                const barWidth = interpolate(s, [0, 1], [0, 1]);
                const widthPct = i === 3 ? 3 : 100 / Math.pow(2, i);

                return (
                  <div key={step.stride} style={{ display: "flex", alignItems: "center", gap: 10, opacity: interpolate(s, [0, 1], [0, 1]) }}>
                    <span style={{ width: 140, fontSize: 13, color: THEME.colors.textSecondary, fontFamily: fontFamilyCode }}>
                      {step.label}
                    </span>
                    <div style={{ flex: 1, height: 14, backgroundColor: "rgba(255,255,255,0.05)", borderRadius: 3 }}>
                      <div
                        style={{
                          width: `${widthPct * barWidth}%`,
                          height: "100%",
                          backgroundColor: i === 3 ? THEME.colors.nvidiaGreen : THEME.colors.accentCyan,
                          borderRadius: 3,
                          minWidth: barWidth > 0 ? 4 : 0,
                        }}
                      />
                    </div>
                    <span style={{ fontSize: 12, color: THEME.colors.textMuted, fontFamily: fontFamilyCode, width: 60 }}>
                      T{step.activeThreads}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right: Solution */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text={showSolution ? "Solution" : "Think about it first..."}
            delay={showSolution ? 8 * fps : 5 * fps}
            fontSize={22}
            fontWeight={700}
            color={showSolution ? THEME.colors.nvidiaGreen : THEME.colors.textMuted}
            style={{ marginBottom: 12 }}
          />

          {showSolution && (
            <CodeBlock
              delay={8.5 * fps}
              title="reduction_solution.cu"
              fontSize={15}
              code={`__global__ void block_reduce(
    const float *input, float *output, int N
) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Load to shared memory
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    // Tree reduction: stride halves each step
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // CRITICAL: sync each step!
    }

    // Thread 0 has the block's sum
    if (tid == 0) output[blockIdx.x] = sdata[0];
}`}
              highlightLines={[13, 14, 15, 17]}
            />
          )}

          {showSolution && (
            <div
              style={{
                marginTop: 14,
                padding: "14px 18px",
                backgroundColor: "rgba(118,185,0,0.08)",
                borderRadius: 8,
                opacity: insightOpacity,
              }}
            >
              <div style={{ fontSize: 15, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.7 }}>
                <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>Key points:</span>
              </div>
              <div style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, lineHeight: 1.7, marginTop: 6 }}>
                <span style={{ fontFamily: fontFamilyCode, color: THEME.colors.accentCyan }}>__syncthreads()</span> between every reduction step is{" "}
                <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>mandatory</span> — without it, threads may read stale data.
              </div>
              <div style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, lineHeight: 1.7, marginTop: 4 }}>
                Log2(256) = 8 steps to reduce 256 elements. For full array: reduce per-block, then reduce the block results.
              </div>
            </div>
          )}

          {!showSolution && (
            <div
              style={{
                marginTop: 16,
                padding: "14px 18px",
                backgroundColor: "rgba(79,195,247,0.08)",
                borderRadius: 8,
                borderLeft: `4px solid ${THEME.colors.accentBlue}`,
                opacity: interpolate(frame - 5 * fps, [0, 0.5 * fps], [0, 1], {
                  extrapolateLeft: "clamp", extrapolateRight: "clamp",
                }),
              }}
            >
              <div style={{ fontSize: 16, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.6 }}>
                <span style={{ color: THEME.colors.accentBlue, fontWeight: 700 }}>Think:</span>{" "}
                How do you combine values from 256 threads into 1 result? Each step should halve the number of active threads.
              </div>
            </div>
          )}
        </div>
      </div>
    </SlideLayout>
  );
};
