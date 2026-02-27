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

export const M4S16_Exercise_Reduction: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const showSolution = frame > 8 * fps;

  const insightOpacity = interpolate(
    frame - 11 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="accent" moduleNumber={4} slideNumber={16} totalSlides={18}>
      <SlideTitle
        title="Exercise: Dot Product"
        subtitle="Combine two patterns: elementwise multiply + parallel reduction"
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
            title="dot_product_skeleton.cu"
            fontSize={14}
            code={`__global__ void dot_product(
    const float *a, const float *b,
    float *block_results, int N
) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Step 1: Elementwise multiply -> shared mem
    sdata[tid] = (i < N) ? a[i] * b[i] : 0.0f;
    __syncthreads();

    // Step 2: Parallel reduction to sum
    // YOUR CODE HERE
    // Hint: same tree reduction as before
    // but the data is already in shared memory

    // Step 3: Thread 0 writes block result
    if (tid == 0)
        block_results[blockIdx.x] = sdata[0];
}
// Phase 2: reduce block_results on host
// or launch a second kernel`}
            highlightLines={[14, 15, 16]}
          />

          {/* Hint box before solution */}
          {!showSolution && (
            <div
              style={{
                marginTop: 14,
                padding: "14px 18px",
                backgroundColor: "rgba(79,195,247,0.08)",
                borderRadius: 8,
                borderLeft: `4px solid ${THEME.colors.accentBlue}`,
                opacity: interpolate(frame - 5 * fps, [0, 0.5 * fps], [0, 1], {
                  extrapolateLeft: "clamp",
                  extrapolateRight: "clamp",
                }),
              }}
            >
              <div
                style={{
                  fontSize: 16,
                  color: THEME.colors.textPrimary,
                  fontFamily: fontFamilyBody,
                  lineHeight: 1.6,
                }}
              >
                <span style={{ color: THEME.colors.accentBlue, fontWeight: 700 }}>
                  Think:
                </span>{" "}
                The multiply is already done. You just need the tree reduction loop
                from the previous exercise. What changes?
              </div>
            </div>
          )}
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
              title="dot_product_solution.cu"
              fontSize={14}
              code={`__global__ void dot_product(
    const float *a, const float *b,
    float *block_results, int N
) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Elementwise multiply into shared mem
    sdata[tid] = (i < N) ? a[i] * b[i] : 0.0f;
    __syncthreads();

    // Tree reduction: sum products
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Block's partial dot product
    if (tid == 0)
        block_results[blockIdx.x] = sdata[0];
}`}
              highlightLines={[10, 13, 14, 15, 16, 17]}
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
              <div
                style={{
                  fontSize: 15,
                  color: THEME.colors.textPrimary,
                  fontFamily: fontFamilyBody,
                  lineHeight: 1.7,
                }}
              >
                <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>
                  Key insight:
                </span>{" "}
                Dot product ={" "}
                <span style={{ fontFamily: fontFamilyCode, color: THEME.colors.accentCyan }}>
                  elementwise multiply
                </span>{" "}
                +{" "}
                <span style={{ fontFamily: fontFamilyCode, color: THEME.colors.accentCyan }}>
                  reduction
                </span>
              </div>
              <div
                style={{
                  fontSize: 14,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  lineHeight: 1.7,
                  marginTop: 6,
                }}
              >
                Patterns compose! The reduction code is identical -- only the input changes.
                This is the power of mastering fundamental parallel patterns.
              </div>
            </div>
          )}
        </div>
      </div>
    </SlideLayout>
  );
};
