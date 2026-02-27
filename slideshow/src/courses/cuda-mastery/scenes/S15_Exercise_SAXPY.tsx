import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, FadeInText, BulletPoint } from "../../../components/AnimatedText";
import { CodeBlock } from "../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../styles/fonts";

export const S15_Exercise_SAXPY: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Show solution animation
  const showSolution = frame > 8 * fps;

  return (
    <SlideLayout variant="accent" slideNumber={15} totalSlides={18}>
      <SlideTitle
        title="Exercise: SAXPY Kernel"
        subtitle="result[i] = a * x[i] + y[i] — a fundamental ML operation"
      />

      <div style={{ display: "flex", gap: 40, flex: 1 }}>
        {/* Left: Challenge */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text="Your Challenge"
            delay={0.5 * fps}
            fontSize={24}
            fontWeight={700}
            color={THEME.colors.accentOrange}
            style={{ marginBottom: 16 }}
          />

          <CodeBlock
            delay={1 * fps}
            title="Implement this kernel"
            fontSize={18}
            code={`// SAXPY: Single-precision A*X Plus Y
// Used in: gradient updates, linear layers,
// optimizer steps, batch normalization
__global__ void saxpy(
    float a,
    const float *x, const float *y,
    float *result, int n
) {
    // YOUR CODE HERE
    // 1. Compute global thread index
    // 2. Bounds check
    // 3. result[i] = a * x[i] + y[i]
}

// Launch:
saxpy<<<(N+255)/256, 256>>>(
    2.5f, d_x, d_y, d_result, N
);`}
            highlightLines={[9, 10, 11, 12]}
          />

          <div style={{ marginTop: 16 }}>
            <BulletPoint index={0} delay={3 * fps} text="This is MEMORY BOUND (not compute-bound)" />
            <BulletPoint index={1} delay={3 * fps} text="2 reads + 1 write + 1 FMA per element" />
            <BulletPoint index={2} delay={3 * fps} text="Peak: ~80% memory bandwidth utilization" />
          </div>
        </div>

        {/* Right: Solution (appears later) */}
        <div style={{ flex: 0.8 }}>
          <FadeInText
            text={showSolution ? "Solution" : "Think about it first..."}
            delay={showSolution ? 8 * fps : 5 * fps}
            fontSize={24}
            fontWeight={700}
            color={showSolution ? THEME.colors.nvidiaGreen : THEME.colors.textMuted}
            style={{ marginBottom: 16 }}
          />

          {showSolution && (
            <CodeBlock
              delay={8.5 * fps}
              title="saxpy_solution.cu"
              fontSize={18}
              code={`__global__ void saxpy(
    float a,
    const float *x, const float *y,
    float *result, int n
) {
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;

    if (idx < n) {
        result[idx] = a * x[idx] + y[idx];
    }
}`}
              highlightLines={[6, 7, 9, 10]}
            />
          )}

          {showSolution && (
            <div
              style={{
                marginTop: 16,
                padding: "14px 18px",
                backgroundColor: "rgba(118,185,0,0.08)",
                borderRadius: 8,
                opacity: interpolate(
                  frame - 10 * fps,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <div style={{ fontSize: 16, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.6 }}>
                Just 3 lines of real logic! The pattern is always the same:
              </div>
              <div style={{ fontFamily: fontFamilyCode, fontSize: 16, color: THEME.colors.nvidiaGreen, marginTop: 8, lineHeight: 1.8 }}>
                1. Compute index<br />
                2. Bounds check<br />
                3. Do the math
              </div>
            </div>
          )}
        </div>
      </div>
    </SlideLayout>
  );
};
