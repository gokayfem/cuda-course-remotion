import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../components/AnimatedText";
import { CodeBlock } from "../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../styles/fonts";

export const S16_Exercise_ReLU: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const showSolution = frame > 7 * fps;

  // ReLU graph animation
  const graphProgress = interpolate(
    frame - 2 * fps,
    [0, 1.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="accent" slideNumber={16} totalSlides={18}>
      <SlideTitle
        title="Exercise: ReLU Activation Kernel"
        subtitle="output[i] = max(0, input[i]) — the most used activation in deep learning"
      />

      <div style={{ display: "flex", gap: 40, flex: 1 }}>
        {/* Left: Visual + Code */}
        <div style={{ flex: 1 }}>
          {/* ReLU graph */}
          <div
            style={{
              height: 160,
              position: "relative",
              marginBottom: 20,
              opacity: interpolate(
                frame - 1 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <svg width="100%" height="160" viewBox="0 0 500 160">
              {/* Axes */}
              <line x1="50" y1="80" x2="450" y2="80" stroke={THEME.colors.textMuted} strokeWidth="1" />
              <line x1="250" y1="10" x2="250" y2="150" stroke={THEME.colors.textMuted} strokeWidth="1" />
              <text x="455" y="85" fill={THEME.colors.textMuted} fontSize="14">x</text>
              <text x="255" y="20" fill={THEME.colors.textMuted} fontSize="14">y</text>

              {/* ReLU line - negative (y=0) */}
              <line
                x1="50"
                y1="80"
                x2={50 + 200 * graphProgress}
                y2="80"
                stroke={THEME.colors.accentRed}
                strokeWidth="3"
              />

              {/* ReLU line - positive (y=x) */}
              <line
                x1="250"
                y1="80"
                x2={250 + 180 * graphProgress}
                y2={80 - 160 * graphProgress}
                stroke={THEME.colors.nvidiaGreen}
                strokeWidth="3"
              />

              {/* Label */}
              <text x="320" y="40" fill={THEME.colors.nvidiaGreen} fontSize="16" fontWeight="700">
                {graphProgress > 0.5 ? "ReLU(x) = max(0, x)" : ""}
              </text>
              <text x="100" y="100" fill={THEME.colors.accentRed} fontSize="14">
                {graphProgress > 0.3 ? "= 0 when x < 0" : ""}
              </text>
            </svg>
          </div>

          <CodeBlock
            delay={3 * fps}
            title="Implement this kernel"
            fontSize={18}
            code={`__global__ void relu(
    const float *input,
    float *output, int n
) {
    // YOUR CODE HERE
    // Hint: use fmaxf(0.0f, value)
}

// Launch:
relu<<<(N+255)/256, 256>>>(
    d_input, d_output, N
);`}
            highlightLines={[5, 6]}
          />
        </div>

        {/* Right: Solution + explanation */}
        <div style={{ flex: 0.8 }}>
          <FadeInText
            text="Why ReLU matters in ML"
            delay={4 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentBlue}
            style={{ marginBottom: 12 }}
          />
          <FadeInText
            text="Every neural network layer applies an activation function. ReLU is O(1) per element, making it perfect for GPU parallelization."
            delay={4.3 * fps}
            fontSize={17}
            color={THEME.colors.textSecondary}
            style={{ marginBottom: 24 }}
          />

          {showSolution && (
            <>
              <FadeInText
                text="Solution"
                delay={7 * fps}
                fontSize={22}
                fontWeight={700}
                color={THEME.colors.nvidiaGreen}
                style={{ marginBottom: 12 }}
              />
              <CodeBlock
                delay={7.3 * fps}
                title="relu_solution.cu"
                fontSize={18}
                code={`__global__ void relu(
    const float *input,
    float *output, int n
) {
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f,
                            input[idx]);
    }
}`}
                highlightLines={[5, 6, 8, 9]}
              />
            </>
          )}

          {showSolution && (
            <FadeInText
              text="Same 3-line pattern: index, check, compute. You'll write this pattern hundreds of times!"
              delay={9 * fps}
              fontSize={16}
              color={THEME.colors.textSecondary}
              style={{ marginTop: 12 }}
            />
          )}
        </div>
      </div>
    </SlideLayout>
  );
};
