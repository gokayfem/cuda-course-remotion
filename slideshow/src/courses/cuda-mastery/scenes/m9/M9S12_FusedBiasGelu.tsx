import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

type PerfBar = {
  label: string;
  passes: number;
  speedup: string;
  color: string;
  widthPct: number;
};

const perfBars: PerfBar[] = [
  {
    label: "Separate (2 passes)",
    passes: 2,
    speedup: "1.0x",
    color: THEME.colors.accentRed,
    widthPct: 55,
  },
  {
    label: "Fused (1 pass)",
    passes: 1,
    speedup: "1.8x",
    color: THEME.colors.nvidiaGreen,
    widthPct: 100,
  },
];

export const M9S12_FusedBiasGelu: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={9}
      leftWidth="55%"
      left={
        <div style={{ width: 820 }}>
          <SlideTitle title="Example: Fused Bias + GELU" />

          {/* Separate version */}
          <FadeInText
            text="SEPARATE:"
            delay={0.5 * fps}
            fontSize={14}
            fontWeight={700}
            color={THEME.colors.accentRed}
            style={{ marginBottom: 6, letterSpacing: "1px" }}
          />
          <CodeBlock
            delay={0.8 * fps}
            fontSize={14}
            code={`bias_add_kernel<<<...>>>(x, bias, temp, N);
gelu_kernel<<<...>>>(temp, output, N);
// 2 kernels, temp in HBM`}
            showLineNumbers={false}
          />

          {/* Fused version */}
          <FadeInText
            text="FUSED:"
            delay={3 * fps}
            fontSize={14}
            fontWeight={700}
            color={THEME.colors.nvidiaGreen}
            style={{ marginBottom: 6, marginTop: 16, letterSpacing: "1px" }}
          />
          <CodeBlock
            delay={3.3 * fps}
            fontSize={13}
            code={`__global__ void fused_bias_gelu(
    float* x, float* bias, float* out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float val = x[i] + bias[i % hidden_dim];
    out[i] = val * 0.5f * (1.0f + tanhf(
      0.7978845608f * (val + 0.044715f
                        * val*val*val)));
  }
}`}
            highlightLines={[5, 6, 7, 8]}
            showLineNumbers={false}
          />
        </div>
      }
      right={
        <div style={{ width: 500, marginTop: 80 }}>
          <FadeInText
            text="Performance Comparison"
            delay={6 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 24 }}
          />

          {/* Performance bars */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {perfBars.map((bar, i) => {
              const barDelay = 6.5 * fps + i * 1.2 * fps;
              const barProgress = interpolate(
                frame - barDelay,
                [0, 1 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              const labelSpring = spring({
                frame: frame - barDelay,
                fps,
                config: { damping: 200 },
              });
              const labelOpacity = interpolate(labelSpring, [0, 1], [0, 1]);

              return (
                <div key={bar.label} style={{ opacity: labelOpacity }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      marginBottom: 8,
                      width: 440,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 15,
                        color: THEME.colors.textPrimary,
                        fontFamily: fontFamilyBody,
                        fontWeight: 500,
                      }}
                    >
                      {bar.label}
                    </span>
                    <span
                      style={{
                        fontSize: 15,
                        color: bar.color,
                        fontFamily: fontFamilyCode,
                        fontWeight: 700,
                      }}
                    >
                      {bar.speedup}
                    </span>
                  </div>
                  <div
                    style={{
                      width: 440,
                      height: 32,
                      backgroundColor: "rgba(255,255,255,0.04)",
                      borderRadius: 6,
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: (bar.widthPct / 100) * 440 * barProgress,
                        backgroundColor: `${bar.color}CC`,
                        borderRadius: 6,
                        boxShadow: `0 0 12px ${bar.color}30`,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Bottom callout */}
          <div
            style={{
              marginTop: 32,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}30`,
              width: 440,
              opacity: interpolate(
                frame - 10 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div
              style={{
                fontSize: 16,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              Same output, half the memory traffic
            </div>
          </div>
        </div>
      }
    />
  );
};
