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

const checklistItems = [
  "Experiment with block sizes (128, 256, 512)",
  "Add float4 vectorized loads/stores",
  "Process multiple elements per thread (ILP)",
  "Add #pragma unroll to inner loop",
  "Measure bandwidth at each step",
];

type HintRow = {
  step: string;
  bw: string;
  color: string;
};

const hints: HintRow[] = [
  { step: "blockSize=128", bw: "~280 GB/s", color: THEME.colors.accentRed },
  { step: "blockSize=256", bw: "~310 GB/s", color: THEME.colors.accentOrange },
  { step: "+ float4 loads", bw: "~440 GB/s", color: THEME.colors.accentYellow },
  { step: "+ 4 elem/thread", bw: "~500 GB/s", color: THEME.colors.nvidiaGreen },
  { step: "+ #pragma unroll", bw: "~520 GB/s", color: THEME.colors.nvidiaGreen },
];

export const M5S16_Exercise: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const showHints = frame > 8 * fps;

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={5}
      leftWidth="50%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle
            title="Exercise: Optimize a SAXPY Kernel"
            subtitle="Apply everything you've learned to hit peak bandwidth"
          />

          <CodeBlock
            delay={0.8 * fps}
            title="saxpy_naive.cu"
            fontSize={16}
            code={`__global__ void saxpy(float a, float* x,
                       float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = a * x[i] + y[i];
}`}
            highlightLines={[3, 4]}
          />

          {/* Target box */}
          <div
            style={{
              marginTop: 20,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.nvidiaGreen}30`,
              opacity: interpolate(
                frame - 4 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div style={{ fontSize: 16, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyBody, fontWeight: 700 }}>
              Target: Achieve {">"}80% of peak bandwidth
            </div>
            <div style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, marginTop: 4 }}>
              SAXPY reads 2 floats, writes 1 float = 12 bytes, does 2 FLOPs. AI = 0.17 (memory-bound)
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ width: 560 }}>
          <FadeInText
            text="Optimization Checklist"
            delay={1.5 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 16, marginTop: 10 }}
          />

          {/* Checklist */}
          {checklistItems.map((item, i) => {
            const itemDelay = 2 * fps + i * 0.4 * fps;
            const itemSpring = spring({
              frame: frame - itemDelay,
              fps,
              config: { damping: 200 },
            });
            const itemOpacity = interpolate(itemSpring, [0, 1], [0, 1]);
            const itemX = interpolate(itemSpring, [0, 1], [-15, 0]);

            return (
              <div
                key={i}
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 12,
                  marginBottom: 12,
                  opacity: itemOpacity,
                  transform: `translateX(${itemX}px)`,
                }}
              >
                <div
                  style={{
                    width: 22,
                    height: 22,
                    borderRadius: 4,
                    border: `2px solid ${THEME.colors.accentCyan}60`,
                    flexShrink: 0,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 14,
                    color: THEME.colors.accentCyan,
                    marginTop: 2,
                  }}
                >
                  {i + 1}
                </div>
                <span
                  style={{
                    fontSize: 17,
                    color: THEME.colors.textPrimary,
                    fontFamily: fontFamilyBody,
                    lineHeight: 1.5,
                  }}
                >
                  {item}
                </span>
              </div>
            );
          })}

          {/* Solution hints */}
          {showHints && (
            <div
              style={{
                marginTop: 24,
                padding: "16px 20px",
                backgroundColor: "rgba(255,255,255,0.03)",
                borderRadius: 10,
                border: `1px solid ${THEME.colors.accentPurple}30`,
                opacity: interpolate(
                  frame - 8 * fps,
                  [0, 0.5 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <div
                style={{
                  fontSize: 15,
                  fontWeight: 700,
                  color: THEME.colors.accentPurple,
                  fontFamily: fontFamilyBody,
                  marginBottom: 12,
                }}
              >
                Solution Hints -- Achieved Bandwidth
              </div>

              {hints.map((hint, i) => {
                const hintDelay = 8.5 * fps + i * 0.4 * fps;
                const hintOpacity = interpolate(
                  frame - hintDelay,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                );

                return (
                  <div
                    key={hint.step}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "6px 12px",
                      marginBottom: 4,
                      borderRadius: 6,
                      backgroundColor: `${hint.color}08`,
                      opacity: hintOpacity,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 14,
                        color: THEME.colors.textPrimary,
                        fontFamily: fontFamilyCode,
                      }}
                    >
                      {hint.step}
                    </span>
                    <span
                      style={{
                        fontSize: 14,
                        fontWeight: 700,
                        color: hint.color,
                        fontFamily: fontFamilyCode,
                      }}
                    >
                      {hint.bw}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      }
    />
  );
};
