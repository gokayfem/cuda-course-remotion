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
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

type Task = {
  number: number;
  title: string;
  description: string;
};

const tasks: Task[] = [
  {
    number: 1,
    title: "Use cuBLAS for Q/K/V and FFN projections",
    description: "Wrap cublasGemmEx for all linear layers with FP16 compute.",
  },
  {
    number: 2,
    title: "Write a fused RMSNorm kernel",
    description: "Single kernel: compute RMS, normalize, scale. Target >500 GB/s.",
  },
  {
    number: 3,
    title: "Implement a fused bias+SiLU+gate kernel",
    description: "Combine bias add, SiLU activation, and gating in one pass.",
  },
  {
    number: 4,
    title: "Use Flash Attention (or simplified version)",
    description: "Tiled attention with online softmax. Use shared memory for Q/K tiles.",
  },
  {
    number: 5,
    title: "Profile with Nsight Compute",
    description: "Identify bottlenecks: memory throughput, occupancy, warp stalls.",
  },
];

type Metric = {
  label: string;
  target: string;
  color: string;
};

const metrics: Metric[] = [
  {
    label: "RMSNorm",
    target: ">500 GB/s effective bandwidth",
    color: THEME.colors.nvidiaGreen,
  },
  {
    label: "Fused SiLU",
    target: ">1.5x vs separate kernels",
    color: THEME.colors.accentBlue,
  },
  {
    label: "Total",
    target: "Within 2x of PyTorch compiled",
    color: THEME.colors.accentPurple,
  },
];

type Hint = {
  component: string;
  hint: string;
};

const hints: Hint[] = [
  { component: "RMSNorm", hint: "Use warp shuffles for reduction, vectorized loads" },
  { component: "SiLU+Gate", hint: "SiLU(x) = x * sigmoid(x), fuse with gate multiply" },
  { component: "cuBLAS", hint: "Use CUBLAS_COMPUTE_16F for Tensor Core acceleration" },
  { component: "Attention", hint: "Start with small seq_len, use __syncthreads for tiles" },
];

export const M10S16_Exercise: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const showHints = frame > 8 * fps;

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={10}
      leftWidth="50%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle
            title="Final Exercise: End-to-End Optimization"
            subtitle="Build an optimized 2-layer transformer block"
          />

          {/* Task list */}
          {tasks.map((task, i) => {
            const taskDelay = 1 * fps + i * 0.5 * fps;
            const taskSpring = spring({
              frame: frame - taskDelay,
              fps,
              config: { damping: 200 },
            });
            const taskOpacity = interpolate(taskSpring, [0, 1], [0, 1]);
            const taskX = interpolate(taskSpring, [0, 1], [-15, 0]);

            return (
              <div
                key={task.number}
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 14,
                  marginBottom: 14,
                  opacity: taskOpacity,
                  transform: `translateX(${taskX}px)`,
                }}
              >
                <div
                  style={{
                    width: 28,
                    height: 28,
                    borderRadius: 4,
                    border: `2px solid ${THEME.colors.accentCyan}60`,
                    flexShrink: 0,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 14,
                    fontWeight: 700,
                    color: THEME.colors.accentCyan,
                    fontFamily: fontFamilyCode,
                    marginTop: 2,
                  }}
                >
                  {task.number}
                </div>
                <div>
                  <div
                    style={{
                      fontSize: 16,
                      color: THEME.colors.textPrimary,
                      fontFamily: fontFamilyBody,
                      fontWeight: 600,
                      lineHeight: 1.4,
                    }}
                  >
                    {task.title}
                  </div>
                  <div
                    style={{
                      fontSize: 13,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyBody,
                      marginTop: 3,
                      lineHeight: 1.4,
                    }}
                  >
                    {task.description}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      }
      right={
        <div style={{ width: 560 }}>
          <FadeInText
            text="Target Metrics"
            delay={3.5 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 16, marginTop: 10 }}
          />

          {/* Metrics */}
          <div
            style={{
              padding: "16px 20px",
              backgroundColor: "rgba(255,255,255,0.03)",
              borderRadius: 10,
              border: "1px solid rgba(255,255,255,0.08)",
              marginBottom: 20,
              opacity: interpolate(
                frame - 4 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            {metrics.map((metric, i) => (
              <div
                key={metric.label}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "8px 12px",
                  marginBottom: i < metrics.length - 1 ? 8 : 0,
                  borderRadius: 6,
                  backgroundColor: `${metric.color}08`,
                }}
              >
                <span
                  style={{
                    fontSize: 15,
                    fontWeight: 700,
                    color: metric.color,
                    fontFamily: fontFamilyBody,
                  }}
                >
                  {metric.label}
                </span>
                <span
                  style={{
                    fontSize: 14,
                    color: THEME.colors.textSecondary,
                    fontFamily: fontFamilyCode,
                  }}
                >
                  {metric.target}
                </span>
              </div>
            ))}
          </div>

          {/* Hints (revealed after 8s) */}
          {showHints && (
            <div
              style={{
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
                Component Hints
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
                    key={hint.component}
                    style={{
                      display: "flex",
                      alignItems: "flex-start",
                      gap: 10,
                      marginBottom: 8,
                      opacity: hintOpacity,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 13,
                        fontWeight: 700,
                        color: THEME.colors.accentPurple,
                        fontFamily: fontFamilyCode,
                        flexShrink: 0,
                        width: 90,
                      }}
                    >
                      {hint.component}
                    </span>
                    <span
                      style={{
                        fontSize: 13,
                        color: THEME.colors.textSecondary,
                        fontFamily: fontFamilyBody,
                        lineHeight: 1.4,
                      }}
                    >
                      {hint.hint}
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
