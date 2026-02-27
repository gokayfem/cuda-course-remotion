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
  isBonus: boolean;
};

const tasks: Task[] = [
  {
    number: 1,
    title: "Implement safe softmax (3-pass)",
    description: "Pass 1: find max. Pass 2: sum exp(x-max). Pass 3: normalize.",
    isBonus: false,
  },
  {
    number: 2,
    title: "Implement fused LayerNorm",
    description: "Use Welford's algorithm for numerically stable mean+variance in one pass.",
    isBonus: false,
  },
  {
    number: 3,
    title: "Implement fused bias + GELU",
    description: "Single kernel: read input, add bias, apply GELU approximation, write output.",
    isBonus: false,
  },
  {
    number: 4,
    title: "Simplified Flash Attention",
    description: "Tiled attention for small seq_len. Use shared memory for Q/K/V tiles.",
    isBonus: true,
  },
];

type PerfHint = {
  task: string;
  gain: string;
  color: string;
};

const perfHints: PerfHint[] = [
  { task: "Safe softmax", gain: "3x vs naive (no overflow)", color: THEME.colors.accentBlue },
  { task: "Fused LayerNorm", gain: "2x vs separate mean+var", color: THEME.colors.nvidiaGreen },
  { task: "Fused bias+GELU", gain: "1.8x vs 2 kernels", color: THEME.colors.accentPurple },
  { task: "Flash Attention", gain: "4x vs materialized NxN", color: THEME.colors.accentOrange },
];

export const M9S16_Exercise: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const showHints = frame > 8 * fps;

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={9}
      leftWidth="50%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle
            title="Exercise: Build Transformer Kernels"
            subtitle="Implement the core building blocks from this module"
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
                  marginBottom: 16,
                  opacity: taskOpacity,
                  transform: `translateX(${taskX}px)`,
                }}
              >
                <div
                  style={{
                    width: 28,
                    height: 28,
                    borderRadius: task.isBonus ? 14 : 4,
                    border: `2px solid ${task.isBonus ? THEME.colors.accentOrange : THEME.colors.accentCyan}60`,
                    flexShrink: 0,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 14,
                    fontWeight: 700,
                    color: task.isBonus ? THEME.colors.accentOrange : THEME.colors.accentCyan,
                    fontFamily: fontFamilyCode,
                    marginTop: 2,
                  }}
                >
                  {task.isBonus ? "\u2605" : task.number}
                </div>
                <div>
                  <div
                    style={{
                      fontSize: 17,
                      color: THEME.colors.textPrimary,
                      fontFamily: fontFamilyBody,
                      fontWeight: 600,
                      lineHeight: 1.4,
                    }}
                  >
                    {task.title}
                    {task.isBonus && (
                      <span
                        style={{
                          fontSize: 12,
                          color: THEME.colors.accentOrange,
                          marginLeft: 8,
                          fontWeight: 700,
                          letterSpacing: "0.5px",
                        }}
                      >
                        BONUS
                      </span>
                    )}
                  </div>
                  <div
                    style={{
                      fontSize: 14,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyBody,
                      marginTop: 4,
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
            text="Skeleton Hints"
            delay={3 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 16, marginTop: 10 }}
          />

          {/* Skeleton hints */}
          <div
            style={{
              padding: "14px 18px",
              backgroundColor: "rgba(13,17,23,0.6)",
              borderRadius: 8,
              border: "1px solid rgba(255,255,255,0.08)",
              marginBottom: 20,
              opacity: interpolate(
                frame - 3.5 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            {[
              "// Each kernel: 1 block per row/vector",
              "// Threads = min(N, 256)",
              "// Use warp shuffles for reductions",
              "// Shared memory for cross-warp results",
              "// Welford: combine (count, mean, M2)",
            ].map((line, i) => (
              <div
                key={i}
                style={{
                  fontSize: 14,
                  color: THEME.colors.syntaxComment,
                  fontFamily: fontFamilyCode,
                  lineHeight: 1.6,
                }}
              >
                {line}
              </div>
            ))}
          </div>

          {/* Performance hints (revealed after 8s) */}
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
                Expected Performance Gains
              </div>

              {perfHints.map((hint, i) => {
                const hintDelay = 8.5 * fps + i * 0.4 * fps;
                const hintOpacity = interpolate(
                  frame - hintDelay,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                );

                return (
                  <div
                    key={hint.task}
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
                        fontFamily: fontFamilyBody,
                      }}
                    >
                      {hint.task}
                    </span>
                    <span
                      style={{
                        fontSize: 14,
                        fontWeight: 700,
                        color: hint.color,
                        fontFamily: fontFamilyCode,
                      }}
                    >
                      {hint.gain}
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
