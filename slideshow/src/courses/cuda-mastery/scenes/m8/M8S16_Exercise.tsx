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

type ResultRow = {
  label: string;
  gflops: string;
  peak: string;
  color: string;
};

const results: ResultRow[] = [
  {
    label: "Naive",
    gflops: "~50",
    peak: "0.3%",
    color: THEME.colors.accentRed,
  },
  {
    label: "Tiled",
    gflops: "~500",
    peak: "3%",
    color: THEME.colors.accentYellow,
  },
  {
    label: "Reg Tiled",
    gflops: "~5,000",
    peak: "30%",
    color: THEME.colors.accentBlue,
  },
  {
    label: "cuBLAS",
    gflops: "~16,500",
    peak: "95%",
    color: THEME.colors.nvidiaGreen,
  },
];

const REVEAL_TIME = 8;

export const M8S16_Exercise: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const showAnswers = frame > REVEAL_TIME * fps;

  const answerOpacity = interpolate(
    frame - REVEAL_TIME * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const tasks = [
    {
      num: 1,
      text: "Naive (1 element/thread)",
      color: THEME.colors.accentRed,
    },
    {
      num: 2,
      text: "Shared memory tiled (32x32)",
      color: THEME.colors.accentYellow,
    },
    {
      num: 3,
      text: "Register tiled (8x8 per thread)",
      color: THEME.colors.accentBlue,
    },
    {
      num: 4,
      text: "Compare all vs cuBLAS",
      color: THEME.colors.nvidiaGreen,
    },
  ];

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={8}
      leftWidth="50%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle
            title="Exercise: The GEMM Optimization Challenge"
            subtitle="Implement 4 versions of 1024x1024 matmul"
          />

          {/* Task list */}
          <div style={{ marginTop: 12 }}>
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
                  key={i}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 14,
                    padding: "12px 18px",
                    backgroundColor: `${task.color}08`,
                    borderLeft: `3px solid ${task.color}`,
                    borderRadius: 8,
                    marginBottom: 10,
                    opacity: taskOpacity,
                    transform: `translateX(${taskX}px)`,
                    width: 600,
                  }}
                >
                  <div
                    style={{
                      width: 28,
                      height: 28,
                      borderRadius: 14,
                      backgroundColor: `${task.color}20`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 14,
                      fontWeight: 700,
                      color: task.color,
                      fontFamily: fontFamilyBody,
                      flexShrink: 0,
                    }}
                  >
                    {task.num}
                  </div>
                  <span
                    style={{
                      fontSize: 18,
                      color: THEME.colors.textPrimary,
                      fontFamily: fontFamilyBody,
                    }}
                  >
                    {task.text}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Goal */}
          <div
            style={{
              marginTop: 20,
              padding: "14px 20px",
              backgroundColor: "rgba(79,195,247,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.accentBlue}30`,
              width: 600,
              opacity: interpolate(
                frame - 3.5 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div
              style={{
                fontSize: 16,
                color: THEME.colors.accentBlue,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
              }}
            >
              Measure GFLOPS for each implementation
            </div>
            <div
              style={{
                fontSize: 14,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                marginTop: 4,
              }}
            >
              GFLOPS = (2 * N^3) / (time_ms * 1e6)
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ width: 560, marginTop: 20 }}>
          <FadeInText
            text="Measurement Template"
            delay={4 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 20 }}
          />

          {/* Results table */}
          <div
            style={{
              backgroundColor: "rgba(13,17,23,0.7)",
              borderRadius: 10,
              padding: "20px 24px",
              border: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            {/* Header */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                paddingBottom: 10,
                borderBottom: "1px solid rgba(255,255,255,0.08)",
                marginBottom: 14,
              }}
            >
              <span
                style={{
                  fontSize: 14,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyBody,
                  fontWeight: 600,
                  width: 140,
                }}
              >
                Version
              </span>
              <span
                style={{
                  fontSize: 14,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyBody,
                  fontWeight: 600,
                  width: 160,
                  textAlign: "right",
                }}
              >
                GFLOPS
              </span>
              <span
                style={{
                  fontSize: 14,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyBody,
                  fontWeight: 600,
                  width: 100,
                  textAlign: "right",
                }}
              >
                % Peak
              </span>
            </div>

            {/* Rows */}
            {results.map((row, i) => {
              const rowDelay = 5 * fps + i * 0.4 * fps;
              const rowSpring = spring({
                frame: frame - rowDelay,
                fps,
                config: { damping: 200 },
              });
              const rowOpacity = interpolate(rowSpring, [0, 1], [0, 1]);

              return (
                <div
                  key={row.label}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    padding: "10px 0",
                    borderBottom:
                      i < results.length - 1
                        ? "1px solid rgba(255,255,255,0.05)"
                        : "none",
                    opacity: rowOpacity,
                  }}
                >
                  <span
                    style={{
                      fontSize: 17,
                      color: row.color,
                      fontFamily: fontFamilyCode,
                      fontWeight: 700,
                      width: 140,
                    }}
                  >
                    {row.label}:
                  </span>
                  <span
                    style={{
                      fontSize: 17,
                      color: showAnswers
                        ? THEME.colors.textPrimary
                        : THEME.colors.textMuted,
                      fontFamily: fontFamilyCode,
                      width: 160,
                      textAlign: "right",
                      opacity: showAnswers ? answerOpacity : 1,
                    }}
                  >
                    {showAnswers
                      ? `${row.gflops} GFLOPS`
                      : "____ GFLOPS"}
                  </span>
                  <span
                    style={{
                      fontSize: 17,
                      color: showAnswers
                        ? row.color
                        : THEME.colors.textMuted,
                      fontFamily: fontFamilyCode,
                      fontWeight: showAnswers ? 700 : 400,
                      width: 100,
                      textAlign: "right",
                      opacity: showAnswers ? answerOpacity : 1,
                    }}
                  >
                    {showAnswers ? `(${row.peak})` : "(___%)"}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Insight box */}
          {showAnswers && (
            <div
              style={{
                marginTop: 20,
                padding: "14px 20px",
                backgroundColor: "rgba(118,185,0,0.08)",
                borderRadius: 8,
                border: `1px solid ${THEME.colors.nvidiaGreen}30`,
                opacity: answerOpacity,
              }}
            >
              <div
                style={{
                  fontSize: 16,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                  fontWeight: 700,
                }}
              >
                330x speedup from naive to cuBLAS
              </div>
              <div
                style={{
                  fontSize: 14,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  marginTop: 4,
                }}
              >
                Each optimization level compounds: tiling, register blocking, vectorization, Tensor Cores
              </div>
            </div>
          )}
        </div>
      }
    />
  );
};
