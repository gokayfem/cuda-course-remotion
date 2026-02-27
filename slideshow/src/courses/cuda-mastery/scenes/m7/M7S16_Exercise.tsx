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
  gflops: number;
  peak: string;
  color: string;
};

const results: ResultRow[] = [
  { label: "Naive", gflops: 50, peak: "0.3%", color: THEME.colors.accentRed },
  { label: "Tiled", gflops: 500, peak: "3%", color: THEME.colors.accentYellow },
  { label: "cuBLAS", gflops: 15000, peak: "95%", color: THEME.colors.nvidiaGreen },
];

const REVEAL_TIME = 8; // seconds before answer reveal

export const M7S16_Exercise: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const showAnswers = frame > REVEAL_TIME * fps;

  const answerOpacity = interpolate(
    frame - REVEAL_TIME * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={7}
      leftWidth="50%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle
            title="Exercise: GEMM Showdown"
            subtitle="Compare 3 implementations of matrix multiply (1024 x 1024)"
          />

          {/* Task list */}
          <div style={{ marginTop: 12 }}>
            {[
              { num: 1, text: "Naive kernel (3 nested loops)", color: THEME.colors.accentRed },
              { num: 2, text: "Shared memory tiled kernel", color: THEME.colors.accentYellow },
              { num: 3, text: "cuBLAS cublasSgemm", color: THEME.colors.nvidiaGreen },
            ].map((task, i) => {
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
                frame - 3 * fps,
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
            text="Expected Results"
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
              border: `1px solid rgba(255,255,255,0.08)`,
            }}
          >
            {/* Header */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                paddingBottom: 10,
                borderBottom: `1px solid rgba(255,255,255,0.08)`,
                marginBottom: 14,
              }}
            >
              <span
                style={{
                  fontSize: 14,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyBody,
                  fontWeight: 600,
                  width: 120,
                }}
              >
                Method
              </span>
              <span
                style={{
                  fontSize: 14,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyBody,
                  fontWeight: 600,
                  width: 140,
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
                      width: 120,
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
                      width: 140,
                      textAlign: "right",
                    }}
                  >
                    {showAnswers
                      ? `~${row.gflops.toLocaleString()} GFLOPS`
                      : "~____ GFLOPS"}
                  </span>
                  <span
                    style={{
                      fontSize: 17,
                      color: showAnswers ? row.color : THEME.colors.textMuted,
                      fontFamily: fontFamilyCode,
                      fontWeight: showAnswers ? 700 : 400,
                      width: 100,
                      textAlign: "right",
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
                cuBLAS: 300x faster than naive, 30x faster than tiled
              </div>
              <div
                style={{
                  fontSize: 14,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  marginTop: 4,
                }}
              >
                Hand-optimized assembly + Tensor Cores = near-peak performance
              </div>
            </div>
          )}
        </div>
      }
    />
  );
};
