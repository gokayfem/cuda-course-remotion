import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

type OptStep = {
  label: string;
  latency: number;
  color: string;
};

const optimizations: OptStep[] = [
  { label: "Serial (1 stream)", latency: 100, color: THEME.colors.accentRed },
  { label: "+ Async H2D", latency: 75, color: THEME.colors.accentOrange },
  {
    label: "+ Overlapped pipeline (4 streams)",
    latency: 40,
    color: THEME.colors.accentYellow,
  },
  { label: "+ CUDA Graphs", latency: 28, color: THEME.colors.nvidiaGreen },
];

const TARGET_LATENCY = 25;
const MAX_LATENCY = 110;
const BAR_MAX_WIDTH = 550;

export const M6S15_CaseStudy: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={6}
      leftWidth="55%"
      left={
        <div style={{ width: 820 }}>
          <SlideTitle title="Case Study: Inference Pipeline Optimization" />

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 18,
            }}
          >
            {optimizations.map((opt, i) => {
              const barDelay = 1 * fps + i * 1.2 * fps;
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

              const barWidth =
                (opt.latency / MAX_LATENCY) * BAR_MAX_WIDTH * barProgress;

              return (
                <div key={opt.label} style={{ opacity: labelOpacity }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "baseline",
                      marginBottom: 6,
                      width: BAR_MAX_WIDTH + 120,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 16,
                        color: THEME.colors.textPrimary,
                        fontFamily: fontFamilyBody,
                        fontWeight: 500,
                      }}
                    >
                      <span
                        style={{
                          color: opt.color,
                          fontWeight: 700,
                          marginRight: 8,
                        }}
                      >
                        {i + 1}.
                      </span>
                      {opt.label}
                    </div>
                    <div
                      style={{
                        fontSize: 16,
                        color: opt.color,
                        fontFamily: fontFamilyCode,
                        fontWeight: 700,
                      }}
                    >
                      {Math.round(opt.latency * barProgress)} ms
                    </div>
                  </div>

                  <div
                    style={{
                      width: BAR_MAX_WIDTH + 120,
                      height: 28,
                      backgroundColor: "rgba(255,255,255,0.04)",
                      borderRadius: 6,
                      overflow: "hidden",
                      position: "relative",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: barWidth,
                        backgroundColor: `${opt.color}CC`,
                        borderRadius: 6,
                        boxShadow: `0 0 12px ${opt.color}30`,
                      }}
                    />
                  </div>
                </div>
              );
            })}

            {/* Target dashed line */}
            <div
              style={{
                opacity: interpolate(
                  frame - 6 * fps,
                  [0, 0.5 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  width: BAR_MAX_WIDTH + 120,
                }}
              >
                <div
                  style={{
                    width:
                      (TARGET_LATENCY / MAX_LATENCY) * BAR_MAX_WIDTH,
                    height: 2,
                    borderTop: `2px dashed ${THEME.colors.textMuted}`,
                    flexShrink: 0,
                  }}
                />
                <span
                  style={{
                    fontSize: 14,
                    color: THEME.colors.textMuted,
                    fontFamily: fontFamilyCode,
                    flexShrink: 0,
                  }}
                >
                  Target: {TARGET_LATENCY} ms
                </span>
              </div>
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ width: 460, marginTop: 80 }}>
          <div
            style={{
              fontSize: 20,
              fontWeight: 700,
              color: THEME.colors.accentCyan,
              fontFamily: fontFamilyBody,
              marginBottom: 20,
              opacity: interpolate(
                frame - 7 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            Results
          </div>

          <BulletPoint
            text="3.6x latency reduction"
            index={0}
            delay={7.5 * fps}
            highlight
          />
          <BulletPoint
            text="Throughput: 1 -> 4 concurrent requests"
            index={1}
            delay={7.5 * fps}
          />
          <BulletPoint
            text="CUDA Graphs eliminated launch jitter"
            index={2}
            delay={7.5 * fps}
          />
          <BulletPoint
            text="Achieved 89% hardware utilization"
            index={3}
            delay={7.5 * fps}
            highlight
          />
        </div>
      }
    />
  );
};
