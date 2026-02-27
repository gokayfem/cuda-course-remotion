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

export const M5S02_WhyOptimize: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const bars = [
    {
      label: "Theoretical Peak",
      value: 100,
      color: THEME.colors.nvidiaGreen,
      delay: 1 * fps,
    },
    {
      label: "Typical Naive Kernel",
      value: 15,
      color: THEME.colors.accentRed,
      delay: 2.5 * fps,
    },
    {
      label: "Optimized Kernel",
      value: 78,
      color: THEME.colors.accentBlue,
      delay: 4 * fps,
    },
  ];

  const BAR_MAX_WIDTH = 360;
  const BAR_HEIGHT = 42;

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={5}
      leftWidth="52%"
      left={
        <div>
          <SlideTitle
            title="The GPU Utilization Gap"
            subtitle="Most kernels waste massive amounts of GPU potential"
          />

          {/* Bar chart */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 24,
              marginTop: 20,
              width: 500,
            }}
          >
            {bars.map((bar, i) => {
              const barSpring = spring({
                frame: frame - bar.delay,
                fps,
                config: { damping: 100, stiffness: 80 },
              });
              const barWidth = interpolate(barSpring, [0, 1], [0, (bar.value / 100) * BAR_MAX_WIDTH]);
              const barOpacity = interpolate(barSpring, [0, 1], [0, 1]);

              const percentOpacity = interpolate(
                frame - bar.delay - 0.3 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              return (
                <div
                  key={bar.label}
                  style={{
                    opacity: barOpacity,
                    width: 500,
                  }}
                >
                  <div
                    style={{
                      fontSize: 16,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyBody,
                      marginBottom: 6,
                      fontWeight: 600,
                    }}
                  >
                    {bar.label}
                  </div>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 14,
                    }}
                  >
                    <div
                      style={{
                        width: BAR_MAX_WIDTH,
                        height: BAR_HEIGHT,
                        backgroundColor: "rgba(255,255,255,0.05)",
                        borderRadius: 6,
                        overflow: "hidden",
                      }}
                    >
                      <div
                        style={{
                          width: barWidth,
                          height: BAR_HEIGHT,
                          backgroundColor: `${bar.color}cc`,
                          borderRadius: 6,
                          background: `linear-gradient(90deg, ${bar.color}90, ${bar.color})`,
                        }}
                      />
                    </div>
                    <span
                      style={{
                        fontSize: 22,
                        fontWeight: 700,
                        color: bar.color,
                        fontFamily: fontFamilyCode,
                        opacity: percentOpacity,
                        minWidth: 60,
                      }}
                    >
                      {bar.value}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Gap arrow annotation */}
          <div
            style={{
              marginTop: 20,
              display: "flex",
              alignItems: "center",
              gap: 10,
              opacity: interpolate(
                frame - 5.5 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div
              style={{
                fontSize: 28,
                color: THEME.colors.accentOrange,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
              }}
            >
              5x
            </div>
            <div
              style={{
                fontSize: 16,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
              }}
            >
              speedup potential from naive to optimized
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 60, width: 420 }}>
          <BulletPoint
            index={0}
            delay={2 * fps}
            text="Most kernels leave 80%+ performance on the table"
            icon="1"
          />
          <BulletPoint
            index={1}
            delay={2 * fps}
            text="GPU has massive parallelism — but only if you use it right"
            icon="2"
          />
          <BulletPoint
            index={2}
            delay={2 * fps}
            text="Three bottlenecks: compute, memory, latency"
            icon="3"
            highlight
          />
          <BulletPoint
            index={3}
            delay={2 * fps}
            text="This module: systematic optimization techniques"
            icon="4"
          />

          {/* Insight box */}
          <div
            style={{
              marginTop: 28,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: interpolate(
                frame - 7 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
              width: 400,
            }}
          >
            <span
              style={{
                fontSize: 17,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              Optimization is not guessing — it is{" "}
              <span style={{ color: THEME.colors.nvidiaGreen }}>measuring</span>{" "}
              and removing bottlenecks systematically.
            </span>
          </div>
        </div>
      }
    />
  );
};
