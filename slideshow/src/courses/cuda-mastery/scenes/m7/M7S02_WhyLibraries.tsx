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

interface PerfBar {
  label: string;
  gflops: number;
  maxGflops: number;
  color: string;
  displayLabel: string;
  delay: number;
}

const PERF_BARS: PerfBar[] = [
  {
    label: "Your kernel",
    gflops: 200,
    maxGflops: 19000,
    color: THEME.colors.accentRed,
    displayLabel: "~200 GFLOPS",
    delay: 0,
  },
  {
    label: "Optimized kernel",
    gflops: 800,
    maxGflops: 19000,
    color: THEME.colors.accentOrange,
    displayLabel: "~800 GFLOPS",
    delay: 0.5,
  },
  {
    label: "cuBLAS",
    gflops: 19000,
    maxGflops: 19000,
    color: THEME.colors.nvidiaGreen,
    displayLabel: "~19,000 GFLOPS",
    delay: 1.0,
  },
];

const BAR_MAX_WIDTH = 460;
const BAR_HEIGHT = 38;

export const M7S02_WhyLibraries: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const insightOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={7}
      leftWidth="55%"
      left={
        <div style={{ width: 620 }}>
          <SlideTitle
            title="Why Use NVIDIA Libraries?"
            subtitle="Performance you can't match by hand"
          />

          <div style={{ marginTop: 12, width: 600 }}>
            {PERF_BARS.map((bar, i) => {
              const barDelay = (1.5 + bar.delay) * fps;
              const barSpring = spring({
                frame: frame - barDelay,
                fps,
                config: { damping: 200, stiffness: 80 },
              });

              const barWidth = interpolate(
                barSpring,
                [0, 1],
                [0, Math.max((bar.gflops / bar.maxGflops) * BAR_MAX_WIDTH, 24)]
              );
              const labelOpacity = interpolate(barSpring, [0, 1], [0, 1]);

              return (
                <div key={`bar-${i}`} style={{ marginBottom: 20 }}>
                  <div
                    style={{
                      fontSize: 16,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyBody,
                      fontWeight: 600,
                      marginBottom: 6,
                      opacity: labelOpacity,
                    }}
                  >
                    {bar.label}
                  </div>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 12,
                      width: 580,
                    }}
                  >
                    <div
                      style={{
                        width: barWidth,
                        height: BAR_HEIGHT,
                        backgroundColor: bar.color,
                        borderRadius: 6,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "flex-end",
                        paddingRight: 10,
                        minWidth: 24,
                      }}
                    />
                    <span
                      style={{
                        fontSize: 15,
                        fontFamily: fontFamilyCode,
                        fontWeight: 700,
                        color: bar.color,
                        opacity: labelOpacity,
                        whiteSpace: "nowrap",
                      }}
                    >
                      {bar.displayLabel}
                    </span>
                  </div>
                </div>
              );
            })}

            {/* A100 label */}
            <div
              style={{
                fontSize: 12,
                color: THEME.colors.textMuted,
                fontFamily: fontFamilyBody,
                marginTop: 4,
                opacity: interpolate(
                  frame - 4 * fps,
                  [0, 0.5 * fps],
                  [0, 0.7],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              SGEMM on A100 (FP32)
            </div>
          </div>

          {/* Bottom insight */}
          <div
            style={{
              marginTop: 28,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: insightOpacity,
              width: 580,
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
              cuBLAS GEMM achieves{" "}
              <span style={{ color: THEME.colors.nvidiaGreen }}>
                95%+ of theoretical peak
              </span>{" "}
              on modern GPUs.
            </span>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 60, width: 420 }}>
          <BulletPoint
            index={0}
            delay={3 * fps}
            text="Years of optimization by NVIDIA engineers"
            icon="1"
          />
          <BulletPoint
            index={1}
            delay={3 * fps}
            text="Tensor Core utilization built-in"
            icon="2"
            highlight
          />
          <BulletPoint
            index={2}
            delay={3 * fps}
            text="Auto-tuned for each GPU architecture"
            icon="3"
          />
          <BulletPoint
            index={3}
            delay={3 * fps}
            text="Focus on your algorithm, not low-level optimization"
            icon="4"
            highlight
          />
        </div>
      }
    />
  );
};
