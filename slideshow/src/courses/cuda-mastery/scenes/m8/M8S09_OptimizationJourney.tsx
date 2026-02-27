import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

interface OptStep {
  label: string;
  gflops: number;
  color: string;
  peakLabel: string;
  delay: number;
}

const PEAK_GFLOPS = 19500;
const BAR_MAX_WIDTH = 1100;
const BAR_HEIGHT = 52;
const BAR_GAP = 16;

const STEPS: OptStep[] = [
  {
    label: "Naive",
    gflops: 50,
    color: THEME.colors.accentRed,
    peakLabel: "0.3% peak",
    delay: 0,
  },
  {
    label: "Tiled (32x32)",
    gflops: 500,
    color: THEME.colors.accentOrange,
    peakLabel: "3% peak",
    delay: 1.2,
  },
  {
    label: "+ Register Tiling (8x8)",
    gflops: 5000,
    color: THEME.colors.accentYellow,
    peakLabel: "30% peak",
    delay: 2.4,
  },
  {
    label: "+ float4 + Double Buffer",
    gflops: 14000,
    color: THEME.colors.nvidiaGreen,
    peakLabel: "85% peak",
    delay: 3.6,
  },
  {
    label: "cuBLAS",
    gflops: 16500,
    color: THEME.colors.nvidiaGreenLight,
    peakLabel: "95% peak",
    delay: 4.8,
  },
];

export const M8S09_OptimizationJourney: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const bottomOpacity = interpolate(
    frame - 11 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Peak line
  const peakLineOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={8}>
      <SlideTitle
        title="The Optimization Journey"
        subtitle="280x speedup from naive to optimized"
      />

      <div style={{ marginTop: 12, position: "relative", width: 1776 }}>
        {/* Bars */}
        {STEPS.map((step, i) => {
          const barDelay = (1.5 + step.delay) * fps;
          const barSpring = spring({
            frame: frame - barDelay,
            fps,
            config: { damping: 200, stiffness: 80 },
          });

          const barWidth = interpolate(
            barSpring,
            [0, 1],
            [0, Math.max((step.gflops / PEAK_GFLOPS) * BAR_MAX_WIDTH, 30)]
          );
          const labelOpacity = interpolate(barSpring, [0, 1], [0, 1]);

          return (
            <div
              key={i}
              style={{
                display: "flex",
                alignItems: "center",
                marginBottom: BAR_GAP,
                gap: 16,
                width: 1400,
              }}
            >
              {/* Label */}
              <div
                style={{
                  width: 260,
                  textAlign: "right",
                  fontSize: 17,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  fontWeight: 600,
                  opacity: labelOpacity,
                  flexShrink: 0,
                }}
              >
                {step.label}
              </div>

              {/* Bar */}
              <div
                style={{
                  width: barWidth,
                  height: BAR_HEIGHT,
                  backgroundColor: step.color,
                  borderRadius: 6,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "flex-end",
                  paddingRight: barWidth > 60 ? 12 : 0,
                  minWidth: 30,
                  position: "relative",
                }}
              >
                {barWidth > 100 && (
                  <span
                    style={{
                      fontSize: 15,
                      fontWeight: 800,
                      color: "#000",
                      fontFamily: fontFamilyCode,
                      opacity: labelOpacity,
                    }}
                  >
                    {step.gflops >= 1000
                      ? `${(step.gflops / 1000).toFixed(step.gflops % 1000 === 0 ? 0 : 1)}K`
                      : step.gflops}
                  </span>
                )}
              </div>

              {/* GFLOPS + peak label */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  opacity: labelOpacity,
                  flexShrink: 0,
                }}
              >
                {barWidth <= 100 && (
                  <span
                    style={{
                      fontSize: 15,
                      fontWeight: 700,
                      color: step.color,
                      fontFamily: fontFamilyCode,
                    }}
                  >
                    {step.gflops >= 1000
                      ? `${(step.gflops / 1000).toFixed(step.gflops % 1000 === 0 ? 0 : 1)}K`
                      : step.gflops}{" "}
                    GFLOPS
                  </span>
                )}
                {barWidth > 100 && (
                  <span
                    style={{
                      fontSize: 15,
                      fontWeight: 700,
                      color: step.color,
                      fontFamily: fontFamilyCode,
                    }}
                  >
                    {step.gflops >= 1000
                      ? `${(step.gflops / 1000).toFixed(step.gflops % 1000 === 0 ? 0 : 1)}K`
                      : step.gflops}{" "}
                    GFLOPS
                  </span>
                )}
                <span
                  style={{
                    fontSize: 13,
                    color: THEME.colors.textMuted,
                    fontFamily: fontFamilyBody,
                    fontWeight: 600,
                  }}
                >
                  {step.peakLabel}
                </span>
              </div>
            </div>
          );
        })}

        {/* Peak (A100) dashed line */}
        <div
          style={{
            position: "absolute",
            left: 260 + 16,
            top: 0,
            width: 0,
            height: STEPS.length * (BAR_HEIGHT + BAR_GAP),
            borderRight: `2px dashed ${THEME.colors.textMuted}`,
            marginLeft: BAR_MAX_WIDTH,
            opacity: peakLineOpacity,
          }}
        >
          <div
            style={{
              position: "absolute",
              top: -24,
              left: -50,
              width: 120,
              textAlign: "center",
              fontSize: 13,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyCode,
              fontWeight: 700,
            }}
          >
            Peak ~19.5K GFLOPS
          </div>
          <div
            style={{
              position: "absolute",
              top: -10,
              left: -40,
              width: 100,
              textAlign: "center",
              fontSize: 11,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
            }}
          >
            (A100 FP32)
          </div>
        </div>
      </div>

      {/* Bottom insight */}
      <div
        style={{
          marginTop: 24,
          padding: "14px 28px",
          backgroundColor: "rgba(118,185,0,0.08)",
          borderRadius: 10,
          border: `1px solid ${THEME.colors.nvidiaGreen}40`,
          opacity: bottomOpacity,
          alignSelf: "center",
        }}
      >
        <span
          style={{
            fontSize: 19,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyBody,
            fontWeight: 700,
          }}
        >
          280x speedup from naive to optimized. Each step targeted a specific bottleneck.
        </span>
      </div>
    </SlideLayout>
  );
};
