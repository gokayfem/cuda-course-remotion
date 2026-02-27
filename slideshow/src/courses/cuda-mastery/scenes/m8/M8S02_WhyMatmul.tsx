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

interface PieSlice {
  label: string;
  percent: number;
  color: string;
  delay: number;
}

const SLICES: PieSlice[] = [
  { label: "GEMM", percent: 90, color: THEME.colors.nvidiaGreen, delay: 0 },
  { label: "Activation", percent: 3, color: THEME.colors.accentBlue, delay: 0.6 },
  { label: "Normalization", percent: 3, color: THEME.colors.accentPurple, delay: 0.9 },
  { label: "Other", percent: 4, color: THEME.colors.textMuted, delay: 1.2 },
];

const PIE_CX = 170;
const PIE_CY = 170;
const PIE_R = 130;

const describeSliceArc = (
  cx: number,
  cy: number,
  r: number,
  startDeg: number,
  endDeg: number
): string => {
  const startRad = ((startDeg - 90) * Math.PI) / 180;
  const endRad = ((endDeg - 90) * Math.PI) / 180;
  const x1 = cx + Math.cos(startRad) * r;
  const y1 = cy + Math.sin(startRad) * r;
  const x2 = cx + Math.cos(endRad) * r;
  const y2 = cy + Math.sin(endRad) * r;
  const largeArc = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2} Z`;
};

export const M8S02_WhyMatmul: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const insightOpacity = interpolate(
    frame - 10 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  let cumulativeDeg = 0;
  const sliceData = SLICES.map((slice) => {
    const startDeg = cumulativeDeg;
    const endDeg = cumulativeDeg + (slice.percent / 100) * 360;
    cumulativeDeg = endDeg;
    return { ...slice, startDeg, endDeg };
  });

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={8}
      leftWidth="50%"
      left={
        <div style={{ width: 520 }}>
          <SlideTitle
            title="Why MatMul is Everything in ML"
            subtitle="Compute breakdown of a typical transformer"
          />

          {/* Pie chart */}
          <div style={{ marginTop: 12, width: 400 }}>
            <svg width={400} height={360} viewBox="0 0 400 360">
              {sliceData.map((slice, i) => {
                const sliceDelay = (1.5 + slice.delay) * fps;
                const sliceProgress = interpolate(
                  frame - sliceDelay,
                  [0, 0.5 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                );

                const sweepEnd =
                  slice.startDeg +
                  (slice.endDeg - slice.startDeg) * sliceProgress;

                if (sliceProgress <= 0) return null;

                const path = describeSliceArc(
                  PIE_CX,
                  PIE_CY,
                  PIE_R,
                  slice.startDeg,
                  sweepEnd
                );

                // Label position at midpoint of slice
                const midDeg = (slice.startDeg + slice.endDeg) / 2;
                const midRad = ((midDeg - 90) * Math.PI) / 180;
                const labelR = PIE_R * 0.6;
                const lx = PIE_CX + Math.cos(midRad) * labelR;
                const ly = PIE_CY + Math.sin(midRad) * labelR;

                const labelOpacity = interpolate(
                  frame - (sliceDelay + 0.3 * fps),
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                );

                return (
                  <React.Fragment key={i}>
                    <path
                      d={path}
                      fill={slice.color}
                      opacity={0.85}
                      stroke="#0a0a0a"
                      strokeWidth={2}
                    />
                    {sliceProgress >= 0.8 && (
                      <text
                        x={lx}
                        y={ly}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fill={THEME.colors.textPrimary}
                        fontSize={slice.percent >= 10 ? 18 : 12}
                        fontWeight={700}
                        fontFamily={fontFamilyBody}
                        opacity={labelOpacity}
                      >
                        {slice.percent}%
                      </text>
                    )}
                  </React.Fragment>
                );
              })}
            </svg>

            {/* Legend */}
            <div
              style={{
                display: "flex",
                gap: 20,
                marginTop: -16,
                flexWrap: "wrap",
                width: 400,
              }}
            >
              {SLICES.map((slice, i) => {
                const legendOpacity = interpolate(
                  frame - (2 + slice.delay) * fps,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                );
                return (
                  <div
                    key={i}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                      opacity: legendOpacity,
                    }}
                  >
                    <div
                      style={{
                        width: 12,
                        height: 12,
                        borderRadius: 3,
                        backgroundColor: slice.color,
                      }}
                    />
                    <span
                      style={{
                        fontSize: 14,
                        color: THEME.colors.textSecondary,
                        fontFamily: fontFamilyBody,
                      }}
                    >
                      {slice.label}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 60, width: 480 }}>
          <BulletPoint
            index={0}
            delay={4 * fps}
            text='Linear layers: Y = XW + b (GEMM)'
            icon="1"
          />
          <BulletPoint
            index={1}
            delay={4 * fps}
            text="Attention: QxK^T, (QK^T)xV (two GEMMs)"
            icon="2"
            highlight
          />
          <BulletPoint
            index={2}
            delay={4 * fps}
            text="Convolution = im2col + GEMM"
            icon="3"
          />
          <BulletPoint
            index={3}
            delay={4 * fps}
            text="90%+ of training FLOPS are matrix multiply"
            icon="4"
            highlight
          />

          {/* Bottom insight */}
          <div
            style={{
              marginTop: 40,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: insightOpacity,
              width: 440,
            }}
          >
            <span
              style={{
                fontSize: 18,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
              }}
            >
              Optimizing GEMM = optimizing everything
            </span>
          </div>
        </div>
      }
    />
  );
};
