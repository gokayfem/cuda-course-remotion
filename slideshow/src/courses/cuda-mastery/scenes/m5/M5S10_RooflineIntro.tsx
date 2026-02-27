import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

const CHART = {
  x: 0,
  y: 0,
  width: 700,
  height: 460,
  padLeft: 80,
  padRight: 30,
  padTop: 30,
  padBottom: 60,
} as const;

const LOG_LABELS = [0.25, 0.5, 1, 2, 4, 8, 16];
const PEAK_COMPUTE = 10000; // GFLOPS
const PEAK_BW = 900; // GB/s
const RIDGE_AI = PEAK_COMPUTE / PEAK_BW; // ~11.1

const logScale = (val: number, min: number, max: number) => {
  const logMin = Math.log2(min);
  const logMax = Math.log2(max);
  return (Math.log2(val) - logMin) / (logMax - logMin);
};

const toChartX = (ai: number) => {
  const frac = logScale(ai, 0.125, 32);
  return CHART.padLeft + frac * (CHART.width - CHART.padLeft - CHART.padRight);
};

const toChartY = (gflops: number) => {
  const logMin = Math.log2(50);
  const logMax = Math.log2(PEAK_COMPUTE * 1.5);
  const frac = (Math.log2(Math.max(gflops, 50)) - logMin) / (logMax - logMin);
  return CHART.height - CHART.padBottom - frac * (CHART.height - CHART.padTop - CHART.padBottom);
};

export const M5S10_RooflineIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const bandwidthProgress = interpolate(
    frame - 1.5 * fps,
    [0, 2 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const computeProgress = interpolate(
    frame - 4 * fps,
    [0, 1.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const ridgeSpring = spring({
    frame: frame - 6.5 * fps,
    fps,
    config: { damping: 200 },
  });
  const ridgeOpacity = interpolate(ridgeSpring, [0, 1], [0, 1]);

  const axisOpacity = interpolate(
    frame - 0.5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const labelOpacity = interpolate(
    frame - 8 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Bandwidth line: from left to ridge point
  const bwStartAI = 0.125;
  const bwEndAI = RIDGE_AI;
  const bwPoints: Array<{ x: number; y: number }> = [];
  for (let i = 0; i <= 40; i++) {
    const t = i / 40;
    const ai = bwStartAI * Math.pow(bwEndAI / bwStartAI, t);
    const perf = Math.min(ai * PEAK_BW, PEAK_COMPUTE);
    bwPoints.push({ x: toChartX(ai), y: toChartY(perf) });
  }

  // Compute ceiling: from ridge to right
  const compStartAI = RIDGE_AI;
  const compEndAI = 32;
  const compY = toChartY(PEAK_COMPUTE);

  const ridgeX = toChartX(RIDGE_AI);
  const ridgeY = toChartY(PEAK_COMPUTE);

  const buildPathString = (pts: Array<{ x: number; y: number }>, progress: number) => {
    const count = Math.max(1, Math.floor(pts.length * progress));
    const subset = pts.slice(0, count);
    return subset.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
  };

  const bwPath = buildPathString(bwPoints, bandwidthProgress);

  const computeEndX = toChartX(compStartAI) + (toChartX(compEndAI) - toChartX(compStartAI)) * computeProgress;

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={5}
      leftWidth="60%"
      left={
        <div style={{ width: 740 }}>
          <SlideTitle title="The Roofline Model" />

          <svg
            width={CHART.width}
            height={CHART.height}
            style={{ overflow: "visible" }}
          >
            {/* Axes */}
            <g opacity={axisOpacity}>
              {/* X axis */}
              <line
                x1={CHART.padLeft}
                y1={CHART.height - CHART.padBottom}
                x2={CHART.width - CHART.padRight}
                y2={CHART.height - CHART.padBottom}
                stroke={THEME.colors.textMuted}
                strokeWidth={1.5}
              />
              {/* Y axis */}
              <line
                x1={CHART.padLeft}
                y1={CHART.padTop}
                x2={CHART.padLeft}
                y2={CHART.height - CHART.padBottom}
                stroke={THEME.colors.textMuted}
                strokeWidth={1.5}
              />
              {/* X axis label */}
              <text
                x={(CHART.padLeft + CHART.width - CHART.padRight) / 2}
                y={CHART.height - 10}
                fill={THEME.colors.textSecondary}
                fontSize={14}
                fontFamily={fontFamilyBody}
                textAnchor="middle"
              >
                Arithmetic Intensity (FLOPs/byte)
              </text>
              {/* Y axis label */}
              <text
                x={16}
                y={(CHART.padTop + CHART.height - CHART.padBottom) / 2}
                fill={THEME.colors.textSecondary}
                fontSize={14}
                fontFamily={fontFamilyBody}
                textAnchor="middle"
                transform={`rotate(-90, 16, ${(CHART.padTop + CHART.height - CHART.padBottom) / 2})`}
              >
                Performance (GFLOPS)
              </text>

              {/* X axis tick labels */}
              {LOG_LABELS.map((val) => {
                const cx = toChartX(val);
                return (
                  <g key={val}>
                    <line
                      x1={cx}
                      y1={CHART.height - CHART.padBottom}
                      x2={cx}
                      y2={CHART.height - CHART.padBottom + 6}
                      stroke={THEME.colors.textMuted}
                      strokeWidth={1}
                    />
                    <text
                      x={cx}
                      y={CHART.height - CHART.padBottom + 20}
                      fill={THEME.colors.textSecondary}
                      fontSize={12}
                      fontFamily={fontFamilyCode}
                      textAnchor="middle"
                    >
                      {val}
                    </text>
                    {/* Grid line */}
                    <line
                      x1={cx}
                      y1={CHART.padTop}
                      x2={cx}
                      y2={CHART.height - CHART.padBottom}
                      stroke="rgba(255,255,255,0.04)"
                      strokeWidth={1}
                    />
                  </g>
                );
              })}

              {/* Y axis labels */}
              {[100, 500, 1000, 5000, 10000].map((val) => {
                const cy = toChartY(val);
                return (
                  <g key={val}>
                    <line
                      x1={CHART.padLeft - 6}
                      y1={cy}
                      x2={CHART.padLeft}
                      y2={cy}
                      stroke={THEME.colors.textMuted}
                      strokeWidth={1}
                    />
                    <text
                      x={CHART.padLeft - 10}
                      y={cy + 4}
                      fill={THEME.colors.textSecondary}
                      fontSize={11}
                      fontFamily={fontFamilyCode}
                      textAnchor="end"
                    >
                      {val >= 1000 ? `${val / 1000}K` : val}
                    </text>
                    <line
                      x1={CHART.padLeft}
                      y1={cy}
                      x2={CHART.width - CHART.padRight}
                      y2={cy}
                      stroke="rgba(255,255,255,0.04)"
                      strokeWidth={1}
                    />
                  </g>
                );
              })}
            </g>

            {/* Bandwidth line (diagonal) */}
            {bandwidthProgress > 0 && (
              <path
                d={bwPath}
                fill="none"
                stroke={THEME.colors.accentBlue}
                strokeWidth={3}
                strokeLinecap="round"
              />
            )}

            {/* Bandwidth label */}
            {bandwidthProgress > 0.8 && (
              <text
                x={toChartX(1)}
                y={toChartY(1 * PEAK_BW) - 12}
                fill={THEME.colors.accentBlue}
                fontSize={13}
                fontFamily={fontFamilyBody}
                fontWeight={600}
                opacity={labelOpacity}
                transform={`rotate(-32, ${toChartX(1)}, ${toChartY(1 * PEAK_BW) - 12})`}
              >
                Memory Bandwidth Ceiling
              </text>
            )}

            {/* Compute ceiling line (flat) */}
            {computeProgress > 0 && (
              <line
                x1={toChartX(compStartAI)}
                y1={compY}
                x2={computeEndX}
                y2={compY}
                stroke={THEME.colors.nvidiaGreen}
                strokeWidth={3}
                strokeLinecap="round"
              />
            )}

            {/* Compute label */}
            {computeProgress > 0.5 && (
              <text
                x={(toChartX(compStartAI) + toChartX(compEndAI)) / 2}
                y={compY - 12}
                fill={THEME.colors.nvidiaGreen}
                fontSize={13}
                fontFamily={fontFamilyBody}
                fontWeight={600}
                textAnchor="middle"
                opacity={labelOpacity}
              >
                Peak Compute Ceiling
              </text>
            )}

            {/* Ridge point */}
            <circle
              cx={ridgeX}
              cy={ridgeY}
              r={8 * ridgeOpacity}
              fill={THEME.colors.accentOrange}
              stroke={THEME.colors.textPrimary}
              strokeWidth={2}
              opacity={ridgeOpacity}
            />
            {ridgeOpacity > 0.5 && (
              <>
                <text
                  x={ridgeX}
                  y={ridgeY - 18}
                  fill={THEME.colors.accentOrange}
                  fontSize={14}
                  fontFamily={fontFamilyBody}
                  fontWeight={700}
                  textAnchor="middle"
                  opacity={ridgeOpacity}
                >
                  Ridge Point
                </text>
                {/* Vertical dashed line from ridge */}
                <line
                  x1={ridgeX}
                  y1={ridgeY}
                  x2={ridgeX}
                  y2={CHART.height - CHART.padBottom}
                  stroke={THEME.colors.accentOrange}
                  strokeWidth={1}
                  strokeDasharray="4,4"
                  opacity={ridgeOpacity * 0.5}
                />
              </>
            )}

            {/* Region labels */}
            {ridgeOpacity > 0.7 && (
              <>
                <text
                  x={toChartX(0.6)}
                  y={CHART.height - CHART.padBottom - 16}
                  fill={THEME.colors.accentRed}
                  fontSize={13}
                  fontFamily={fontFamilyBody}
                  fontWeight={700}
                  textAnchor="middle"
                  opacity={ridgeOpacity}
                >
                  MEMORY BOUND
                </text>
                <text
                  x={toChartX(20)}
                  y={CHART.height - CHART.padBottom - 16}
                  fill={THEME.colors.nvidiaGreen}
                  fontSize={13}
                  fontFamily={fontFamilyBody}
                  fontWeight={700}
                  textAnchor="middle"
                  opacity={ridgeOpacity}
                >
                  COMPUTE BOUND
                </text>
              </>
            )}
          </svg>
        </div>
      }
      right={
        <div style={{ width: 460 }}>
          <div style={{ marginTop: 80 }}>
            {/* Formula box */}
            <div
              style={{
                padding: "24px 28px",
                backgroundColor: "rgba(255,255,255,0.04)",
                borderRadius: 12,
                border: `1px solid ${THEME.colors.accentPurple}40`,
                marginBottom: 24,
                opacity: interpolate(
                  frame - 2 * fps,
                  [0, 0.5 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <div
                style={{
                  fontSize: 16,
                  fontWeight: 700,
                  color: THEME.colors.accentPurple,
                  fontFamily: fontFamilyBody,
                  marginBottom: 14,
                }}
              >
                Key Formula
              </div>
              <div
                style={{
                  fontSize: 22,
                  color: THEME.colors.textPrimary,
                  fontFamily: fontFamilyCode,
                  textAlign: "center",
                  padding: "12px 0",
                  lineHeight: 1.6,
                }}
              >
                <span style={{ color: THEME.colors.accentOrange }}>AI</span> ={" "}
                <span style={{ color: THEME.colors.nvidiaGreen }}>FLOPs</span> /{" "}
                <span style={{ color: THEME.colors.accentBlue }}>Bytes Accessed</span>
              </div>
            </div>

            {/* Decision rules */}
            <div
              style={{
                opacity: interpolate(
                  frame - 7 * fps,
                  [0, 0.5 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <div
                style={{
                  padding: "16px 20px",
                  backgroundColor: "rgba(79,195,247,0.08)",
                  borderLeft: `4px solid ${THEME.colors.accentBlue}`,
                  borderRadius: 8,
                  marginBottom: 12,
                }}
              >
                <div style={{ fontSize: 16, color: THEME.colors.textPrimary, fontFamily: fontFamilyCode, lineHeight: 1.5 }}>
                  If{" "}
                  <span style={{ color: THEME.colors.accentOrange, fontWeight: 700 }}>AI</span>{" "}
                  {"<"} ridge
                </div>
                <div style={{ fontSize: 15, color: THEME.colors.accentBlue, fontFamily: fontFamilyBody, marginTop: 4 }}>
                  Memory bound -- optimize bandwidth
                </div>
              </div>

              <div
                style={{
                  padding: "16px 20px",
                  backgroundColor: "rgba(118,185,0,0.08)",
                  borderLeft: `4px solid ${THEME.colors.nvidiaGreen}`,
                  borderRadius: 8,
                }}
              >
                <div style={{ fontSize: 16, color: THEME.colors.textPrimary, fontFamily: fontFamilyCode, lineHeight: 1.5 }}>
                  If{" "}
                  <span style={{ color: THEME.colors.accentOrange, fontWeight: 700 }}>AI</span>{" "}
                  {">"} ridge
                </div>
                <div style={{ fontSize: 15, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyBody, marginTop: 4 }}>
                  Compute bound -- optimize arithmetic
                </div>
              </div>
            </div>

            {/* Insight */}
            <div
              style={{
                marginTop: 24,
                padding: "14px 18px",
                backgroundColor: "rgba(255,171,64,0.08)",
                borderRadius: 8,
                border: `1px solid ${THEME.colors.accentOrange}30`,
                opacity: interpolate(
                  frame - 9 * fps,
                  [0, 0.5 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <div style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, lineHeight: 1.5 }}>
                The roofline tells you the{" "}
                <span style={{ color: THEME.colors.accentOrange, fontWeight: 600 }}>theoretical maximum</span>{" "}
                performance for any given kernel. Your optimization goal is to move toward the roof.
              </div>
            </div>
          </div>
        </div>
      }
    />
  );
};
