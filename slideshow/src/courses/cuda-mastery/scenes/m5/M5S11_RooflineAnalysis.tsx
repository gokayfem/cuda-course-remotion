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

const CHART = {
  width: 1700,
  height: 440,
  padLeft: 90,
  padRight: 40,
  padTop: 20,
  padBottom: 60,
} as const;

const PEAK_COMPUTE = 10000;
const PEAK_BW = 900;
const RIDGE_AI = PEAK_COMPUTE / PEAK_BW;

const logScale = (val: number, min: number, max: number) => {
  const logMin = Math.log2(min);
  const logMax = Math.log2(max);
  return (Math.log2(val) - logMin) / (logMax - logMin);
};

const toChartX = (ai: number) => {
  const frac = logScale(ai, 0.1, 128);
  return CHART.padLeft + frac * (CHART.width - CHART.padLeft - CHART.padRight);
};

const toChartY = (gflops: number) => {
  const logMin = Math.log2(30);
  const logMax = Math.log2(PEAK_COMPUTE * 1.5);
  const frac = (Math.log2(Math.max(gflops, 30)) - logMin) / (logMax - logMin);
  return CHART.height - CHART.padBottom - frac * (CHART.height - CHART.padTop - CHART.padBottom);
};

type KernelDot = {
  name: string;
  ai: number;
  color: string;
  delayOffset: number;
};

const kernels: KernelDot[] = [
  { name: "vectorAdd", ai: 0.25, color: THEME.colors.accentRed, delayOffset: 0 },
  { name: "Reduction", ai: 0.17, color: THEME.colors.accentRed, delayOffset: 1.2 },
  { name: "SAXPY", ai: 0.33, color: THEME.colors.accentOrange, delayOffset: 2.4 },
  { name: "Matrix Multiply", ai: 32, color: THEME.colors.nvidiaGreen, delayOffset: 3.6 },
  { name: "Convolution", ai: 6, color: THEME.colors.accentBlue, delayOffset: 4.8 },
];

export const M5S11_RooflineAnalysis: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const chartOpacity = interpolate(
    frame - 0.5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Build the full roofline path
  const roofPoints: Array<{ x: number; y: number }> = [];
  for (let i = 0; i <= 80; i++) {
    const t = i / 80;
    const ai = 0.1 * Math.pow(128 / 0.1, t);
    const perf = Math.min(ai * PEAK_BW, PEAK_COMPUTE);
    roofPoints.push({ x: toChartX(ai), y: toChartY(perf) });
  }
  const roofPath = roofPoints
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`)
    .join(" ");

  const insightOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="dark" moduleNumber={5}>
      <SlideTitle title="Placing Kernels on the Roofline" />

      <svg
        width={CHART.width}
        height={CHART.height}
        style={{ overflow: "visible", opacity: chartOpacity }}
      >
        {/* Axes */}
        <line
          x1={CHART.padLeft}
          y1={CHART.height - CHART.padBottom}
          x2={CHART.width - CHART.padRight}
          y2={CHART.height - CHART.padBottom}
          stroke={THEME.colors.textMuted}
          strokeWidth={1.5}
        />
        <line
          x1={CHART.padLeft}
          y1={CHART.padTop}
          x2={CHART.padLeft}
          y2={CHART.height - CHART.padBottom}
          stroke={THEME.colors.textMuted}
          strokeWidth={1.5}
        />

        {/* X axis labels */}
        {[0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64].map((val) => {
          const cx = toChartX(val);
          return (
            <g key={val}>
              <line
                x1={cx}
                y1={CHART.height - CHART.padBottom}
                x2={cx}
                y2={CHART.height - CHART.padBottom + 5}
                stroke={THEME.colors.textMuted}
                strokeWidth={1}
              />
              <text
                x={cx}
                y={CHART.height - CHART.padBottom + 18}
                fill={THEME.colors.textSecondary}
                fontSize={11}
                fontFamily={fontFamilyCode}
                textAnchor="middle"
              >
                {val}
              </text>
              <line
                x1={cx}
                y1={CHART.padTop}
                x2={cx}
                y2={CHART.height - CHART.padBottom}
                stroke="rgba(255,255,255,0.03)"
                strokeWidth={1}
              />
            </g>
          );
        })}

        <text
          x={(CHART.padLeft + CHART.width - CHART.padRight) / 2}
          y={CHART.height - 6}
          fill={THEME.colors.textSecondary}
          fontSize={13}
          fontFamily={fontFamilyBody}
          textAnchor="middle"
        >
          Arithmetic Intensity (FLOPs/byte)
        </text>

        {/* Y axis labels */}
        {[100, 500, 1000, 5000, 10000].map((val) => {
          const cy = toChartY(val);
          return (
            <g key={val}>
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
                stroke="rgba(255,255,255,0.03)"
                strokeWidth={1}
              />
            </g>
          );
        })}

        {/* Roofline */}
        <path
          d={roofPath}
          fill="none"
          stroke="rgba(255,255,255,0.2)"
          strokeWidth={2.5}
          strokeLinecap="round"
        />

        {/* Bandwidth region fill */}
        <path
          d={`${roofPath} L ${toChartX(128)} ${CHART.height - CHART.padBottom} L ${CHART.padLeft} ${CHART.height - CHART.padBottom} Z`}
          fill="rgba(255,255,255,0.02)"
        />

        {/* Ridge point marker */}
        <circle
          cx={toChartX(RIDGE_AI)}
          cy={toChartY(PEAK_COMPUTE)}
          r={5}
          fill={THEME.colors.accentOrange}
          opacity={0.6}
        />

        {/* Kernel dots */}
        {kernels.map((kernel) => {
          const dotDelay = 2 * fps + kernel.delayOffset * fps;
          const dotSpring = spring({
            frame: frame - dotDelay,
            fps,
            config: { damping: 200 },
          });
          const dotOpacity = interpolate(dotSpring, [0, 1], [0, 1]);
          const dotScale = interpolate(dotSpring, [0, 1], [0, 1]);

          const perfOnRoof = Math.min(kernel.ai * PEAK_BW, PEAK_COMPUTE);
          const actualPerf = perfOnRoof * 0.6; // Show kernels below the roof
          const dx = toChartX(kernel.ai);
          const dy = toChartY(actualPerf);
          const roofY = toChartY(perfOnRoof);

          return (
            <g key={kernel.name} opacity={dotOpacity}>
              {/* Line to roofline */}
              <line
                x1={dx}
                y1={dy}
                x2={dx}
                y2={roofY}
                stroke={kernel.color}
                strokeWidth={1.5}
                strokeDasharray="4,3"
                opacity={0.5}
              />
              {/* Dot */}
              <circle
                cx={dx}
                cy={dy}
                r={10 * dotScale}
                fill={kernel.color}
                stroke={THEME.colors.textPrimary}
                strokeWidth={2}
              />
              {/* Label */}
              <text
                x={dx}
                y={dy + 24}
                fill={kernel.color}
                fontSize={13}
                fontFamily={fontFamilyBody}
                fontWeight={600}
                textAnchor="middle"
              >
                {kernel.name}
              </text>
              <text
                x={dx}
                y={dy + 40}
                fill={THEME.colors.textSecondary}
                fontSize={11}
                fontFamily={fontFamilyCode}
                textAnchor="middle"
              >
                AI={kernel.ai}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Insight box */}
      <div
        style={{
          marginTop: 16,
          padding: "14px 24px",
          backgroundColor: "rgba(118,185,0,0.06)",
          borderRadius: 10,
          border: `1px solid ${THEME.colors.nvidiaGreen}30`,
          opacity: insightOpacity,
          width: 1700,
        }}
      >
        <div
          style={{
            fontSize: 16,
            color: THEME.colors.textPrimary,
            fontFamily: fontFamilyBody,
            lineHeight: 1.5,
          }}
        >
          Most ML element-wise ops are{" "}
          <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>memory-bound</span>.
          MatMul and Conv are{" "}
          <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>compute-bound</span>.
          This determines your optimization strategy.
        </div>
      </div>
    </SlideLayout>
  );
};
