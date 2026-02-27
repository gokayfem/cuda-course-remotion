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

const BELL_POINTS = 60;

const bellCurve = (x: number, mean: number, sigma: number): number =>
  Math.exp(-0.5 * Math.pow((x - mean) / sigma, 2));

export const M10S05_LossScaling: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const chartWidth = 700;
  const chartHeight = 260;
  const chartPadLeft = 40;
  const chartPadBottom = 40;
  const plotW = chartWidth - chartPadLeft - 20;
  const plotH = chartHeight - chartPadBottom - 20;

  const curveDelay = 1 * fps;
  const curveOpacity = interpolate(
    frame - curveDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const underflowDelay = 2.5 * fps;
  const underflowOpacity = interpolate(
    frame - underflowDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const shiftDelay = 4 * fps;
  const shiftProgress = interpolate(
    frame - shiftDelay,
    [0, 1.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const shiftedOpacity = interpolate(
    frame - shiftDelay,
    [0, 0.3 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const fp16RangeDelay = 1.5 * fps;
  const fp16Opacity = interpolate(
    frame - fp16RangeDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const bottomDelay = 9 * fps;
  const bottomOpacity = interpolate(
    frame - bottomDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const originalMean = 0.25;
  const sigma = 0.15;
  const shiftAmount = 0.25 * shiftProgress;
  const shiftedMean = originalMean + shiftAmount;

  const fp16Min = 0.35;

  const buildPath = (mean: number): string => {
    const points: string[] = [];
    for (let i = 0; i <= BELL_POINTS; i++) {
      const t = i / BELL_POINTS;
      const x = chartPadLeft + t * plotW;
      const val = bellCurve(t, mean, sigma);
      const y = 20 + plotH - val * plotH * 0.9;
      points.push(`${i === 0 ? "M" : "L"}${x},${y}`);
    }
    return points.join(" ");
  };

  const originalPath = buildPath(originalMean);
  const shiftedPath = buildPath(shiftedMean);

  const fp16X = chartPadLeft + fp16Min * plotW;

  const FLOW_STEPS = [
    "Scale up",
    "forward",
    "backward",
    "check Inf/NaN",
    "scale down if needed",
    "optimizer step",
  ] as const;

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={10}
      leftWidth="55%"
      left={
        <div style={{ width: 860 }}>
          <SlideTitle
            title="Loss Scaling -- Making FP16 Work"
            subtitle="Preventing gradient underflow in half precision"
          />

          {/* Gradient distribution chart */}
          <div
            style={{
              position: "relative",
              width: chartWidth,
              height: chartHeight,
              marginTop: 8,
            }}
          >
            <svg width={chartWidth} height={chartHeight}>
              {/* Axis */}
              <line
                x1={chartPadLeft}
                y1={chartHeight - chartPadBottom}
                x2={chartWidth - 20}
                y2={chartHeight - chartPadBottom}
                stroke={THEME.colors.textMuted}
                strokeWidth={1}
              />
              <text
                x={chartWidth / 2}
                y={chartHeight - 10}
                fill={THEME.colors.textMuted}
                fontSize={12}
                fontFamily={fontFamilyBody}
                textAnchor="middle"
              >
                Gradient Magnitude (log scale)
              </text>

              {/* FP16 representable range */}
              <rect
                x={fp16X}
                y={20}
                width={chartWidth - 20 - fp16X}
                height={plotH}
                fill={`${THEME.colors.nvidiaGreen}10`}
                opacity={fp16Opacity}
              />
              <line
                x1={fp16X}
                y1={20}
                x2={fp16X}
                y2={20 + plotH}
                stroke={THEME.colors.nvidiaGreen}
                strokeWidth={2}
                strokeDasharray="6,4"
                opacity={fp16Opacity}
              />
              <text
                x={fp16X + 8}
                y={36}
                fill={THEME.colors.nvidiaGreen}
                fontSize={11}
                fontFamily={fontFamilyBody}
                opacity={fp16Opacity}
              >
                FP16 min
              </text>

              {/* Underflow zone label */}
              <text
                x={chartPadLeft + (fp16X - chartPadLeft) / 2}
                y={50}
                fill={THEME.colors.accentRed}
                fontSize={12}
                fontWeight={700}
                fontFamily={fontFamilyBody}
                textAnchor="middle"
                opacity={underflowOpacity}
              >
                Underflow Zone
              </text>
              <rect
                x={chartPadLeft}
                y={20}
                width={fp16X - chartPadLeft}
                height={plotH}
                fill={`${THEME.colors.accentRed}12`}
                opacity={underflowOpacity}
              />

              {/* Original curve */}
              <path
                d={originalPath}
                fill="none"
                stroke={THEME.colors.accentRed}
                strokeWidth={2.5}
                opacity={curveOpacity * (1 - shiftProgress * 0.5)}
              />

              {/* Shifted curve */}
              {shiftedOpacity > 0 && (
                <path
                  d={shiftedPath}
                  fill="none"
                  stroke={THEME.colors.nvidiaGreen}
                  strokeWidth={2.5}
                  opacity={shiftedOpacity}
                />
              )}

              {/* Scale arrow */}
              {shiftProgress > 0.1 && (
                <>
                  <line
                    x1={chartPadLeft + originalMean * plotW}
                    y1={chartHeight - chartPadBottom - 20}
                    x2={chartPadLeft + shiftedMean * plotW}
                    y2={chartHeight - chartPadBottom - 20}
                    stroke={THEME.colors.accentYellow}
                    strokeWidth={2}
                    markerEnd="url(#arrowhead)"
                    opacity={shiftProgress}
                  />
                  <defs>
                    <marker
                      id="arrowhead"
                      markerWidth="8"
                      markerHeight="6"
                      refX="8"
                      refY="3"
                      orient="auto"
                    >
                      <polygon
                        points="0 0, 8 3, 0 6"
                        fill={THEME.colors.accentYellow}
                      />
                    </marker>
                  </defs>
                  <text
                    x={(chartPadLeft + originalMean * plotW + chartPadLeft + shiftedMean * plotW) / 2}
                    y={chartHeight - chartPadBottom - 28}
                    fill={THEME.colors.accentYellow}
                    fontSize={12}
                    fontWeight={700}
                    fontFamily={fontFamilyBody}
                    textAnchor="middle"
                    opacity={shiftProgress}
                  >
                    x scale factor
                  </text>
                </>
              )}
            </svg>
          </div>

          {/* Dynamic loss scaling flow */}
          <div
            style={{
              marginTop: 16,
              display: "flex",
              gap: 6,
              alignItems: "center",
              flexWrap: "wrap",
              opacity: bottomOpacity,
            }}
          >
            {FLOW_STEPS.map((step, i) => {
              const stepDelay = bottomDelay + i * 0.15 * fps;
              const stepOpacity = interpolate(
                frame - stepDelay,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              return (
                <React.Fragment key={step}>
                  <div
                    style={{
                      padding: "5px 10px",
                      backgroundColor: "rgba(255,255,255,0.06)",
                      borderRadius: 6,
                      fontSize: 12,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyCode,
                      fontWeight: 600,
                      opacity: stepOpacity,
                      flexShrink: 0,
                    }}
                  >
                    {step}
                  </div>
                  {i < FLOW_STEPS.length - 1 && (
                    <span
                      style={{
                        color: THEME.colors.textMuted,
                        fontSize: 14,
                        opacity: stepOpacity,
                      }}
                    >
                      {"\u2192"}
                    </span>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>
      }
      right={
        <div style={{ width: 540, marginTop: 80 }}>
          <BulletPoint
            text="Problem: small gradients underflow to zero in FP16"
            index={0}
            delay={2.5 * fps}
            highlight
            subtext="Gradients near 1e-8 cannot be represented in half precision"
          />
          <BulletPoint
            text="Solution: scale loss by large factor (e.g., 1024)"
            index={1}
            delay={2.5 * fps}
            subtext="All gradients scale proportionally via chain rule"
          />
          <BulletPoint
            text="Gradients scale proportionally -- stay in FP16 range"
            index={2}
            delay={2.5 * fps}
            highlight
            subtext="Shifts the entire gradient distribution into representable values"
          />
          <BulletPoint
            text="Unscale before optimizer step"
            index={3}
            delay={2.5 * fps}
            subtext="Divide by scale factor so weight updates remain correct"
          />
        </div>
      }
    />
  );
};
