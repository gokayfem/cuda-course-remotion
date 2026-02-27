import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
  AbsoluteFill,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideBackground } from "../../../../components/SlideBackground";
import { fontFamilyHeading, fontFamilyBody } from "../../../../styles/fonts";

export const M5S01_Title: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const barWidth = interpolate(frame, [0, 1 * fps], [0, 400], {
    extrapolateRight: "clamp",
  });

  const titleSpring = spring({
    frame,
    fps,
    config: { damping: 200 },
    delay: 0.3 * fps,
  });
  const titleOpacity = interpolate(titleSpring, [0, 1], [0, 1]);
  const titleY = interpolate(titleSpring, [0, 1], [40, 0]);

  const subtitleOpacity = interpolate(
    frame,
    [1 * fps, 1.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const moduleOpacity = interpolate(
    frame,
    [1.5 * fps, 2 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Performance gauge animation
  const GAUGE_CENTER_X = 180;
  const GAUGE_CENTER_Y = 180;
  const GAUGE_RADIUS = 140;
  const GAUGE_STROKE = 18;

  // Animate from 30% to 95% over time
  const gaugeDelay = 1.5 * fps;
  const gaugeProgress = interpolate(
    frame - gaugeDelay,
    [0, 3 * fps],
    [0.3, 0.95],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const gaugeOpacity = interpolate(
    frame - 1.2 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Gauge arc: 180 degrees from left to right (bottom half arc inverted = top arc)
  const startAngle = Math.PI * 0.75;
  const endAngle = Math.PI * 2.25;
  const totalArc = endAngle - startAngle;
  const circumference = GAUGE_RADIUS * totalArc;

  const bgArcPath = describeArc(GAUGE_CENTER_X, GAUGE_CENTER_Y, GAUGE_RADIUS, startAngle, endAngle);
  const fillAngle = startAngle + totalArc * gaugeProgress;
  const fillArcPath = describeArc(GAUGE_CENTER_X, GAUGE_CENTER_Y, GAUGE_RADIUS, startAngle, fillAngle);

  // Color transitions from red to yellow to green
  const gaugeColor = gaugeProgress < 0.5
    ? THEME.colors.accentRed
    : gaugeProgress < 0.75
      ? THEME.colors.accentOrange
      : THEME.colors.nvidiaGreen;

  const percentText = `${Math.round(gaugeProgress * 100)}%`;

  // Label animation
  const labelOpacity = interpolate(
    frame - gaugeDelay,
    [0, 0.5 * fps, 2.5 * fps, 3 * fps],
    [0, 1, 1, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const labelText = gaugeProgress < 0.5 ? "Naive" : gaugeProgress < 0.85 ? "Improving..." : "Optimized";

  // Tick marks for the gauge
  const tickCount = 10;
  const ticks = Array.from({ length: tickCount + 1 }).map((_, i) => {
    const angle = startAngle + (totalArc * i) / tickCount;
    const innerR = GAUGE_RADIUS - GAUGE_STROKE / 2 - 6;
    const outerR = GAUGE_RADIUS - GAUGE_STROKE / 2 - 2;
    return {
      x1: GAUGE_CENTER_X + Math.cos(angle) * innerR,
      y1: GAUGE_CENTER_Y + Math.sin(angle) * innerR,
      x2: GAUGE_CENTER_X + Math.cos(angle) * outerR,
      y2: GAUGE_CENTER_Y + Math.sin(angle) * outerR,
    };
  });

  return (
    <AbsoluteFill>
      <SlideBackground variant="accent" />

      {/* Performance gauge on right */}
      <div
        style={{
          position: "absolute",
          right: 100,
          top: 140,
          width: 360,
          height: 360,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          opacity: gaugeOpacity,
        }}
      >
        <svg width={360} height={280} viewBox="0 0 360 280">
          {/* Background arc */}
          <path
            d={bgArcPath}
            fill="none"
            stroke="rgba(255,255,255,0.08)"
            strokeWidth={GAUGE_STROKE}
            strokeLinecap="round"
          />

          {/* Tick marks */}
          {ticks.map((t, i) => (
            <line
              key={i}
              x1={t.x1}
              y1={t.y1}
              x2={t.x2}
              y2={t.y2}
              stroke="rgba(255,255,255,0.2)"
              strokeWidth={1.5}
            />
          ))}

          {/* Filled arc */}
          <path
            d={fillArcPath}
            fill="none"
            stroke={gaugeColor}
            strokeWidth={GAUGE_STROKE}
            strokeLinecap="round"
          />

          {/* Percentage text */}
          <text
            x={GAUGE_CENTER_X}
            y={GAUGE_CENTER_Y + 10}
            textAnchor="middle"
            fill={gaugeColor}
            fontSize={52}
            fontWeight={800}
            fontFamily={fontFamilyHeading}
          >
            {percentText}
          </text>

          {/* Label */}
          <text
            x={GAUGE_CENTER_X}
            y={GAUGE_CENTER_Y + 45}
            textAnchor="middle"
            fill={THEME.colors.textSecondary}
            fontSize={18}
            fontWeight={400}
            fontFamily={fontFamilyBody}
            opacity={labelOpacity}
          >
            {labelText}
          </text>

          {/* Scale labels */}
          <text
            x={GAUGE_CENTER_X - GAUGE_RADIUS - 10}
            y={GAUGE_CENTER_Y + 40}
            textAnchor="middle"
            fill={THEME.colors.textMuted}
            fontSize={13}
            fontFamily={fontFamilyBody}
          >
            0%
          </text>
          <text
            x={GAUGE_CENTER_X + GAUGE_RADIUS + 10}
            y={GAUGE_CENTER_Y + 40}
            textAnchor="middle"
            fill={THEME.colors.textMuted}
            fontSize={13}
            fontFamily={fontFamilyBody}
          >
            100%
          </text>
        </svg>

        {/* GPU Utilization label */}
        <div
          style={{
            fontSize: 16,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
            fontWeight: 600,
            marginTop: -20,
            textAlign: "center",
            letterSpacing: "1px",
            textTransform: "uppercase",
          }}
        >
          GPU Utilization
        </div>
      </div>

      {/* Main content on left */}
      <div
        style={{
          position: "absolute",
          left: 100,
          top: "50%",
          transform: "translateY(-50%)",
          maxWidth: 800,
        }}
      >
        <div
          style={{
            width: barWidth,
            height: 6,
            backgroundColor: THEME.colors.nvidiaGreen,
            borderRadius: 3,
            marginBottom: 32,
          }}
        />

        <h1
          style={{
            fontSize: 72,
            fontWeight: 900,
            color: THEME.colors.textPrimary,
            fontFamily: fontFamilyHeading,
            margin: 0,
            opacity: titleOpacity,
            transform: `translateY(${titleY}px)`,
            lineHeight: 1.1,
            letterSpacing: "-2px",
          }}
        >
          Performance{" "}
          <span style={{ color: THEME.colors.nvidiaGreen }}>Optimization</span>
        </h1>

        <p
          style={{
            fontSize: 28,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
            margin: 0,
            marginTop: 24,
            opacity: subtitleOpacity,
            fontWeight: 400,
          }}
        >
          Occupancy, Bandwidth, ILP & Profiling
        </p>

        <div
          style={{
            marginTop: 48,
            display: "flex",
            gap: 16,
            alignItems: "center",
            opacity: moduleOpacity,
          }}
        >
          <div
            style={{
              padding: "12px 28px",
              backgroundColor: "rgba(118,185,0,0.15)",
              border: `2px solid ${THEME.colors.nvidiaGreen}`,
              borderRadius: 30,
              fontSize: 22,
              color: THEME.colors.nvidiaGreen,
              fontFamily: fontFamilyBody,
              fontWeight: 700,
            }}
          >
            Module 5
          </div>
          <span
            style={{
              fontSize: 22,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
            }}
          >
            Squeezing every drop of performance
          </span>
        </div>
      </div>

      {/* Bottom gradient bar */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          height: 6,
          background: `linear-gradient(90deg, ${THEME.colors.nvidiaGreen}, ${THEME.colors.accentBlue}, ${THEME.colors.accentPurple})`,
        }}
      />
    </AbsoluteFill>
  );
};

function describeArc(
  cx: number,
  cy: number,
  r: number,
  startAngle: number,
  endAngle: number
): string {
  const x1 = cx + Math.cos(startAngle) * r;
  const y1 = cy + Math.sin(startAngle) * r;
  const x2 = cx + Math.cos(endAngle) * r;
  const y2 = cy + Math.sin(endAngle) * r;
  const largeArc = endAngle - startAngle > Math.PI ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`;
}
