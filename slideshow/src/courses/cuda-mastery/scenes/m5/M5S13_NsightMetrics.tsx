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

type MetricCard = {
  title: string;
  icon: string;
  value: string;
  description: string;
  color: string;
  status: "good" | "ok" | "bad";
};

const metrics: MetricCard[] = [
  {
    title: "Achieved Occupancy",
    icon: "\u25CB",
    value: "75%",
    description: "Active warps vs theoretical max",
    color: THEME.colors.nvidiaGreen,
    status: "good",
  },
  {
    title: "Memory Throughput",
    icon: "\u2261",
    value: "420 GB/s",
    description: "% of peak bandwidth utilized",
    color: THEME.colors.nvidiaGreen,
    status: "good",
  },
  {
    title: "Compute Throughput",
    icon: "\u26A1",
    value: "6.2 TFLOPS",
    description: "% of peak compute utilized",
    color: THEME.colors.accentYellow,
    status: "ok",
  },
  {
    title: "Warp Stall Reasons",
    icon: "\u23F3",
    value: "memory: 42%",
    description: "Why warps are waiting (memory, sync, etc.)",
    color: THEME.colors.accentRed,
    status: "bad",
  },
  {
    title: "L1/L2 Hit Rate",
    icon: "\u2714",
    value: "L1: 88% L2: 72%",
    description: "Cache efficiency",
    color: THEME.colors.nvidiaGreen,
    status: "good",
  },
  {
    title: "Register Spilling",
    icon: "\u26A0",
    value: "24 regs spilled",
    description: "Registers -> local memory (slow!)",
    color: THEME.colors.accentRed,
    status: "bad",
  },
];

const statusColors: Record<string, string> = {
  good: THEME.colors.nvidiaGreen,
  ok: THEME.colors.accentYellow,
  bad: THEME.colors.accentRed,
};

const statusLabels: Record<string, string> = {
  good: "GOOD",
  ok: "WARNING",
  bad: "CRITICAL",
};

const MetricCardComponent: React.FC<{
  metric: MetricCard;
  index: number;
  baseDelay: number;
}> = ({ metric, index, baseDelay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const delay = baseDelay + index * 0.5 * fps;
  const cardSpring = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });
  const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
  const cardScale = interpolate(cardSpring, [0, 1], [0.9, 1]);

  const statusColor = statusColors[metric.status];

  // Gauge animation for occupancy
  const gaugeProgress = interpolate(
    frame - (delay + 0.3 * fps),
    [0, 0.8 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <div
      style={{
        padding: "20px 24px",
        backgroundColor: "rgba(255,255,255,0.03)",
        borderRadius: 12,
        border: `1px solid ${statusColor}30`,
        opacity: cardOpacity,
        transform: `scale(${cardScale})`,
        display: "flex",
        flexDirection: "column",
        width: 520,
      }}
    >
      {/* Header row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 22 }}>{metric.icon}</span>
          <span
            style={{
              fontSize: 17,
              fontWeight: 700,
              color: THEME.colors.textPrimary,
              fontFamily: fontFamilyBody,
            }}
          >
            {metric.title}
          </span>
        </div>
        <span
          style={{
            fontSize: 11,
            fontWeight: 700,
            color: statusColor,
            fontFamily: fontFamilyBody,
            padding: "3px 10px",
            backgroundColor: `${statusColor}15`,
            borderRadius: 10,
            letterSpacing: "0.5px",
          }}
        >
          {statusLabels[metric.status]}
        </span>
      </div>

      {/* Value bar */}
      <div style={{ marginBottom: 10 }}>
        <div
          style={{
            height: 8,
            backgroundColor: "rgba(255,255,255,0.06)",
            borderRadius: 4,
            overflow: "hidden",
            marginBottom: 8,
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${gaugeProgress * (metric.status === "good" ? 85 : metric.status === "ok" ? 55 : 35)}%`,
              backgroundColor: statusColor,
              borderRadius: 4,
            }}
          />
        </div>
        <div
          style={{
            fontSize: 22,
            fontWeight: 700,
            color: statusColor,
            fontFamily: fontFamilyCode,
          }}
        >
          {metric.value}
        </div>
      </div>

      {/* Description */}
      <div
        style={{
          fontSize: 14,
          color: THEME.colors.textSecondary,
          fontFamily: fontFamilyBody,
          lineHeight: 1.4,
        }}
      >
        {metric.description}
      </div>
    </div>
  );
};

export const M5S13_NsightMetrics: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="dark" moduleNumber={5}>
      <SlideTitle
        title="Key Profiling Metrics"
        subtitle="What to look for in Nsight Compute output"
      />

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: 20,
          flex: 1,
          width: 1776,
        }}
      >
        {metrics.map((metric, i) => (
          <MetricCardComponent
            key={metric.title}
            metric={metric}
            index={i}
            baseDelay={1 * fps}
          />
        ))}
      </div>

      {/* Tip */}
      <div
        style={{
          marginTop: 16,
          padding: "12px 20px",
          backgroundColor: "rgba(79,195,247,0.06)",
          borderRadius: 8,
          border: `1px solid ${THEME.colors.accentBlue}20`,
          width: 1776,
          opacity: interpolate(
            frame - 8 * fps,
            [0, 0.5 * fps],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          ),
        }}
      >
        <div style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, lineHeight: 1.5 }}>
          <span style={{ color: THEME.colors.accentBlue, fontWeight: 700 }}>Pro tip:</span>{" "}
          Focus on the{" "}
          <span style={{ color: THEME.colors.accentRed, fontWeight: 600 }}>red</span>{" "}
          metrics first. Fix the biggest bottleneck before moving to the next one.
        </div>
      </div>
    </SlideLayout>
  );
};
