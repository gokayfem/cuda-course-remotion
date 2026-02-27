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
import { fontFamilyBody, fontFamilyCode, fontFamilyHeading } from "../../../../styles/fonts";

export const M5S07_BandwidthOptimization: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const techniques = [
    {
      title: "Vectorized Loads",
      icon: "x4",
      description: "Load 16 bytes per instruction instead of 4",
      code: "float4 val = reinterpret_cast<float4*>(input)[tid];",
      color: THEME.colors.accentBlue,
      delay: 1.5 * fps,
    },
    {
      title: "Alignment",
      icon: "[ ]",
      description: "Ensure 128-byte aligned allocations",
      code: "cudaMallocPitch(&ptr, &pitch, width, height);",
      color: THEME.colors.accentPurple,
      delay: 3.5 * fps,
    },
    {
      title: "Cache Utilization",
      icon: "L1",
      description: "Fit working set in L1/L2",
      code: "__launch_bounds__(256, 2)",
      color: THEME.colors.nvidiaGreen,
      delay: 5.5 * fps,
    },
  ];

  // Bottom comparison bars
  const comparisonData = [
    { label: "Scalar (float)", pct: 40, color: THEME.colors.accentRed },
    { label: "float2", pct: 68, color: THEME.colors.accentOrange },
    { label: "float4", pct: 92, color: THEME.colors.nvidiaGreen },
  ];
  const compDelay = 8 * fps;
  const COMP_BAR_WIDTH = 280;

  return (
    <SlideLayout variant="gradient" moduleNumber={5}>
      <SlideTitle
        title="Maximizing Memory Throughput"
        subtitle="Three key techniques for better bandwidth utilization"
      />

      {/* Three technique cards */}
      <div
        style={{
          display: "flex",
          gap: 28,
          flex: 1,
          width: 1776,
        }}
      >
        {techniques.map((tech, i) => {
          const cardSpring = spring({
            frame: frame - tech.delay,
            fps,
            config: { damping: 200 },
          });
          const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
          const cardY = interpolate(cardSpring, [0, 1], [40, 0]);

          return (
            <div
              key={tech.title}
              style={{
                flex: 1,
                opacity: cardOpacity,
                transform: `translateY(${cardY}px)`,
                display: "flex",
                flexDirection: "column",
                maxWidth: 560,
              }}
            >
              {/* Card */}
              <div
                style={{
                  padding: "22px 24px",
                  backgroundColor: `${tech.color}08`,
                  borderRadius: 12,
                  border: `1px solid ${tech.color}30`,
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                {/* Icon */}
                <div
                  style={{
                    width: 52,
                    height: 42,
                    borderRadius: 8,
                    backgroundColor: `${tech.color}18`,
                    border: `2px solid ${tech.color}60`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 16,
                    fontWeight: 800,
                    color: tech.color,
                    fontFamily: fontFamilyCode,
                    marginBottom: 14,
                  }}
                >
                  {tech.icon}
                </div>

                {/* Title */}
                <div
                  style={{
                    fontSize: 22,
                    fontWeight: 700,
                    color: tech.color,
                    fontFamily: fontFamilyHeading,
                    marginBottom: 8,
                  }}
                >
                  {tech.title}
                </div>

                {/* Description */}
                <div
                  style={{
                    fontSize: 16,
                    color: THEME.colors.textSecondary,
                    fontFamily: fontFamilyBody,
                    marginBottom: 16,
                    lineHeight: 1.5,
                  }}
                >
                  {tech.description}
                </div>

                {/* Code */}
                <div
                  style={{
                    padding: "10px 14px",
                    backgroundColor: THEME.colors.bgCode,
                    borderRadius: 8,
                    border: "1px solid rgba(255,255,255,0.06)",
                    marginTop: "auto",
                  }}
                >
                  <code
                    style={{
                      fontSize: 13,
                      color: THEME.colors.textCode,
                      fontFamily: fontFamilyCode,
                      wordBreak: "break-all",
                      lineHeight: 1.5,
                    }}
                  >
                    {tech.code}
                  </code>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Bottom: bandwidth comparison */}
      <div
        style={{
          marginTop: 20,
          width: 1776,
        }}
      >
        <div
          style={{
            fontSize: 15,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
            fontWeight: 600,
            marginBottom: 10,
            opacity: interpolate(
              frame - compDelay,
              [0, 0.3 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          Bandwidth Achieved (% of peak)
        </div>
        <div
          style={{
            display: "flex",
            gap: 32,
            alignItems: "center",
            width: 1776,
          }}
        >
          {comparisonData.map((d, i) => {
            const barDelay = compDelay + i * 0.3 * fps;
            const barSpring = spring({
              frame: frame - barDelay,
              fps,
              config: { damping: 100, stiffness: 80 },
            });
            const barWidth = interpolate(barSpring, [0, 1], [0, (d.pct / 100) * COMP_BAR_WIDTH]);
            const barOpacity = interpolate(barSpring, [0, 1], [0, 1]);

            return (
              <div
                key={d.label}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  opacity: barOpacity,
                }}
              >
                <span
                  style={{
                    fontSize: 14,
                    color: THEME.colors.textSecondary,
                    fontFamily: fontFamilyBody,
                    fontWeight: 600,
                    minWidth: 90,
                    textAlign: "right",
                  }}
                >
                  {d.label}
                </span>
                <div
                  style={{
                    width: COMP_BAR_WIDTH,
                    height: 22,
                    backgroundColor: "rgba(255,255,255,0.04)",
                    borderRadius: 4,
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      width: barWidth,
                      height: 22,
                      backgroundColor: `${d.color}80`,
                      borderRadius: 4,
                      background: `linear-gradient(90deg, ${d.color}60, ${d.color})`,
                    }}
                  />
                </div>
                <span
                  style={{
                    fontSize: 16,
                    fontWeight: 700,
                    color: d.color,
                    fontFamily: fontFamilyCode,
                    minWidth: 40,
                  }}
                >
                  {d.pct}%
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </SlideLayout>
  );
};
