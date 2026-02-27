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

type VersionCard = {
  title: string;
  badge: string;
  badgeColor: string;
  description: string;
  codeLine: string;
  details: string[];
  perfValue: number;
  perfLabel: string;
};

const versions: VersionCard[] = [
  {
    title: "Version 1: Global Atomics",
    badge: "Simple but Slow",
    badgeColor: THEME.colors.accentOrange,
    description: "Every thread does atomicAdd directly to global memory histogram.",
    codeLine: "atomicAdd(&hist[data[i]], 1);",
    details: [
      "One line of code",
      "High contention on popular bins",
      "Serialized at global memory",
    ],
    perfValue: 1,
    perfLabel: "1x (baseline)",
  },
  {
    title: "Version 2: Shared Privatization",
    badge: "10-20x Faster",
    badgeColor: THEME.colors.nvidiaGreen,
    description: "Each block builds a local histogram in shared memory, then merges to global.",
    codeLine: "atomicAdd(&s_hist[data[i]], 1);  // shared mem",
    details: [
      "Per-block private histogram",
      "__syncthreads between phases",
      "Merge: atomicAdd shared to global",
    ],
    perfValue: 15,
    perfLabel: "~15x faster",
  },
  {
    title: "Version 3: Warp-Level",
    badge: "Maximum Performance",
    badgeColor: THEME.colors.accentCyan,
    description: "Per-warp private histograms eliminate even intra-block contention.",
    codeLine: "atomicAdd(&w_hist[warpId][data[i]], 1);",
    details: [
      "32 warps = 32 private copies",
      "Zero contention within warp",
      "Final merge across warps + blocks",
    ],
    perfValue: 20,
    perfLabel: "~20x faster",
  },
];

export const M4S11_HistogramVersions: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const perfBarDelay = 6 * fps;
  const perfOpacity = interpolate(
    frame - perfBarDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={4} slideNumber={11} totalSlides={18}>
      <SlideTitle
        title="Histogram: Three Approaches"
        subtitle="From naive to high-performance -- privatization is the key optimization"
      />

      {/* Three version cards */}
      <div style={{ display: "flex", gap: 20, flex: 1 }}>
        {versions.map((v, i) => {
          const cardDelay = 0.8 * fps + i * 0.8 * fps;
          const cardSpring = spring({
            frame: frame - cardDelay,
            fps,
            config: { damping: 200 },
          });
          const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
          const cardY = interpolate(cardSpring, [0, 1], [20, 0]);

          return (
            <div
              key={i}
              style={{
                flex: 1,
                padding: "18px 20px",
                backgroundColor: "rgba(255,255,255,0.03)",
                borderRadius: 12,
                border: `1px solid ${v.badgeColor}30`,
                opacity: cardOpacity,
                transform: `translateY(${cardY}px)`,
                display: "flex",
                flexDirection: "column",
              }}
            >
              {/* Badge */}
              <div
                style={{
                  alignSelf: "flex-start",
                  padding: "4px 12px",
                  backgroundColor: `${v.badgeColor}18`,
                  borderRadius: 6,
                  fontSize: 13,
                  fontWeight: 700,
                  color: v.badgeColor,
                  fontFamily: fontFamilyBody,
                  marginBottom: 10,
                }}
              >
                {v.badge}
              </div>

              {/* Title */}
              <div
                style={{
                  fontSize: 18,
                  fontWeight: 700,
                  color: THEME.colors.textPrimary,
                  fontFamily: fontFamilyBody,
                  marginBottom: 6,
                }}
              >
                {v.title}
              </div>

              {/* Description */}
              <div
                style={{
                  fontSize: 14,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  lineHeight: 1.5,
                  marginBottom: 12,
                }}
              >
                {v.description}
              </div>

              {/* Code snippet */}
              <div
                style={{
                  padding: "8px 12px",
                  backgroundColor: "#0d1117",
                  borderRadius: 6,
                  border: "1px solid rgba(255,255,255,0.08)",
                  marginBottom: 14,
                }}
              >
                <pre
                  style={{
                    margin: 0,
                    fontSize: 13,
                    fontFamily: fontFamilyCode,
                    color: THEME.colors.syntaxFunction,
                    whiteSpace: "pre-wrap",
                    lineHeight: 1.5,
                  }}
                >
                  {v.codeLine}
                </pre>
              </div>

              {/* Detail bullets */}
              <div style={{ flex: 1 }}>
                {v.details.map((d, j) => (
                  <div
                    key={j}
                    style={{
                      display: "flex",
                      alignItems: "flex-start",
                      gap: 8,
                      marginBottom: 6,
                    }}
                  >
                    <span
                      style={{
                        color: v.badgeColor,
                        fontSize: 14,
                        fontWeight: 700,
                        flexShrink: 0,
                      }}
                    >
                      {"▸"}
                    </span>
                    <span
                      style={{
                        fontSize: 14,
                        color: THEME.colors.textSecondary,
                        fontFamily: fontFamilyBody,
                        lineHeight: 1.4,
                      }}
                    >
                      {d}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Performance comparison bar chart */}
      <div
        style={{
          marginTop: 16,
          padding: "16px 24px",
          backgroundColor: "rgba(255,255,255,0.03)",
          borderRadius: 10,
          border: "1px solid rgba(255,255,255,0.06)",
          opacity: perfOpacity,
        }}
      >
        <div
          style={{
            fontSize: 15,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
            fontWeight: 700,
            marginBottom: 12,
          }}
        >
          Relative Performance
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {versions.map((v, i) => {
            const barDelay = perfBarDelay + 0.3 * fps * i;
            const barWidth = interpolate(
              frame - barDelay,
              [0, 0.8 * fps],
              [0, (v.perfValue / 20) * 100],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );

            return (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 14 }}>
                <span
                  style={{
                    width: 180,
                    fontSize: 13,
                    color: THEME.colors.textSecondary,
                    fontFamily: fontFamilyBody,
                    flexShrink: 0,
                  }}
                >
                  V{i + 1}: {v.perfLabel}
                </span>
                <div
                  style={{
                    flex: 1,
                    height: 18,
                    backgroundColor: "rgba(255,255,255,0.05)",
                    borderRadius: 4,
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      width: `${barWidth}%`,
                      height: "100%",
                      backgroundColor: v.badgeColor,
                      borderRadius: 4,
                      minWidth: barWidth > 0 ? 4 : 0,
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </SlideLayout>
  );
};
