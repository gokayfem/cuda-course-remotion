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

type PatternCard = {
  title: string;
  description: string;
  diagramTop: string;
  diagramBottom: string;
  color: string;
};

const patterns: PatternCard[] = [
  {
    title: "Data Loading Pipeline",
    description: "Hide transfer latency behind training compute",
    diagramTop: "[Load batch N+1]",
    diagramBottom: "[Train batch N]",
    color: THEME.colors.accentBlue,
  },
  {
    title: "Multi-Stream Inference",
    description: "Distribute requests across 4 streams",
    diagramTop: "Req 1 | Req 2",
    diagramBottom: "Req 3 | Req 4",
    color: THEME.colors.nvidiaGreen,
  },
  {
    title: "Gradient All-Reduce",
    description: "Pipeline compute and reduce per layer",
    diagramTop: "[Compute grad]",
    diagramBottom: "[Reduce across GPUs]",
    color: THEME.colors.accentPurple,
  },
  {
    title: "Prefetch + Compute",
    description: "Overlap next data fetch with current work",
    diagramTop: "[Prefetch next]",
    diagramBottom: "[Process current]",
    color: THEME.colors.accentOrange,
  },
];

export const M6S13_PracticalPatterns: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="dark" moduleNumber={6}>
      <SlideTitle
        title="Practical Stream Patterns for ML"
        subtitle="Common patterns that leverage stream concurrency"
      />

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 20,
          flex: 1,
          width: 1776,
        }}
      >
        {patterns.map((pattern, i) => {
          const cardDelay = 1 * fps + i * 0.6 * fps;
          const cardSpring = spring({
            frame: frame - cardDelay,
            fps,
            config: { damping: 200 },
          });
          const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
          const cardScale = interpolate(cardSpring, [0, 1], [0.92, 1]);

          return (
            <div
              key={pattern.title}
              style={{
                padding: "22px 28px",
                backgroundColor: `${pattern.color}08`,
                border: `2px solid ${pattern.color}40`,
                borderRadius: 12,
                opacity: cardOpacity,
                transform: `scale(${cardScale})`,
                display: "flex",
                flexDirection: "column",
              }}
            >
              {/* Title */}
              <div
                style={{
                  fontSize: 20,
                  fontWeight: 700,
                  color: pattern.color,
                  fontFamily: fontFamilyBody,
                  marginBottom: 14,
                }}
              >
                {pattern.title}
              </div>

              {/* Mini diagram */}
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 6,
                  marginBottom: 14,
                }}
              >
                {/* Overlap visualization */}
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div
                    style={{
                      flex: 1,
                      height: 32,
                      backgroundColor: `${pattern.color}20`,
                      border: `1px solid ${pattern.color}60`,
                      borderRadius: 6,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <span
                      style={{
                        fontSize: 13,
                        color: pattern.color,
                        fontFamily: fontFamilyCode,
                        fontWeight: 600,
                      }}
                    >
                      {pattern.diagramTop}
                    </span>
                  </div>
                </div>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    marginLeft: 60,
                  }}
                >
                  <div
                    style={{
                      flex: 1,
                      height: 32,
                      backgroundColor: `${pattern.color}12`,
                      border: `1px dashed ${pattern.color}50`,
                      borderRadius: 6,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <span
                      style={{
                        fontSize: 13,
                        color: THEME.colors.textSecondary,
                        fontFamily: fontFamilyCode,
                      }}
                    >
                      {pattern.diagramBottom}
                    </span>
                  </div>
                </div>
                {/* Overlap arrow */}
                <div
                  style={{
                    fontSize: 11,
                    color: THEME.colors.textMuted,
                    fontFamily: fontFamilyCode,
                    textAlign: "center",
                  }}
                >
                  {"<-- overlap -->"}
                </div>
              </div>

              {/* Description */}
              <div
                style={{
                  fontSize: 15,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  lineHeight: 1.5,
                }}
              >
                {pattern.description}
              </div>
            </div>
          );
        })}
      </div>
    </SlideLayout>
  );
};
