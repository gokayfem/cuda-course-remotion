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

type Quadrant = {
  title: string;
  pattern: string;
  icon: string;
  color: string;
  description: string;
  detail: string;
};

const quadrants: Quadrant[] = [
  {
    title: "Softmax",
    pattern: "Reduction",
    icon: "\u03C3",
    color: THEME.colors.accentBlue,
    description: "Reduction for max, reduction for sum, then elementwise divide",
    detail: "softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))",
  },
  {
    title: "Loss Computation",
    pattern: "Reduction",
    icon: "\u2211",
    color: THEME.colors.accentOrange,
    description: "Reduction to sum individual per-sample losses into a scalar",
    detail: "L = (1/N) * reduce_sum(loss_per_sample)",
  },
  {
    title: "Sparse Attention",
    pattern: "Compaction",
    icon: "\u2737",
    color: THEME.colors.accentPurple,
    description: "Compaction to filter relevant tokens based on attention scores",
    detail: "Keep only top-k tokens: predicate + scan + scatter",
  },
  {
    title: "Quantization",
    pattern: "Histogram",
    icon: "\u2581\u2583\u2585\u2587",
    color: THEME.colors.nvidiaGreen,
    description: "Histogram to compute value distributions for calibration",
    detail: "Collect activation statistics to choose scale and zero-point",
  },
];

export const M4S15_PracticalApplications: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const summaryOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="accent" moduleNumber={4} slideNumber={15} totalSlides={18}>
      <SlideTitle
        title="Patterns in ML Workloads"
        subtitle="These building blocks appear everywhere in machine learning"
      />

      {/* 2x2 grid of quadrants */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 18,
          flex: 1,
        }}
      >
        {quadrants.map((q, i) => {
          const cardDelay = 1 * fps + i * 0.7 * fps;
          const cardSpring = spring({
            frame: frame - cardDelay,
            fps,
            config: { damping: 200 },
          });
          const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
          const cardScale = interpolate(cardSpring, [0, 1], [0.95, 1]);

          return (
            <div
              key={i}
              style={{
                padding: "20px 24px",
                backgroundColor: `${q.color}08`,
                borderRadius: 12,
                border: `1px solid ${q.color}30`,
                opacity: cardOpacity,
                transform: `scale(${cardScale})`,
                display: "flex",
                flexDirection: "column",
              }}
            >
              {/* Header row: icon + title + pattern badge */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                  marginBottom: 12,
                }}
              >
                <div
                  style={{
                    width: 48,
                    height: 48,
                    borderRadius: 10,
                    backgroundColor: `${q.color}18`,
                    border: `2px solid ${q.color}50`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 22,
                    color: q.color,
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                    flexShrink: 0,
                  }}
                >
                  {q.icon}
                </div>

                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      fontSize: 22,
                      fontWeight: 700,
                      color: THEME.colors.textPrimary,
                      fontFamily: fontFamilyBody,
                    }}
                  >
                    {q.title}
                  </div>
                </div>

                <div
                  style={{
                    padding: "4px 12px",
                    backgroundColor: `${q.color}18`,
                    borderRadius: 6,
                    fontSize: 13,
                    fontWeight: 700,
                    color: q.color,
                    fontFamily: fontFamilyBody,
                  }}
                >
                  {q.pattern}
                </div>
              </div>

              {/* Description */}
              <div
                style={{
                  fontSize: 16,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  lineHeight: 1.5,
                  marginBottom: 10,
                  flex: 1,
                }}
              >
                {q.description}
              </div>

              {/* Code-style detail */}
              <div
                style={{
                  padding: "8px 12px",
                  backgroundColor: "#0d1117",
                  borderRadius: 6,
                  border: "1px solid rgba(255,255,255,0.08)",
                }}
              >
                <pre
                  style={{
                    margin: 0,
                    fontSize: 13,
                    fontFamily: fontFamilyCode,
                    color: THEME.colors.textCode,
                    whiteSpace: "pre-wrap",
                    lineHeight: 1.4,
                  }}
                >
                  {q.detail}
                </pre>
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary footer */}
      <div
        style={{
          marginTop: 14,
          padding: "14px 24px",
          backgroundColor: "rgba(118,185,0,0.08)",
          borderRadius: 10,
          border: `1px solid ${THEME.colors.nvidiaGreen}30`,
          textAlign: "center",
          opacity: summaryOpacity,
        }}
      >
        <span
          style={{
            fontSize: 18,
            color: THEME.colors.textPrimary,
            fontFamily: fontFamilyBody,
            fontWeight: 600,
          }}
        >
          Master{" "}
          <span style={{ color: THEME.colors.nvidiaGreen }}>reduction</span>,{" "}
          <span style={{ color: THEME.colors.accentCyan }}>scan</span>,{" "}
          <span style={{ color: THEME.colors.accentPurple }}>compaction</span>, and{" "}
          <span style={{ color: THEME.colors.accentOrange }}>histogram</span>
          {" "}-- and you can build most GPU-accelerated ML kernels.
        </span>
      </div>
    </SlideLayout>
  );
};
