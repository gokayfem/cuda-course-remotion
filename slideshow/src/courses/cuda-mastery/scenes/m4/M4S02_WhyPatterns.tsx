import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M4S02_WhyPatterns: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const patterns = [
    {
      name: "Reduction",
      color: THEME.colors.nvidiaGreen,
      uses: "loss computation, gradient accumulation, softmax",
      icon: "SUM",
    },
    {
      name: "Scan",
      color: THEME.colors.accentCyan,
      uses: "cumulative sums, stream compaction, sorting",
      icon: "PRE",
    },
    {
      name: "Histogram",
      color: THEME.colors.accentOrange,
      uses: "data analysis, quantization, distributions",
      icon: "BIN",
    },
    {
      name: "Compaction",
      color: THEME.colors.accentPurple,
      uses: "sparse ops, filtering, top-k selection",
      icon: "FLT",
    },
  ];

  // Compose visual on the right: patterns feed into a "GPU Pipeline" box
  const composeDelay = 5 * fps;
  const composeSpring = spring({
    frame: frame - composeDelay,
    fps,
    config: { damping: 200 },
  });
  const composeOpacity = interpolate(composeSpring, [0, 1], [0, 1]);

  return (
    <SlideLayout variant="gradient" moduleNumber={4} slideNumber={2} totalSlides={18}>
      <SlideTitle
        title="Why Parallel Patterns Matter"
        subtitle="These four patterns are the building blocks for ALL GPU computing"
      />

      <div style={{ display: "flex", gap: 48, flex: 1 }}>
        {/* Left: Bullet points */}
        <div style={{ flex: 1 }}>
          {patterns.map((p, i) => (
            <BulletPoint
              key={p.name}
              index={i}
              delay={1 * fps}
              text={`${p.name} \u2192 ${p.uses}`}
              highlight={i === 0}
            />
          ))}

          {/* Bottom insight box */}
          <div
            style={{
              marginTop: 28,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: interpolate(
                frame - 4 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
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
              Master these four and you can build{" "}
              <span style={{ color: THEME.colors.nvidiaGreen }}>anything</span>{" "}
              on a GPU.
            </span>
          </div>
        </div>

        {/* Right: Composition visual */}
        <div
          style={{
            flex: 0.8,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 12,
          }}
        >
          {/* Pattern boxes flowing into pipeline */}
          {patterns.map((p, i) => {
            const boxDelay = 2 * fps + i * 0.4 * fps;
            const boxSpring = spring({
              frame: frame - boxDelay,
              fps,
              config: { damping: 200 },
            });
            const boxOpacity = interpolate(boxSpring, [0, 1], [0, 1]);
            const boxX = interpolate(boxSpring, [0, 1], [30, 0]);

            return (
              <div
                key={p.name}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                  opacity: boxOpacity,
                  transform: `translateX(${boxX}px)`,
                  width: 340,
                }}
              >
                <div
                  style={{
                    width: 52,
                    height: 40,
                    borderRadius: 6,
                    backgroundColor: `${p.color}20`,
                    border: `2px solid ${p.color}80`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 13,
                    fontWeight: 700,
                    color: p.color,
                    fontFamily: fontFamilyCode,
                    flexShrink: 0,
                  }}
                >
                  {p.icon}
                </div>
                <div
                  style={{
                    flex: 1,
                    height: 3,
                    backgroundColor: `${p.color}40`,
                    borderRadius: 2,
                  }}
                />
                <div
                  style={{
                    fontSize: 14,
                    color: p.color,
                    fontFamily: fontFamilyCode,
                    fontWeight: 600,
                  }}
                >
                  {p.name}
                </div>
              </div>
            );
          })}

          {/* Arrow down */}
          <div
            style={{
              opacity: composeOpacity,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 4,
            }}
          >
            <div
              style={{
                width: 3,
                height: 30,
                background: `linear-gradient(180deg, ${THEME.colors.nvidiaGreen}, ${THEME.colors.accentBlue})`,
                borderRadius: 2,
              }}
            />
            <div
              style={{
                width: 0,
                height: 0,
                borderLeft: "8px solid transparent",
                borderRight: "8px solid transparent",
                borderTop: `10px solid ${THEME.colors.accentBlue}`,
              }}
            />
          </div>

          {/* Compose result */}
          <div
            style={{
              padding: "16px 32px",
              backgroundColor: "rgba(79,195,247,0.10)",
              border: `2px solid ${THEME.colors.accentBlue}60`,
              borderRadius: 10,
              opacity: composeOpacity,
              textAlign: "center",
            }}
          >
            <div
              style={{
                fontSize: 18,
                fontWeight: 700,
                color: THEME.colors.accentBlue,
                fontFamily: fontFamilyBody,
                marginBottom: 4,
              }}
            >
              Compose Into
            </div>
            <div
              style={{
                fontSize: 14,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                lineHeight: 1.5,
              }}
            >
              Sorting, SpMV, Attention,
              <br />
              Beam Search, Top-K, Loss
            </div>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
