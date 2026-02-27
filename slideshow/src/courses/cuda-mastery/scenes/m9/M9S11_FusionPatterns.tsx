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

type FusionPattern = {
  title: string;
  color: string;
  formula: string;
  description: string;
};

const patterns: FusionPattern[] = [
  {
    title: "Bias + GELU",
    color: THEME.colors.accentBlue,
    formula: "y = gelu(x + bias)",
    description: "Saves 1 global memory round-trip",
  },
  {
    title: "Residual + LayerNorm",
    color: THEME.colors.nvidiaGreen,
    formula: "y = layernorm(x + residual)",
    description: "Saves 1 read + 1 write",
  },
  {
    title: "Scale + Mask + Softmax",
    color: THEME.colors.accentPurple,
    formula: "y = softmax(x * scale + mask)",
    description: "Critical for attention",
  },
  {
    title: "Fused MHA",
    color: THEME.colors.accentOrange,
    formula: "Q/K/V proj + attention + output",
    description: "Flash Attention does this",
  },
];

export const M9S11_FusionPatterns: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="gradient" moduleNumber={9}>
      <SlideTitle title="Common Fusion Patterns" />

      {/* 2x2 grid */}
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
          const row = Math.floor(i / 2);
          const col = i % 2;
          const staggerDelay = 1 * fps + (row * 2 + col) * 0.5 * fps;

          const cardSpring = spring({
            frame: frame - staggerDelay,
            fps,
            config: { damping: 200 },
          });
          const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
          const cardScale = interpolate(cardSpring, [0, 1], [0.92, 1]);

          return (
            <div
              key={pattern.title}
              style={{
                padding: "28px 32px",
                backgroundColor: `${pattern.color}08`,
                borderLeft: `4px solid ${pattern.color}`,
                borderRadius: 12,
                opacity: cardOpacity,
                transform: `scale(${cardScale})`,
                display: "flex",
                flexDirection: "column",
                gap: 14,
              }}
            >
              {/* Title */}
              <div
                style={{
                  fontSize: 22,
                  fontWeight: 700,
                  color: pattern.color,
                  fontFamily: fontFamilyBody,
                }}
              >
                {pattern.title}
              </div>

              {/* Formula */}
              <div
                style={{
                  padding: "10px 16px",
                  backgroundColor: "rgba(13,17,23,0.6)",
                  borderRadius: 8,
                  fontSize: 17,
                  color: THEME.colors.textCode,
                  fontFamily: fontFamilyCode,
                  fontWeight: 600,
                  letterSpacing: "0.3px",
                }}
              >
                {pattern.formula}
              </div>

              {/* Description */}
              <div
                style={{
                  fontSize: 16,
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
