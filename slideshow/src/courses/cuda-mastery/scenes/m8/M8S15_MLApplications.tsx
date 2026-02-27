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

type MLCard = {
  title: string;
  color: string;
  lines: string[];
};

const cards: MLCard[] = [
  {
    title: "Transformer Attention",
    color: THEME.colors.accentBlue,
    lines: [
      "Q x K^T: (seq x d) x (d x seq) = (seq x seq)",
      "Scores x V: (seq x seq) x (seq x d) = (seq x d)",
      "Batched GEMM across heads",
    ],
  },
  {
    title: "MLP / Feed-Forward",
    color: THEME.colors.nvidiaGreen,
    lines: [
      "Hidden = Input x W1 (up-project)",
      "Output = ReLU(Hidden) x W2 (down-project)",
      "Largest compute cost in transformers",
    ],
  },
  {
    title: "Convolution",
    color: THEME.colors.accentPurple,
    lines: [
      "im2col reshapes patches into matrix",
      "Conv = im2col_output x filters (GEMM)",
      "cuDNN's GEMM algo = this approach",
    ],
  },
];

export const M8S15_MLApplications: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="gradient" moduleNumber={8}>
      <SlideTitle
        title="MatMul in Real ML Workloads"
        subtitle="Matrix multiplication is everywhere in deep learning"
      />

      <div
        style={{
          display: "flex",
          gap: 24,
          flex: 1,
          width: 1776,
        }}
      >
        {cards.map((card, cardIdx) => {
          const cardDelay = 1.5 * fps + cardIdx * 1.5 * fps;
          const cardSpring = spring({
            frame: frame - cardDelay,
            fps,
            config: { damping: 200 },
          });
          const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
          const cardY = interpolate(cardSpring, [0, 1], [30, 0]);

          return (
            <div
              key={card.title}
              style={{
                flex: 1,
                padding: "24px 28px",
                backgroundColor: `${card.color}08`,
                border: `2px solid ${card.color}40`,
                borderRadius: 12,
                opacity: cardOpacity,
                transform: `translateY(${cardY}px)`,
                display: "flex",
                flexDirection: "column",
              }}
            >
              {/* Card title */}
              <div
                style={{
                  fontSize: 22,
                  fontWeight: 700,
                  color: card.color,
                  fontFamily: fontFamilyBody,
                  marginBottom: 20,
                  paddingBottom: 12,
                  borderBottom: `1px solid ${card.color}30`,
                }}
              >
                {card.title}
              </div>

              {/* Lines */}
              {card.lines.map((line, lineIdx) => {
                const lineDelay =
                  cardDelay + 0.6 * fps + lineIdx * 0.5 * fps;
                const lineSpring = spring({
                  frame: frame - lineDelay,
                  fps,
                  config: { damping: 200 },
                });
                const lineOpacity = interpolate(
                  lineSpring,
                  [0, 1],
                  [0, 1]
                );
                const lineX = interpolate(lineSpring, [0, 1], [-10, 0]);

                const isFormula = line.includes("x") && line.includes("=");

                return (
                  <div
                    key={lineIdx}
                    style={{
                      display: "flex",
                      alignItems: "flex-start",
                      gap: 10,
                      marginBottom: 14,
                      opacity: lineOpacity,
                      transform: `translateX(${lineX}px)`,
                    }}
                  >
                    <span
                      style={{
                        color: card.color,
                        fontSize: 16,
                        lineHeight: 1.6,
                        flexShrink: 0,
                      }}
                    >
                      {"\u25B8"}
                    </span>
                    <span
                      style={{
                        fontSize: isFormula ? 15 : 17,
                        color: isFormula
                          ? THEME.colors.textPrimary
                          : THEME.colors.textSecondary,
                        fontFamily: isFormula
                          ? fontFamilyCode
                          : fontFamilyBody,
                        lineHeight: 1.6,
                        fontWeight: lineIdx === card.lines.length - 1 ? 600 : 400,
                      }}
                    >
                      {line}
                    </span>
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </SlideLayout>
  );
};
