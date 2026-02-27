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

interface AlgoCard {
  name: string;
  description: string;
  complexity: string;
  borderColor: string;
  row: number;
  col: number;
}

const ALGO_CARDS: AlgoCard[] = [
  {
    name: "sort",
    description: "Radix sort on GPU",
    complexity: "O(N)",
    borderColor: THEME.colors.accentBlue,
    row: 0,
    col: 0,
  },
  {
    name: "reduce",
    description: "Sum, min, max, custom op",
    complexity: "O(N)",
    borderColor: THEME.colors.nvidiaGreen,
    row: 0,
    col: 1,
  },
  {
    name: "scan",
    description: "Prefix sum (inclusive/exclusive)",
    complexity: "O(N)",
    borderColor: THEME.colors.accentPurple,
    row: 0,
    col: 2,
  },
  {
    name: "transform",
    description: "Element-wise map",
    complexity: "O(N)",
    borderColor: THEME.colors.accentOrange,
    row: 1,
    col: 0,
  },
  {
    name: "remove_if / copy_if",
    description: "Filter elements",
    complexity: "O(N)",
    borderColor: THEME.colors.accentCyan,
    row: 1,
    col: 1,
  },
  {
    name: "unique",
    description: "Remove consecutive duplicates",
    complexity: "O(N)",
    borderColor: THEME.colors.accentYellow,
    row: 1,
    col: 2,
  },
];

const CARD_WIDTH = 340;
const CARD_HEIGHT = 140;
const CARD_GAP_X = 32;
const CARD_GAP_Y = 24;

export const M7S09_ThrustAlgorithms: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Bottom performance comparison
  const perfOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={7}>
      <div style={{ width: 1776 }}>
        <SlideTitle
          title="Key Thrust Algorithms"
          subtitle="GPU-optimized parallel primitives"
        />

        {/* 2x3 grid */}
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: `${CARD_GAP_Y}px ${CARD_GAP_X}px`,
            marginTop: 8,
            width: 3 * CARD_WIDTH + 2 * CARD_GAP_X,
          }}
        >
          {ALGO_CARDS.map((card, i) => {
            const cardIndex = card.row * 3 + card.col;
            const cardDelay = (1.5 + cardIndex * 0.5) * fps;
            const cardSpring = spring({
              frame: frame - cardDelay,
              fps,
              config: { damping: 180, stiffness: 100 },
            });
            const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
            const cardY = interpolate(cardSpring, [0, 1], [30, 0]);
            const cardScale = interpolate(cardSpring, [0, 1], [0.95, 1]);

            return (
              <div
                key={`card-${i}`}
                style={{
                  width: CARD_WIDTH,
                  height: CARD_HEIGHT,
                  backgroundColor: "rgba(255,255,255,0.03)",
                  borderRadius: 12,
                  border: `2px solid ${card.borderColor}50`,
                  borderLeft: `4px solid ${card.borderColor}`,
                  padding: "18px 22px",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                  opacity: cardOpacity,
                  transform: `translateY(${cardY}px) scale(${cardScale})`,
                  boxShadow: `0 4px 16px rgba(0,0,0,0.15)`,
                }}
              >
                {/* Top: name + complexity */}
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "flex-start",
                    width: CARD_WIDTH - 48,
                  }}
                >
                  <div
                    style={{
                      fontSize: 20,
                      fontFamily: fontFamilyCode,
                      fontWeight: 700,
                      color: card.borderColor,
                    }}
                  >
                    {card.name}
                  </div>
                  <div
                    style={{
                      padding: "3px 10px",
                      backgroundColor: `${card.borderColor}15`,
                      border: `1px solid ${card.borderColor}40`,
                      borderRadius: 12,
                      fontSize: 13,
                      fontFamily: fontFamilyCode,
                      fontWeight: 600,
                      color: card.borderColor,
                    }}
                  >
                    {card.complexity}
                  </div>
                </div>

                {/* Description */}
                <div
                  style={{
                    fontSize: 16,
                    fontFamily: fontFamilyBody,
                    color: THEME.colors.textSecondary,
                    lineHeight: 1.4,
                    width: CARD_WIDTH - 48,
                  }}
                >
                  {card.description}
                </div>

                {/* Visual accent bar at bottom */}
                <div
                  style={{
                    width: "100%",
                    height: 3,
                    background: `linear-gradient(90deg, ${card.borderColor}, transparent)`,
                    borderRadius: 2,
                    opacity: 0.5,
                  }}
                />
              </div>
            );
          })}
        </div>

        {/* Performance comparison */}
        <div
          style={{
            marginTop: 28,
            display: "flex",
            alignItems: "center",
            gap: 24,
            opacity: perfOpacity,
            width: 1100,
          }}
        >
          <div
            style={{
              padding: "14px 24px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              display: "flex",
              alignItems: "center",
              gap: 16,
              width: 1060,
            }}
          >
            <span
              style={{
                fontSize: 16,
                fontFamily: fontFamilyBody,
                color: THEME.colors.textSecondary,
              }}
            >
              Performance:
            </span>

            {/* GPU bar */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <span
                style={{
                  fontSize: 14,
                  fontFamily: fontFamilyCode,
                  color: THEME.colors.nvidiaGreen,
                  fontWeight: 700,
                }}
              >
                thrust::sort
              </span>
              <div
                style={{
                  width: 60,
                  height: 16,
                  backgroundColor: THEME.colors.nvidiaGreen,
                  borderRadius: 4,
                }}
              />
              <span
                style={{
                  fontSize: 13,
                  fontFamily: fontFamilyCode,
                  color: THEME.colors.nvidiaGreen,
                }}
              >
                ~5ms
              </span>
            </div>

            <span
              style={{
                fontSize: 14,
                color: THEME.colors.textMuted,
                fontFamily: fontFamilyBody,
              }}
            >
              vs
            </span>

            {/* CPU bar */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <span
                style={{
                  fontSize: 14,
                  fontFamily: fontFamilyCode,
                  color: THEME.colors.accentRed,
                  fontWeight: 700,
                }}
              >
                std::sort
              </span>
              <div
                style={{
                  width: 280,
                  height: 16,
                  backgroundColor: THEME.colors.accentRed,
                  borderRadius: 4,
                }}
              />
              <span
                style={{
                  fontSize: 13,
                  fontFamily: fontFamilyCode,
                  color: THEME.colors.accentRed,
                }}
              >
                ~800ms
              </span>
            </div>

            {/* Speedup */}
            <div
              style={{
                padding: "4px 12px",
                backgroundColor: "rgba(118,185,0,0.2)",
                border: `2px solid ${THEME.colors.nvidiaGreen}`,
                borderRadius: 16,
                fontSize: 16,
                fontFamily: fontFamilyBody,
                fontWeight: 800,
                color: THEME.colors.nvidiaGreen,
                whiteSpace: "nowrap",
              }}
            >
              160x
            </div>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
