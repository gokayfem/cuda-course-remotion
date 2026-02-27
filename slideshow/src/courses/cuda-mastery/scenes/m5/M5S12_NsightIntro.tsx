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

type NsightCard = {
  title: string;
  icon: string;
  description: string;
  features: string[];
  command: string;
  color: string;
};

const cards: NsightCard[] = [
  {
    title: "Nsight Compute (ncu)",
    icon: "\uD83D\uDD0D",
    description: "Profile individual kernel launches",
    features: [
      "Detailed metrics per kernel",
      "Memory throughput, compute throughput, occupancy",
      "Roofline analysis built-in",
    ],
    command: "ncu --set full ./my_program",
    color: THEME.colors.accentBlue,
  },
  {
    title: "Nsight Systems (nsys)",
    icon: "\u2261",
    description: "Profile entire application timeline",
    features: [
      "CPU <-> GPU interactions, stream concurrency",
      "Identify idle gaps and bottlenecks",
      "Visual timeline of all operations",
    ],
    command: "nsys profile ./my_program",
    color: THEME.colors.accentPurple,
  },
];

const NsightCardComponent: React.FC<{
  card: NsightCard;
  index: number;
  delay: number;
}> = ({ card, index, delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const slideDir = index === 0 ? -1 : 1;
  const cardSpring = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });
  const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
  const cardX = interpolate(cardSpring, [0, 1], [60 * slideDir, 0]);

  return (
    <div
      style={{
        flex: 1,
        padding: "32px 36px",
        backgroundColor: "rgba(255,255,255,0.03)",
        borderRadius: 16,
        border: `1px solid ${card.color}30`,
        opacity: cardOpacity,
        transform: `translateX(${cardX}px)`,
        display: "flex",
        flexDirection: "column",
        width: 400,
      }}
    >
      {/* Icon + Title */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20 }}>
        <div
          style={{
            width: 56,
            height: 56,
            borderRadius: 14,
            backgroundColor: `${card.color}15`,
            border: `2px solid ${card.color}40`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 28,
            flexShrink: 0,
          }}
        >
          {card.icon}
        </div>
        <div
          style={{
            fontSize: 24,
            fontWeight: 700,
            color: card.color,
            fontFamily: fontFamilyBody,
            lineHeight: 1.3,
          }}
        >
          {card.title}
        </div>
      </div>

      {/* Description */}
      <div
        style={{
          fontSize: 18,
          color: THEME.colors.textPrimary,
          fontFamily: fontFamilyBody,
          marginBottom: 20,
          fontWeight: 600,
          lineHeight: 1.4,
        }}
      >
        {card.description}
      </div>

      {/* Features */}
      <div style={{ flex: 1 }}>
        {card.features.map((feat, i) => {
          const featDelay = delay + 0.8 * fps + i * 0.3 * fps;
          const featOpacity = interpolate(
            frame - featDelay,
            [0, 0.3 * fps],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          return (
            <div
              key={i}
              style={{
                display: "flex",
                alignItems: "flex-start",
                gap: 10,
                marginBottom: 12,
                opacity: featOpacity,
              }}
            >
              <span
                style={{
                  color: card.color,
                  fontSize: 16,
                  lineHeight: 1.5,
                  flexShrink: 0,
                }}
              >
                {"\u25B8"}
              </span>
              <span
                style={{
                  fontSize: 16,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  lineHeight: 1.5,
                }}
              >
                {feat}
              </span>
            </div>
          );
        })}
      </div>

      {/* Command */}
      <div
        style={{
          marginTop: 20,
          padding: "12px 18px",
          backgroundColor: "rgba(13,17,23,0.8)",
          borderRadius: 8,
          border: `1px solid rgba(255,255,255,0.08)`,
          opacity: interpolate(
            frame - (delay + 2.5 * fps),
            [0, 0.3 * fps],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          ),
        }}
      >
        <div
          style={{
            fontSize: 11,
            color: THEME.colors.textMuted,
            fontFamily: fontFamilyBody,
            marginBottom: 6,
            textTransform: "uppercase",
            letterSpacing: "1px",
          }}
        >
          Command
        </div>
        <div
          style={{
            fontSize: 17,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyCode,
            fontWeight: 600,
          }}
        >
          $ {card.command}
        </div>
      </div>
    </div>
  );
};

export const M5S12_NsightIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="gradient" moduleNumber={5}>
      <SlideTitle
        title="Profiling with NVIDIA Nsight"
        subtitle="Two complementary tools for GPU performance analysis"
      />

      <div
        style={{
          display: "flex",
          gap: 40,
          flex: 1,
          alignItems: "stretch",
          width: 1776,
        }}
      >
        {cards.map((card, i) => (
          <NsightCardComponent
            key={card.title}
            card={card}
            index={i}
            delay={1 * fps + i * 1.5 * fps}
          />
        ))}
      </div>
    </SlideLayout>
  );
};
