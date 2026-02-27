import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { fontFamilyHeading, fontFamilyBody } from "../../../../styles/fonts";

type MasteryCard = {
  title: string;
  modules: string;
  color: string;
};

const masteryCards: MasteryCard[] = [
  {
    title: "GPU Architecture & Memory",
    modules: "Modules 1-2",
    color: THEME.colors.accentBlue,
  },
  {
    title: "Thread Execution & Patterns",
    modules: "Modules 3-4",
    color: THEME.colors.accentCyan,
  },
  {
    title: "Performance & Concurrency",
    modules: "Modules 5-6",
    color: THEME.colors.nvidiaGreen,
  },
  {
    title: "Libraries & MatMul",
    modules: "Modules 7-8",
    color: THEME.colors.accentOrange,
  },
  {
    title: "Transformers & Attention",
    modules: "Module 9",
    color: THEME.colors.accentPurple,
  },
  {
    title: "Advanced: Tensor Cores, Triton",
    modules: "Module 10",
    color: THEME.colors.accentYellow,
  },
];

export const M10S18_Summary: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Congratulations text animation
  const congratsSpring = spring({
    frame: frame - 0.5 * fps,
    fps,
    config: { damping: 120, stiffness: 80 },
  });
  const congratsScale = interpolate(congratsSpring, [0, 1], [0.5, 1]);
  const congratsOpacity = interpolate(congratsSpring, [0, 1], [0, 1]);

  // Subtitle animation
  const subtitleOpacity = interpolate(
    frame - 1.5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Bottom bar glow animation
  const glowPulse = interpolate(
    Math.sin(frame * 0.06),
    [-1, 1],
    [0.5, 1]
  );

  const bottomBarOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="accent" moduleNumber={10}>
      {/* Congratulations header */}
      <div
        style={{
          textAlign: "center",
          marginBottom: 12,
          flexShrink: 0,
        }}
      >
        <div
          style={{
            fontSize: 64,
            fontWeight: 900,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyHeading,
            opacity: congratsOpacity,
            transform: `scale(${congratsScale})`,
            letterSpacing: "-1px",
            textShadow: `0 0 40px ${THEME.colors.nvidiaGreen}40`,
          }}
        >
          Congratulations!
        </div>
        <div
          style={{
            fontSize: 20,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
            marginTop: 8,
            opacity: subtitleOpacity,
          }}
        >
          Course Complete -- CUDA Mastery Achieved!
        </div>
        <div
          style={{
            fontSize: 16,
            color: THEME.colors.textMuted,
            fontFamily: fontFamilyBody,
            marginTop: 6,
            opacity: subtitleOpacity,
          }}
        >
          You've completed all 10 modules of CUDA Mastery
        </div>
      </div>

      {/* 2x3 mastery grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: 16,
          flex: 1,
          width: 1776,
          alignContent: "center",
        }}
      >
        {masteryCards.map((card, i) => {
          const cardDelay = 2 * fps + i * 0.4 * fps;
          const cardSpring = spring({
            frame: frame - cardDelay,
            fps,
            config: { damping: 200 },
          });
          const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
          const cardScale = interpolate(cardSpring, [0, 1], [0.88, 1]);

          return (
            <div
              key={card.title}
              style={{
                padding: "22px 26px",
                backgroundColor: `${card.color}08`,
                borderLeft: `4px solid ${card.color}`,
                borderRadius: 10,
                opacity: cardOpacity,
                transform: `scale(${cardScale})`,
                display: "flex",
                flexDirection: "column",
                gap: 8,
              }}
            >
              <div
                style={{
                  fontSize: 13,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyBody,
                  letterSpacing: "0.5px",
                  textTransform: "uppercase" as const,
                }}
              >
                {card.modules}
              </div>
              <div
                style={{
                  fontSize: 19,
                  fontWeight: 700,
                  color: card.color,
                  fontFamily: fontFamilyBody,
                  lineHeight: 1.3,
                }}
              >
                {card.title}
              </div>
              {/* Checkmark */}
              <div
                style={{
                  fontSize: 18,
                  color: THEME.colors.nvidiaGreen,
                  marginTop: 4,
                }}
              >
                {"\u2713"} Mastered
              </div>
            </div>
          );
        })}
      </div>

      {/* Bottom: glowing green bar with message */}
      <div
        style={{
          marginTop: 16,
          width: 1776,
          opacity: bottomBarOpacity,
        }}
      >
        <div
          style={{
            padding: "18px 32px",
            background: `linear-gradient(135deg, ${THEME.colors.nvidiaGreen}18, ${THEME.colors.accentBlue}10)`,
            borderRadius: 14,
            border: `2px solid ${THEME.colors.nvidiaGreen}50`,
            textAlign: "center",
            position: "relative",
            overflow: "hidden",
          }}
        >
          {/* Animated glow overlay */}
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: `linear-gradient(90deg, transparent, ${THEME.colors.nvidiaGreen}${Math.round(glowPulse * 15).toString(16).padStart(2, "0")}, transparent)`,
              borderRadius: 14,
            }}
          />
          <div
            style={{
              fontSize: 26,
              fontWeight: 800,
              color: THEME.colors.nvidiaGreen,
              fontFamily: fontFamilyHeading,
              position: "relative",
              textShadow: `0 0 ${20 * glowPulse}px ${THEME.colors.nvidiaGreen}60`,
            }}
          >
            Now go build something amazing on GPUs!
          </div>
        </div>

        {/* Glowing green bottom bar */}
        <div
          style={{
            marginTop: 12,
            height: 6,
            borderRadius: 3,
            background: `linear-gradient(90deg, ${THEME.colors.nvidiaGreen}00, ${THEME.colors.nvidiaGreen}, ${THEME.colors.accentBlue}, ${THEME.colors.nvidiaGreen}00)`,
            boxShadow: `0 0 ${16 * glowPulse}px ${THEME.colors.nvidiaGreen}80`,
            opacity: glowPulse,
          }}
        />
      </div>
    </SlideLayout>
  );
};
