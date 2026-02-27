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
import { fontFamilyBody } from "../../../../styles/fonts";

type TipCard = {
  title: string;
  detail: string;
  type: "do" | "dont";
};

const doTips: TipCard[] = [
  {
    title: "Reuse handles across calls",
    detail: "Create once, use many times",
    type: "do",
  },
  {
    title: "Use streams",
    detail: "Pass stream to library calls for async",
    type: "do",
  },
  {
    title: "Let auto-tune pick algorithms",
    detail: "cuDNN finds the fastest",
    type: "do",
  },
];

const dontTips: TipCard[] = [
  {
    title: "Don't create/destroy handles per call",
    detail: "Huge overhead",
    type: "dont",
  },
  {
    title: "Don't forget workspace memory",
    detail: "cuDNN needs scratch space",
    type: "dont",
  },
  {
    title: "Don't assume row-major",
    detail: "cuBLAS is column-major",
    type: "dont",
  },
];

const TipCardComponent: React.FC<{
  tip: TipCard;
  delay: number;
  side: "left" | "right";
}> = ({ tip, delay, side }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const cardSpring = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });
  const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
  const cardX = interpolate(
    cardSpring,
    [0, 1],
    [side === "left" ? -20 : 20, 0]
  );

  const isGood = tip.type === "do";
  const color = isGood ? THEME.colors.nvidiaGreen : THEME.colors.accentRed;
  const icon = isGood ? "\u2713" : "\u2717";

  return (
    <div
      style={{
        padding: "16px 20px",
        backgroundColor: `${color}08`,
        borderLeft: `4px solid ${color}`,
        borderRadius: 10,
        opacity: cardOpacity,
        transform: `translateX(${cardX}px)`,
        marginBottom: 16,
      }}
    >
      <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
        <div
          style={{
            width: 26,
            height: 26,
            borderRadius: 13,
            backgroundColor: `${color}20`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 14,
            fontWeight: 700,
            color,
            fontFamily: fontFamilyBody,
            flexShrink: 0,
            marginTop: 2,
          }}
        >
          {icon}
        </div>
        <div>
          <div
            style={{
              fontSize: 18,
              fontWeight: 700,
              color: THEME.colors.textPrimary,
              fontFamily: fontFamilyBody,
              lineHeight: 1.4,
            }}
          >
            {tip.title}
          </div>
          <div
            style={{
              fontSize: 15,
              color: THEME.colors.textSecondary,
              fontFamily: fontFamilyBody,
              marginTop: 4,
              lineHeight: 1.4,
            }}
          >
            {tip.detail}
          </div>
        </div>
      </div>
    </div>
  );
};

export const M7S14_PracticalTips: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="gradient" moduleNumber={7}>
      <SlideTitle title="Practical Tips for Library Usage" />

      <div
        style={{
          display: "flex",
          gap: 40,
          flex: 1,
          width: 1776,
        }}
      >
        {/* Left column - DO */}
        <div style={{ flex: 1 }}>
          <div
            style={{
              fontSize: 16,
              fontWeight: 700,
              color: THEME.colors.nvidiaGreen,
              fontFamily: fontFamilyBody,
              marginBottom: 16,
              padding: "6px 14px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 6,
              display: "inline-block",
            }}
          >
            DO
          </div>
          {doTips.map((tip, i) => (
            <TipCardComponent
              key={tip.title}
              tip={tip}
              delay={1 * fps + i * 1.5 * fps}
              side="left"
            />
          ))}
        </div>

        {/* Divider */}
        <div
          style={{
            width: 2,
            backgroundColor: THEME.colors.textMuted + "30",
            borderRadius: 1,
            alignSelf: "stretch",
            marginTop: 10,
          }}
        />

        {/* Right column - DON'T */}
        <div style={{ flex: 1 }}>
          <div
            style={{
              fontSize: 16,
              fontWeight: 700,
              color: THEME.colors.accentRed,
              fontFamily: fontFamilyBody,
              marginBottom: 16,
              padding: "6px 14px",
              backgroundColor: "rgba(255,82,82,0.08)",
              borderRadius: 6,
              display: "inline-block",
            }}
          >
            DON'T
          </div>
          {dontTips.map((tip, i) => (
            <TipCardComponent
              key={tip.title}
              tip={tip}
              delay={1.5 * fps + i * 1.5 * fps}
              side="right"
            />
          ))}
        </div>
      </div>
    </SlideLayout>
  );
};
