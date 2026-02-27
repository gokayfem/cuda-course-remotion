import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../styles/theme";
import { fontFamilyHeading, fontFamilyBody } from "../styles/fonts";

export const FadeInText: React.FC<{
  text: string;
  delay?: number;
  fontSize?: number;
  color?: string;
  fontWeight?: number;
  style?: React.CSSProperties;
}> = ({
  text,
  delay = 0,
  fontSize = 24,
  color = THEME.colors.textPrimary,
  fontWeight = 400,
  style,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const opacity = interpolate(frame - delay, [0, 0.5 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const translateY = interpolate(frame - delay, [0, 0.5 * fps], [14, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        opacity,
        transform: `translateY(${translateY}px)`,
        fontSize,
        color,
        fontWeight,
        fontFamily: fontFamilyBody,
        lineHeight: 1.5,
        ...style,
      }}
    >
      {text}
    </div>
  );
};

export const SlideTitle: React.FC<{
  title: string;
  subtitle?: string;
  delay?: number;
}> = ({ title, subtitle, delay = 0 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleSpring = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });

  const titleOpacity = interpolate(titleSpring, [0, 1], [0, 1]);
  const titleX = interpolate(titleSpring, [0, 1], [-30, 0]);

  const subtitleOpacity = interpolate(
    frame - delay,
    [0.3 * fps, 0.8 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const lineWidth = interpolate(
    frame - delay,
    [0.2 * fps, 0.7 * fps],
    [0, 260],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <div style={{ marginBottom: 36, flexShrink: 0 }}>
      <h1
        style={{
          fontSize: 52,
          fontWeight: 800,
          color: THEME.colors.textPrimary,
          fontFamily: fontFamilyHeading,
          margin: 0,
          opacity: titleOpacity,
          transform: `translateX(${titleX}px)`,
          letterSpacing: "-0.5px",
          lineHeight: 1.15,
        }}
      >
        {title}
      </h1>
      <div
        style={{
          height: 3,
          width: lineWidth,
          background: `linear-gradient(90deg, ${THEME.colors.nvidiaGreen}, ${THEME.colors.accentBlue})`,
          borderRadius: 2,
          marginTop: 14,
          marginBottom: 12,
        }}
      />
      {subtitle && (
        <p
          style={{
            fontSize: 22,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
            margin: 0,
            opacity: subtitleOpacity,
            fontWeight: 400,
            lineHeight: 1.4,
          }}
        >
          {subtitle}
        </p>
      )}
    </div>
  );
};

export const BulletPoint: React.FC<{
  text: string;
  index: number;
  delay?: number;
  icon?: string;
  highlight?: boolean;
  subtext?: string;
}> = ({ text, index, delay = 0, icon = "▸", highlight = false, subtext }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const itemDelay = delay + index * 0.25 * fps;

  const itemSpring = spring({
    frame: frame - itemDelay,
    fps,
    config: { damping: 200 },
  });

  const opacity = interpolate(itemSpring, [0, 1], [0, 1]);
  const translateX = interpolate(itemSpring, [0, 1], [-20, 0]);

  return (
    <div
      style={{
        opacity,
        transform: `translateX(${translateX}px)`,
        display: "flex",
        alignItems: "flex-start",
        gap: 12,
        marginBottom: subtext ? 18 : 12,
      }}
    >
      <span
        style={{
          color: highlight ? THEME.colors.nvidiaGreen : THEME.colors.accentBlue,
          fontSize: 22,
          fontWeight: 700,
          lineHeight: 1.55,
          flexShrink: 0,
          width: 18,
          textAlign: "center",
        }}
      >
        {icon}
      </span>
      <div>
        <span
          style={{
            fontSize: 22,
            color: highlight
              ? THEME.colors.nvidiaGreen
              : THEME.colors.textPrimary,
            fontFamily: fontFamilyBody,
            fontWeight: highlight ? 600 : 400,
            lineHeight: 1.55,
          }}
        >
          {text}
        </span>
        {subtext && (
          <div
            style={{
              fontSize: 17,
              color: THEME.colors.textSecondary,
              fontFamily: fontFamilyBody,
              marginTop: 3,
              lineHeight: 1.45,
            }}
          >
            {subtext}
          </div>
        )}
      </div>
    </div>
  );
};
