import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { THEME } from "../styles/theme";

export const SlideBackground: React.FC<{
  variant?: "dark" | "gradient" | "code" | "accent";
}> = ({ variant = "dark" }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const gradientShift = interpolate(frame, [0, 10 * fps], [0, 30], {
    extrapolateRight: "extend",
  });

  const backgrounds: Record<string, React.CSSProperties> = {
    dark: {
      background: `radial-gradient(ellipse at 20% 50%, #1a1a2e 0%, ${THEME.colors.bgPrimary} 70%)`,
    },
    gradient: {
      background: `linear-gradient(${135 + gradientShift}deg, #0a0a0a 0%, #1a1a2e 40%, #16213e 100%)`,
    },
    code: {
      background: THEME.colors.bgCode,
    },
    accent: {
      background: `radial-gradient(ellipse at 70% 30%, rgba(118,185,0,0.15) 0%, ${THEME.colors.bgPrimary} 60%)`,
    },
  };

  return (
    <AbsoluteFill style={backgrounds[variant]}>
      {/* Subtle grid pattern */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)
          `,
          backgroundSize: "60px 60px",
          opacity: 0.5,
        }}
      />
    </AbsoluteFill>
  );
};
