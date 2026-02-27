import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideBackground } from "../../../components/SlideBackground";
import { fontFamilyHeading, fontFamilyBody } from "../../../styles/fonts";

export const S01_TitleSlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // NVIDIA logo bar animation
  const barWidth = interpolate(frame, [0, 1 * fps], [0, 400], {
    extrapolateRight: "clamp",
  });

  // Title animation
  const titleSpring = spring({
    frame,
    fps,
    config: { damping: 200 },
    delay: 0.3 * fps,
  });
  const titleOpacity = interpolate(titleSpring, [0, 1], [0, 1]);
  const titleY = interpolate(titleSpring, [0, 1], [40, 0]);

  // Subtitle
  const subtitleOpacity = interpolate(
    frame,
    [1 * fps, 1.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Module info
  const moduleOpacity = interpolate(
    frame,
    [1.5 * fps, 2 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Floating GPU grid effect
  const gridOpacity = interpolate(
    frame,
    [0.5 * fps, 1.5 * fps],
    [0, 0.15],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <AbsoluteFill>
      <SlideBackground variant="accent" />

      {/* Decorative GPU grid */}
      <div
        style={{
          position: "absolute",
          right: 80,
          top: 100,
          opacity: gridOpacity,
          display: "grid",
          gridTemplateColumns: "repeat(12, 30px)",
          gap: 4,
        }}
      >
        {Array.from({ length: 96 }).map((_, i) => {
          const cellSpring = spring({
            frame: frame - i * 0.5,
            fps,
            config: { damping: 200 },
          });
          return (
            <div
              key={i}
              style={{
                width: 30,
                height: 30,
                backgroundColor: THEME.colors.nvidiaGreen,
                borderRadius: 3,
                opacity: interpolate(cellSpring, [0, 1], [0, 0.6]),
              }}
            />
          );
        })}
      </div>

      {/* Main content */}
      <div
        style={{
          position: "absolute",
          left: 100,
          top: "50%",
          transform: "translateY(-50%)",
          maxWidth: 1000,
        }}
      >
        {/* Green accent bar */}
        <div
          style={{
            width: barWidth,
            height: 6,
            backgroundColor: THEME.colors.nvidiaGreen,
            borderRadius: 3,
            marginBottom: 32,
          }}
        />

        {/* Title */}
        <h1
          style={{
            fontSize: 80,
            fontWeight: 900,
            color: THEME.colors.textPrimary,
            fontFamily: fontFamilyHeading,
            margin: 0,
            opacity: titleOpacity,
            transform: `translateY(${titleY}px)`,
            lineHeight: 1.1,
            letterSpacing: "-2px",
          }}
        >
          CUDA
          <br />
          <span style={{ color: THEME.colors.nvidiaGreen }}>Mastery</span>
        </h1>

        {/* Subtitle */}
        <p
          style={{
            fontSize: 32,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
            margin: 0,
            marginTop: 24,
            opacity: subtitleOpacity,
            fontWeight: 400,
          }}
        >
          GPU Programming for ML Engineers
        </p>

        {/* Module badge */}
        <div
          style={{
            marginTop: 48,
            display: "flex",
            gap: 16,
            alignItems: "center",
            opacity: moduleOpacity,
          }}
        >
          <div
            style={{
              padding: "12px 28px",
              backgroundColor: "rgba(118,185,0,0.15)",
              border: `2px solid ${THEME.colors.nvidiaGreen}`,
              borderRadius: 30,
              fontSize: 22,
              color: THEME.colors.nvidiaGreen,
              fontFamily: fontFamilyBody,
              fontWeight: 700,
            }}
          >
            Module 1
          </div>
          <span
            style={{
              fontSize: 22,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
            }}
          >
            GPU Architecture & Your First Kernel
          </span>
        </div>
      </div>

      {/* Bottom gradient bar */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          height: 6,
          background: `linear-gradient(90deg, ${THEME.colors.nvidiaGreen}, ${THEME.colors.accentBlue}, ${THEME.colors.accentPurple})`,
        }}
      />
    </AbsoluteFill>
  );
};
