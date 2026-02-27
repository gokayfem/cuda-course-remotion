import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideBackground } from "../../../../components/SlideBackground";
import { fontFamilyHeading, fontFamilyBody } from "../../../../styles/fonts";

export const M2S01_Title: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const barWidth = interpolate(frame, [0, 1 * fps], [0, 400], {
    extrapolateRight: "clamp",
  });

  const titleSpring = spring({ frame, fps, config: { damping: 200 }, delay: 0.3 * fps });
  const titleOpacity = interpolate(titleSpring, [0, 1], [0, 1]);
  const titleY = interpolate(titleSpring, [0, 1], [40, 0]);

  const subtitleOpacity = interpolate(frame, [1 * fps, 1.5 * fps], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });

  const moduleOpacity = interpolate(frame, [1.5 * fps, 2 * fps], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });

  // Memory hierarchy layers animation
  const layers = [
    { label: "Registers", color: THEME.colors.accentRed, width: 160 },
    { label: "Shared Memory", color: THEME.colors.accentOrange, width: 280 },
    { label: "L1 / L2 Cache", color: THEME.colors.accentYellow, width: 400 },
    { label: "Global Memory (HBM)", color: THEME.colors.nvidiaGreen, width: 540 },
  ];

  return (
    <AbsoluteFill>
      <SlideBackground variant="accent" />

      {/* Memory hierarchy pyramid on right */}
      <div
        style={{
          position: "absolute",
          right: 100,
          top: 200,
          width: 580,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 14,
        }}
      >
        {layers.map((layer, i) => {
          const layerDelay = 1 * fps + i * 0.3 * fps;
          const layerSpring = spring({ frame: frame - layerDelay, fps, config: { damping: 200 } });
          return (
            <div
              key={layer.label}
              style={{
                width: layer.width,
                height: 48,
                backgroundColor: `${layer.color}20`,
                border: `2px solid ${layer.color}60`,
                borderRadius: 8,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 18,
                color: layer.color,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                opacity: interpolate(layerSpring, [0, 1], [0, 1]),
                transform: `scale(${interpolate(layerSpring, [0, 1], [0.8, 1])})`,
              }}
            >
              {layer.label}
            </div>
          );
        })}
        <div
          style={{
            marginTop: 8,
            fontSize: 16,
            color: THEME.colors.textMuted,
            fontFamily: fontFamilyBody,
            textAlign: "center",
            opacity: interpolate(frame - 2.5 * fps, [0, 0.5 * fps], [0, 1], {
              extrapolateLeft: "clamp", extrapolateRight: "clamp",
            }),
          }}
        >
          Faster & Smaller ↑ &nbsp;&nbsp;&nbsp; ↓ Slower & Larger
        </div>
      </div>

      {/* Main content */}
      <div style={{ position: "absolute", left: 100, top: "50%", transform: "translateY(-50%)", maxWidth: 900 }}>
        <div style={{ width: barWidth, height: 6, backgroundColor: THEME.colors.accentOrange, borderRadius: 3, marginBottom: 32 }} />

        <h1 style={{
          fontSize: 76, fontWeight: 900, color: THEME.colors.textPrimary,
          fontFamily: fontFamilyHeading, margin: 0, opacity: titleOpacity,
          transform: `translateY(${titleY}px)`, lineHeight: 1.1, letterSpacing: "-2px",
        }}>
          Memory<br />
          <span style={{ color: THEME.colors.accentOrange }}>Hierarchy</span>
        </h1>

        <p style={{
          fontSize: 30, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody,
          margin: 0, marginTop: 24, opacity: subtitleOpacity, fontWeight: 400,
        }}>
          The key to 10x-100x GPU performance
        </p>

        <div style={{ marginTop: 48, display: "flex", gap: 16, alignItems: "center", opacity: moduleOpacity }}>
          <div style={{
            padding: "12px 28px", backgroundColor: "rgba(255,171,64,0.15)",
            border: `2px solid ${THEME.colors.accentOrange}`, borderRadius: 30,
            fontSize: 22, color: THEME.colors.accentOrange, fontFamily: fontFamilyBody, fontWeight: 700,
          }}>
            Module 2
          </div>
          <span style={{ fontSize: 22, color: THEME.colors.textMuted, fontFamily: fontFamilyBody }}>
            Global, Shared, Registers, Constant & Texture
          </span>
        </div>
      </div>

      <div style={{
        position: "absolute", bottom: 0, left: 0, right: 0, height: 6,
        background: `linear-gradient(90deg, ${THEME.colors.accentOrange}, ${THEME.colors.accentRed}, ${THEME.colors.accentPurple})`,
      }} />
    </AbsoluteFill>
  );
};
