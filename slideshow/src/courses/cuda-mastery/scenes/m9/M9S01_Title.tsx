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

const GRID_SIZE = 8;

const getAttentionWeight = (row: number, col: number): number => {
  const diagDist = Math.abs(row - col);
  const base = Math.exp(-diagDist * 0.6);
  const causal = col <= row ? 1.0 : 0.05;
  const noise = Math.sin(row * 3.7 + col * 2.3) * 0.15;
  return Math.max(0, Math.min(1, base * causal + noise));
};

export const M9S01_Title: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const barWidth = interpolate(frame, [0, 1 * fps], [0, 400], {
    extrapolateRight: "clamp",
  });

  const titleSpring = spring({
    frame,
    fps,
    config: { damping: 200 },
    delay: 0.3 * fps,
  });
  const titleOpacity = interpolate(titleSpring, [0, 1], [0, 1]);
  const titleY = interpolate(titleSpring, [0, 1], [40, 0]);

  const subtitleOpacity = interpolate(
    frame,
    [1 * fps, 1.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const moduleOpacity = interpolate(
    frame,
    [1.5 * fps, 2 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const cellSize = 44;
  const cellGap = 3;

  return (
    <AbsoluteFill>
      <SlideBackground variant="accent" />

      {/* Attention heatmap grid on right */}
      <div
        style={{
          position: "absolute",
          right: 120,
          top: 180,
          width: 420,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <div
          style={{
            fontSize: 16,
            color: THEME.colors.textMuted,
            fontFamily: fontFamilyBody,
            marginBottom: 12,
            opacity: interpolate(
              frame - 1.2 * fps,
              [0, 0.4 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          Attention Weights (8x8)
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${GRID_SIZE}, ${cellSize}px)`,
            gap: cellGap,
          }}
        >
          {Array.from({ length: GRID_SIZE * GRID_SIZE }).map((_, idx) => {
            const row = Math.floor(idx / GRID_SIZE);
            const col = idx % GRID_SIZE;
            const weight = getAttentionWeight(row, col);

            const cellDelay = 1.0 * fps + (row + col) * 2;
            const cellProgress = interpolate(
              frame - cellDelay,
              [0, 0.8 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );

            const intensity = weight * cellProgress;
            const r = Math.round(118 * intensity);
            const g = Math.round(185 * intensity);
            const b = Math.round(0 * intensity);

            return (
              <div
                key={idx}
                style={{
                  width: cellSize,
                  height: cellSize,
                  backgroundColor: `rgb(${r}, ${g}, ${b})`,
                  borderRadius: 4,
                  border: `1px solid rgba(118, 185, 0, ${0.15 + intensity * 0.3})`,
                  opacity: interpolate(
                    frame - cellDelay,
                    [0, 0.3 * fps],
                    [0, 1],
                    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                  ),
                }}
              />
            );
          })}
        </div>
        <div
          style={{
            marginTop: 12,
            fontSize: 13,
            color: THEME.colors.textMuted,
            fontFamily: fontFamilyBody,
            opacity: interpolate(
              frame - 3 * fps,
              [0, 0.5 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          Diagonal = self-attention, causal mask applied
        </div>
      </div>

      {/* Main content */}
      <div
        style={{
          position: "absolute",
          left: 100,
          top: "50%",
          transform: "translateY(-50%)",
          maxWidth: 900,
          width: 900,
        }}
      >
        <div
          style={{
            width: barWidth,
            height: 6,
            backgroundColor: THEME.colors.nvidiaGreen,
            borderRadius: 3,
            marginBottom: 32,
          }}
        />

        <h1
          style={{
            fontSize: 76,
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
          Attention &<br />
          <span style={{ color: THEME.colors.nvidiaGreen }}>
            Transformer Kernels
          </span>
        </h1>

        <p
          style={{
            fontSize: 28,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
            margin: 0,
            marginTop: 24,
            opacity: subtitleOpacity,
            fontWeight: 400,
          }}
        >
          Softmax, LayerNorm, Flash Attention & Kernel Fusion
        </p>

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
            Module 9
          </div>
          <span
            style={{
              fontSize: 20,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
            }}
          >
            Building the core kernels behind modern LLMs
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
