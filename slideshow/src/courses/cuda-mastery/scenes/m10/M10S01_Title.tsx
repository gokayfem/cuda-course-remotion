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

interface NodeDef {
  readonly x: number;
  readonly y: number;
  readonly label: string;
  readonly color: string;
  readonly delay: number;
}

const NODES: readonly NodeDef[] = [
  { x: 120, y: 80, label: "Tensor Core", color: THEME.colors.nvidiaGreen, delay: 0 },
  { x: 300, y: 160, label: "FP16", color: THEME.colors.accentBlue, delay: 0.3 },
  { x: 80, y: 260, label: "CUTLASS", color: THEME.colors.accentPurple, delay: 0.6 },
  { x: 280, y: 340, label: "Triton", color: THEME.colors.accentOrange, delay: 0.9 },
  { x: 140, y: 420, label: "PyTorch", color: THEME.colors.accentCyan, delay: 1.2 },
  { x: 320, y: 60, label: "BF16", color: THEME.colors.accentYellow, delay: 0.45 },
  { x: 60, y: 160, label: "WMMA", color: THEME.colors.accentRed, delay: 0.75 },
  { x: 340, y: 250, label: "FP8", color: THEME.colors.accentPink, delay: 1.05 },
];

const CONNECTIONS: readonly (readonly [number, number])[] = [
  [0, 1], [0, 2], [1, 3], [2, 4], [3, 4],
  [0, 5], [6, 0], [5, 7], [7, 3], [6, 2],
  [1, 7], [4, 3],
];

export const M10S01_Title: React.FC = () => {
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

  const nodeAreaX = 1380;
  const nodeAreaY = 160;

  return (
    <AbsoluteFill>
      <SlideBackground variant="accent" />

      {/* Circuit-board pattern on right */}
      <div
        style={{
          position: "absolute",
          right: 80,
          top: nodeAreaY,
          width: 440,
          height: 500,
        }}
      >
        {/* Connection lines */}
        <svg
          width={440}
          height={500}
          style={{ position: "absolute", top: 0, left: 0 }}
        >
          {CONNECTIONS.map(([fromIdx, toIdx], i) => {
            const from = NODES[fromIdx];
            const to = NODES[toIdx];
            const connDelay = Math.max(from.delay, to.delay) * fps + 0.5 * fps;
            const connOpacity = interpolate(
              frame - connDelay,
              [0, 0.4 * fps],
              [0, 0.35],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );

            const pulsePhase = (frame - connDelay) / (2 * fps);
            const pulse = Math.sin(pulsePhase * Math.PI * 2 + i * 0.8) * 0.15;

            return (
              <line
                key={`${fromIdx}-${toIdx}`}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke={THEME.colors.nvidiaGreen}
                strokeWidth={1.5}
                opacity={Math.max(0, connOpacity + pulse)}
              />
            );
          })}
        </svg>

        {/* Nodes */}
        {NODES.map((node, i) => {
          const nodeDelay = node.delay * fps + 0.8 * fps;
          const nodeSpring = spring({
            frame: frame - nodeDelay,
            fps,
            config: { damping: 200 },
          });
          const nodeOpacity = interpolate(nodeSpring, [0, 1], [0, 1]);
          const nodeScale = interpolate(nodeSpring, [0, 1], [0.5, 1]);

          const glowPhase = (frame - nodeDelay) / (1.8 * fps);
          const glowIntensity = Math.sin(glowPhase * Math.PI * 2 + i * 1.2) * 0.3 + 0.5;

          return (
            <div
              key={node.label}
              style={{
                position: "absolute",
                left: node.x - 46,
                top: node.y - 18,
                width: 92,
                height: 36,
                backgroundColor: `${node.color}20`,
                border: `1.5px solid ${node.color}80`,
                borderRadius: 18,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                opacity: nodeOpacity,
                transform: `scale(${nodeScale})`,
                boxShadow: `0 0 ${8 + glowIntensity * 12}px ${node.color}${Math.round(glowIntensity * 60).toString(16).padStart(2, "0")}`,
              }}
            >
              <span
                style={{
                  fontSize: 12,
                  fontWeight: 700,
                  color: node.color,
                  fontFamily: fontFamilyBody,
                  whiteSpace: "nowrap",
                }}
              >
                {node.label}
              </span>
            </div>
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
          Advanced{" "}
          <span style={{ color: THEME.colors.nvidiaGreen }}>Topics</span>
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
          Tensor Cores, Mixed Precision, CUTLASS, Triton & PyTorch Extensions
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
            Module 10
          </div>
          <span
            style={{
              fontSize: 20,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
            }}
          >
            Final Module
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
