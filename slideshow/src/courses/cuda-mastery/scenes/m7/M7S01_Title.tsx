import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
  AbsoluteFill,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideBackground } from "../../../../components/SlideBackground";
import { fontFamilyHeading, fontFamilyBody } from "../../../../styles/fonts";

interface LibBlock {
  label: string;
  color: string;
  width: number;
  x: number;
  row: number;
  delay: number;
}

const LIB_BLOCKS: LibBlock[] = [
  // Bottom row: CUDA Runtime (full width)
  { label: "CUDA Runtime", color: "#3a3a4a", width: 340, x: 0, row: 0, delay: 0 },
  // Middle row: three libraries side by side
  { label: "cuBLAS", color: THEME.colors.accentBlue, width: 106, x: 0, row: 1, delay: 0.4 },
  { label: "cuDNN", color: THEME.colors.nvidiaGreen, width: 106, x: 117, row: 1, delay: 0.7 },
  { label: "Thrust", color: THEME.colors.accentPurple, width: 106, x: 234, row: 1, delay: 1.0 },
  // Top row: Your ML Code
  { label: "Your ML Code", color: THEME.colors.nvidiaGreen, width: 340, x: 0, row: 2, delay: 1.4 },
];

const BLOCK_HEIGHT = 56;
const ROW_GAP = 12;

export const M7S01_Title: React.FC = () => {
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

  const greenSpring = spring({
    frame,
    fps,
    config: { damping: 200 },
    delay: 0.7 * fps,
  });
  const greenOpacity = interpolate(greenSpring, [0, 1], [0, 1]);
  const greenY = interpolate(greenSpring, [0, 1], [30, 0]);

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

  const stackOpacity = interpolate(
    frame,
    [1.2 * fps, 1.8 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <AbsoluteFill>
      <SlideBackground variant="accent" />

      {/* Animated library stack on right */}
      <div
        style={{
          position: "absolute",
          right: 100,
          top: 180,
          width: 400,
          height: 320,
          opacity: stackOpacity,
        }}
      >
        {LIB_BLOCKS.map((block, i) => {
          const blockSpring = spring({
            frame: frame - (2 + block.delay) * fps,
            fps,
            config: { damping: 120, stiffness: 80 },
          });
          const blockY = interpolate(blockSpring, [0, 1], [200, 0]);
          const blockOpacity = interpolate(blockSpring, [0, 1], [0, 1]);

          const rowFromBottom = block.row;
          const topPos =
            (2 - rowFromBottom) * (BLOCK_HEIGHT + ROW_GAP);

          const isTopBlock = block.row === 2;
          const labelColor = isTopBlock || block.color === "#3a3a4a"
            ? THEME.colors.textPrimary
            : "#000";

          return (
            <div
              key={`block-${i}`}
              style={{
                position: "absolute",
                left: 30 + block.x,
                top: topPos,
                width: block.width,
                height: BLOCK_HEIGHT,
                backgroundColor: block.color,
                borderRadius: 8,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 16,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
                color: labelColor,
                opacity: blockOpacity,
                transform: `translateY(${blockY}px)`,
                boxShadow: `0 4px 12px rgba(0,0,0,0.3)`,
              }}
            >
              {block.label}
            </div>
          );
        })}

        {/* Arrow label */}
        <div
          style={{
            position: "absolute",
            bottom: -60,
            left: 30,
            width: 340,
            textAlign: "center",
            opacity: interpolate(
              frame - 5 * fps,
              [0, 0.5 * fps],
              [0, 0.7],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          <div
            style={{
              fontSize: 14,
              color: THEME.colors.textSecondary,
              fontFamily: fontFamilyBody,
              letterSpacing: "2px",
            }}
          >
            NVIDIA SOFTWARE STACK
          </div>
        </div>
      </div>

      {/* Main content on left */}
      <div
        style={{
          position: "absolute",
          left: 100,
          top: "50%",
          transform: "translateY(-50%)",
          maxWidth: 800,
          width: 800,
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
            fontSize: 72,
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
          cuBLAS, cuDNN &
        </h1>
        <h1
          style={{
            fontSize: 72,
            fontWeight: 900,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyHeading,
            margin: 0,
            marginTop: 8,
            opacity: greenOpacity,
            transform: `translateY(${greenY}px)`,
            lineHeight: 1.1,
            letterSpacing: "-2px",
          }}
        >
          Libraries
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
            width: 600,
          }}
        >
          Standing on NVIDIA's Shoulders
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
            Module 7
          </div>
          <span
            style={{
              fontSize: 22,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
            }}
          >
            Don't reinvent the wheel
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
