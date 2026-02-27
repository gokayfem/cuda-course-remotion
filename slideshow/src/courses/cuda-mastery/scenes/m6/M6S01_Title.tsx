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

const STREAM_COLORS = {
  h2d: THEME.colors.accentBlue,
  kernel: THEME.colors.nvidiaGreen,
  d2h: THEME.colors.accentOrange,
};

const BLOCK_HEIGHT = 32;
const BLOCK_GAP = 12;
const STREAM_Y_START = 120;
const STREAM_SPACING = 52;

interface StreamBlock {
  type: "h2d" | "kernel" | "d2h";
  x: number;
  width: number;
  delay: number;
}

const streamData: StreamBlock[][] = [
  [
    { type: "h2d", x: 0, width: 70, delay: 0 },
    { type: "kernel", x: 80, width: 100, delay: 0.4 },
    { type: "d2h", x: 190, width: 70, delay: 0.8 },
  ],
  [
    { type: "h2d", x: 80, width: 70, delay: 0.6 },
    { type: "kernel", x: 160, width: 100, delay: 1.0 },
    { type: "d2h", x: 270, width: 70, delay: 1.4 },
  ],
  [
    { type: "h2d", x: 160, width: 70, delay: 1.2 },
    { type: "kernel", x: 240, width: 100, delay: 1.6 },
    { type: "d2h", x: 350, width: 70, delay: 2.0 },
  ],
];

export const M6S01_Title: React.FC = () => {
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

  const timelineOpacity = interpolate(
    frame,
    [1.2 * fps, 1.8 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <AbsoluteFill>
      <SlideBackground variant="accent" />

      {/* Animated stream timeline on right */}
      <div
        style={{
          position: "absolute",
          right: 80,
          top: 160,
          width: 500,
          height: 320,
          opacity: timelineOpacity,
        }}
      >
        {/* Stream labels */}
        {[0, 1, 2].map((i) => {
          const labelOpacity = interpolate(
            frame - (1.5 + i * 0.3) * fps,
            [0, 0.3 * fps],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          return (
            <div
              key={`label-${i}`}
              style={{
                position: "absolute",
                left: 0,
                top: STREAM_Y_START + i * STREAM_SPACING - 10,
                fontSize: 14,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                opacity: labelOpacity,
                width: 80,
              }}
            >
              Stream {i}
            </div>
          );
        })}

        {/* Stream timeline tracks */}
        {[0, 1, 2].map((i) => (
          <div
            key={`track-${i}`}
            style={{
              position: "absolute",
              left: 90,
              top: STREAM_Y_START + i * STREAM_SPACING,
              width: 380,
              height: BLOCK_HEIGHT,
              backgroundColor: "rgba(255,255,255,0.03)",
              borderRadius: 4,
            }}
          />
        ))}

        {/* Stream blocks */}
        {streamData.map((stream, si) =>
          stream.map((block, bi) => {
            const blockDelay = (2 + block.delay) * fps;
            const blockSpring = spring({
              frame: frame - blockDelay,
              fps,
              config: { damping: 120, stiffness: 100 },
            });
            const blockX = interpolate(blockSpring, [0, 1], [-40, 0]);
            const blockOpacity = interpolate(blockSpring, [0, 1], [0, 1]);

            return (
              <div
                key={`block-${si}-${bi}`}
                style={{
                  position: "absolute",
                  left: 90 + block.x,
                  top: STREAM_Y_START + si * STREAM_SPACING,
                  width: block.width,
                  height: BLOCK_HEIGHT,
                  backgroundColor: `${STREAM_COLORS[block.type]}cc`,
                  borderRadius: 5,
                  opacity: blockOpacity,
                  transform: `translateX(${blockX}px)`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 11,
                  fontFamily: fontFamilyBody,
                  fontWeight: 700,
                  color: THEME.colors.textPrimary,
                }}
              >
                {block.type === "h2d"
                  ? "H2D"
                  : block.type === "kernel"
                    ? "Kernel"
                    : "D2H"}
              </div>
            );
          })
        )}

        {/* Legend */}
        <div
          style={{
            position: "absolute",
            bottom: 0,
            left: 90,
            display: "flex",
            gap: 24,
            opacity: interpolate(
              frame - 4 * fps,
              [0, 0.5 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          {[
            { label: "Host to Device", color: STREAM_COLORS.h2d },
            { label: "Kernel", color: STREAM_COLORS.kernel },
            { label: "Device to Host", color: STREAM_COLORS.d2h },
          ].map((item) => (
            <div
              key={item.label}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <div
                style={{
                  width: 14,
                  height: 14,
                  borderRadius: 3,
                  backgroundColor: item.color,
                }}
              />
              <span
                style={{
                  fontSize: 12,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                }}
              >
                {item.label}
              </span>
            </div>
          ))}
        </div>

        {/* Time arrow */}
        <div
          style={{
            position: "absolute",
            left: 90,
            top: STREAM_Y_START + 3 * STREAM_SPACING - 8,
            width: 380,
            display: "flex",
            alignItems: "center",
            opacity: interpolate(
              frame - 3.5 * fps,
              [0, 0.3 * fps],
              [0, 0.5],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          <div
            style={{
              flex: 1,
              height: 1,
              backgroundColor: THEME.colors.textMuted,
            }}
          />
          <span
            style={{
              fontSize: 12,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
              marginLeft: 8,
            }}
          >
            time
          </span>
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
          Streams &
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
          Concurrency
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
          Async Transfers, Pipelines, Events & Multi-GPU
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
            Module 6
          </div>
          <span
            style={{
              fontSize: 22,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
            }}
          >
            Overlap everything, idle nothing
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
