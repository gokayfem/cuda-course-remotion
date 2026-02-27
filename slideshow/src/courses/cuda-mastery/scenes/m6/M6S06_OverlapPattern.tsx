import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode, fontFamilyHeading } from "../../../../styles/fonts";

const BLOCK_H = 32;
const BLOCK_RADIUS = 5;
const TRACK_Y = 150;
const TRACK_SPACING = 48;
const TRACK_LEFT = 140;

const COLORS = {
  h2d: THEME.colors.accentBlue,
  kernel: THEME.colors.nvidiaGreen,
  d2h: THEME.colors.accentOrange,
};

interface PipeBlock {
  type: "h2d" | "kernel" | "d2h";
  x: number;
  width: number;
  delay: number;
}

const pipelineStreams: { label: string; blocks: PipeBlock[] }[] = [
  {
    label: "Stream 0",
    blocks: [
      { type: "h2d", x: 0, width: 80, delay: 2 },
      { type: "kernel", x: 88, width: 110, delay: 2.8 },
      { type: "d2h", x: 206, width: 80, delay: 3.6 },
    ],
  },
  {
    label: "Stream 1",
    blocks: [
      { type: "h2d", x: 88, width: 80, delay: 3.0 },
      { type: "kernel", x: 176, width: 110, delay: 3.8 },
      { type: "d2h", x: 294, width: 80, delay: 4.6 },
    ],
  },
  {
    label: "Stream 2",
    blocks: [
      { type: "h2d", x: 176, width: 80, delay: 4.0 },
      { type: "kernel", x: 264, width: 110, delay: 4.8 },
      { type: "d2h", x: 382, width: 80, delay: 5.6 },
    ],
  },
  {
    label: "Stream 3",
    blocks: [
      { type: "h2d", x: 264, width: 80, delay: 5.0 },
      { type: "kernel", x: 352, width: 110, delay: 5.8 },
      { type: "d2h", x: 470, width: 80, delay: 6.6 },
    ],
  },
];

const requirements = [
  "Pinned host memory",
  "Separate stream per chunk",
  "Independent data chunks",
  "Launch order: H2D \u2192 Kernel \u2192 D2H per chunk",
];

export const M6S06_OverlapPattern: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Legend
  const legendOpacity = interpolate(
    frame - 1.5 * fps,
    [0, 0.4 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Time arrow
  const timeArrowWidth = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 550],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Overlap highlight
  const overlapHighlightOpacity = interpolate(
    frame - 8 * fps,
    [0, 0.5 * fps],
    [0, 0.12],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Requirements section
  const reqOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={6}>
      <SlideTitle
        title="The Overlap Pattern — Pipeline"
        subtitle="Chunk data across multiple streams for maximum overlap"
      />

      {/* Legend */}
      <div
        style={{
          display: "flex",
          gap: 28,
          marginBottom: 12,
          opacity: legendOpacity,
          width: 600,
        }}
      >
        {[
          { label: "H2D", color: COLORS.h2d },
          { label: "Kernel", color: COLORS.kernel },
          { label: "D2H", color: COLORS.d2h },
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
                width: 24,
                height: 16,
                borderRadius: 3,
                backgroundColor: item.color,
              }}
            />
            <span
              style={{
                fontSize: 14,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              {item.label}
            </span>
          </div>
        ))}
      </div>

      {/* Pipeline diagram */}
      <div
        style={{
          position: "relative",
          width: 1200,
          height: pipelineStreams.length * TRACK_SPACING + 40,
          marginTop: 8,
        }}
      >
        {/* Overlap highlight region */}
        <div
          style={{
            position: "absolute",
            left: TRACK_LEFT + 88,
            top: 0,
            width: 380,
            height: pipelineStreams.length * TRACK_SPACING,
            backgroundColor: `rgba(118,185,0,${overlapHighlightOpacity})`,
            borderRadius: 8,
          }}
        />

        {pipelineStreams.map((stream, si) => (
          <div
            key={stream.label}
            style={{
              position: "absolute",
              top: si * TRACK_SPACING,
              left: 0,
              width: 1200,
              height: BLOCK_H + 8,
            }}
          >
            {/* Stream label */}
            <div
              style={{
                position: "absolute",
                left: 0,
                top: (BLOCK_H + 8 - 20) / 2,
                width: 120,
                fontSize: 14,
                fontWeight: 700,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                textAlign: "right",
                paddingRight: 12,
                opacity: interpolate(
                  frame - (1.5 + si * 0.3) * fps,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              {stream.label}
            </div>

            {/* Track background */}
            <div
              style={{
                position: "absolute",
                left: TRACK_LEFT,
                top: 0,
                width: 700,
                height: BLOCK_H + 8,
                backgroundColor: "rgba(255,255,255,0.02)",
                borderRadius: 4,
              }}
            />

            {/* Blocks */}
            {stream.blocks.map((block, bi) => {
              const bSpring = spring({
                frame: frame - block.delay * fps,
                fps,
                config: { damping: 120, stiffness: 100 },
              });
              const bOpacity = interpolate(bSpring, [0, 1], [0, 1]);
              const bX = interpolate(bSpring, [0, 1], [-30, 0]);

              return (
                <div
                  key={`${si}-${bi}`}
                  style={{
                    position: "absolute",
                    left: TRACK_LEFT + block.x + 4,
                    top: 4,
                    width: block.width,
                    height: BLOCK_H,
                    backgroundColor: `${COLORS[block.type]}cc`,
                    borderRadius: BLOCK_RADIUS,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 12,
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                    color: "#000",
                    opacity: bOpacity,
                    transform: `translateX(${bX}px)`,
                  }}
                >
                  {block.type === "h2d"
                    ? `H2D_${si}`
                    : block.type === "kernel"
                      ? `Kernel_${si}`
                      : `D2H_${si}`}
                </div>
              );
            })}
          </div>
        ))}

        {/* Time arrow */}
        <div
          style={{
            position: "absolute",
            left: TRACK_LEFT,
            top: pipelineStreams.length * TRACK_SPACING + 8,
            display: "flex",
            alignItems: "center",
          }}
        >
          <div
            style={{
              width: timeArrowWidth,
              height: 2,
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

      {/* Requirements checklist */}
      <div
        style={{
          display: "flex",
          gap: 16,
          marginTop: 24,
          opacity: reqOpacity,
          width: 1000,
        }}
      >
        <div
          style={{
            fontSize: 16,
            fontWeight: 700,
            color: THEME.colors.accentCyan,
            fontFamily: fontFamilyHeading,
            flexShrink: 0,
            width: 130,
          }}
        >
          Requirements:
        </div>
        <div style={{ display: "flex", gap: 20, flexWrap: "wrap", width: 800 }}>
          {requirements.map((req, i) => {
            const reqItemOpacity = interpolate(
              frame - (9.5 + i * 0.4) * fps,
              [0, 0.3 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );
            return (
              <div
                key={req}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  opacity: reqItemOpacity,
                }}
              >
                <span
                  style={{
                    fontSize: 14,
                    color: THEME.colors.nvidiaGreen,
                    fontWeight: 700,
                  }}
                >
                  {"\u2713"}
                </span>
                <span
                  style={{
                    fontSize: 14,
                    color: THEME.colors.textPrimary,
                    fontFamily: fontFamilyBody,
                  }}
                >
                  {req}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </SlideLayout>
  );
};
