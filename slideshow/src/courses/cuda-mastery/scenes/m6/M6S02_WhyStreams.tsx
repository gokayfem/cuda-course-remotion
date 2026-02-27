import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

const BLOCK_HEIGHT = 28;
const BLOCK_RADIUS = 5;

interface TimelineBlock {
  label: string;
  width: number;
  color: string;
}

const serialBlocks: TimelineBlock[] = [
  { label: "H2D", width: 60, color: THEME.colors.accentBlue },
  { label: "Kernel", width: 90, color: THEME.colors.nvidiaGreen },
  { label: "D2H", width: 60, color: THEME.colors.accentOrange },
  { label: "H2D", width: 60, color: THEME.colors.accentBlue },
  { label: "Kernel", width: 90, color: THEME.colors.nvidiaGreen },
  { label: "D2H", width: 60, color: THEME.colors.accentOrange },
];

const concurrentStreams: TimelineBlock[][] = [
  [
    { label: "H2D", width: 60, color: THEME.colors.accentBlue },
    { label: "Kernel", width: 90, color: THEME.colors.nvidiaGreen },
    { label: "D2H", width: 60, color: THEME.colors.accentOrange },
  ],
  [
    { label: "H2D", width: 60, color: THEME.colors.accentBlue },
    { label: "Kernel", width: 90, color: THEME.colors.nvidiaGreen },
    { label: "D2H", width: 60, color: THEME.colors.accentOrange },
  ],
  [
    { label: "H2D", width: 60, color: THEME.colors.accentBlue },
    { label: "Kernel", width: 90, color: THEME.colors.nvidiaGreen },
    { label: "D2H", width: 60, color: THEME.colors.accentOrange },
  ],
];

export const M6S02_WhyStreams: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Serial timeline animation
  const serialOpacity = interpolate(
    frame,
    [0.5 * fps, 1 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Concurrent timeline animation
  const concurrentOpacity = interpolate(
    frame,
    [3 * fps, 3.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Time arrow widths
  const serialTotalWidth = serialBlocks.reduce(
    (sum, b) => sum + b.width + 4,
    0
  );
  const concurrentTotalWidth = 260; // overlapped blocks are shorter

  const serialArrowWidth = interpolate(
    frame - 2 * fps,
    [0, 0.5 * fps],
    [0, serialTotalWidth],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const concurrentArrowWidth = interpolate(
    frame - 5.5 * fps,
    [0, 0.5 * fps],
    [0, concurrentTotalWidth],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Speedup label
  const speedupOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={6}
      leftWidth="55%"
      left={
        <div style={{ width: 580 }}>
          <SlideTitle
            title="Why Streams Matter"
            subtitle="Serial vs concurrent execution"
          />

          {/* Serial timeline */}
          <div style={{ marginTop: 16, opacity: serialOpacity, width: 560 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 10,
              }}
            >
              <span
                style={{
                  fontSize: 16,
                  fontWeight: 700,
                  color: THEME.colors.accentRed,
                  fontFamily: fontFamilyBody,
                }}
              >
                Serial
              </span>
              <div
                style={{
                  padding: "2px 10px",
                  backgroundColor: "rgba(255,82,82,0.12)",
                  borderRadius: 4,
                  fontSize: 12,
                  color: THEME.colors.accentRed,
                  fontFamily: fontFamilyBody,
                  fontWeight: 600,
                }}
              >
                Slow
              </div>
            </div>

            <div
              style={{
                display: "flex",
                gap: 4,
                padding: "10px 12px",
                backgroundColor: "rgba(255,82,82,0.06)",
                borderRadius: 8,
                border: `1px solid ${THEME.colors.accentRed}30`,
                width: 540,
              }}
            >
              {serialBlocks.map((block, i) => {
                const blockSpring = spring({
                  frame: frame - (1 + i * 0.2) * fps,
                  fps,
                  config: { damping: 200 },
                });
                const blockOpacity = interpolate(blockSpring, [0, 1], [0, 1]);

                return (
                  <div
                    key={`serial-${i}`}
                    style={{
                      width: block.width,
                      height: BLOCK_HEIGHT,
                      backgroundColor: block.color,
                      borderRadius: BLOCK_RADIUS,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 11,
                      fontFamily: fontFamilyCode,
                      fontWeight: 700,
                      color: "#000",
                      opacity: blockOpacity,
                      flexShrink: 0,
                    }}
                  >
                    {block.label}
                  </div>
                );
              })}
            </div>

            {/* Serial time arrow */}
            <div
              style={{
                marginTop: 6,
                marginLeft: 12,
                display: "flex",
                alignItems: "center",
              }}
            >
              <div
                style={{
                  width: serialArrowWidth,
                  height: 2,
                  backgroundColor: THEME.colors.accentRed,
                }}
              />
              <span
                style={{
                  fontSize: 11,
                  color: THEME.colors.accentRed,
                  fontFamily: fontFamilyBody,
                  marginLeft: 6,
                  whiteSpace: "nowrap",
                }}
              >
                6 units
              </span>
            </div>
          </div>

          {/* Concurrent timeline */}
          <div
            style={{ marginTop: 24, opacity: concurrentOpacity, width: 560 }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 10,
              }}
            >
              <span
                style={{
                  fontSize: 16,
                  fontWeight: 700,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                }}
              >
                Concurrent
              </span>
              <div
                style={{
                  padding: "2px 10px",
                  backgroundColor: "rgba(118,185,0,0.12)",
                  borderRadius: 4,
                  fontSize: 12,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                  fontWeight: 600,
                }}
              >
                Fast
              </div>
            </div>

            <div
              style={{
                padding: "10px 12px",
                backgroundColor: "rgba(118,185,0,0.06)",
                borderRadius: 8,
                border: `1px solid ${THEME.colors.nvidiaGreen}30`,
                position: "relative",
                height: 3 * BLOCK_HEIGHT + 2 * 8 + 20,
                width: 540,
              }}
            >
              {concurrentStreams.map((stream, si) => {
                const streamOffset = si * 60; // stagger offset
                let blockX = streamOffset;

                return stream.map((block, bi) => {
                  const x = blockX;
                  blockX += block.width + 4;

                  const blockSpring = spring({
                    frame: frame - (3.5 + si * 0.3 + bi * 0.15) * fps,
                    fps,
                    config: { damping: 200 },
                  });
                  const blockOpacity = interpolate(
                    blockSpring,
                    [0, 1],
                    [0, 1]
                  );

                  return (
                    <div
                      key={`concurrent-${si}-${bi}`}
                      style={{
                        position: "absolute",
                        left: 12 + x,
                        top: 10 + si * (BLOCK_HEIGHT + 8),
                        width: block.width,
                        height: BLOCK_HEIGHT,
                        backgroundColor: block.color,
                        borderRadius: BLOCK_RADIUS,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 11,
                        fontFamily: fontFamilyCode,
                        fontWeight: 700,
                        color: "#000",
                        opacity: blockOpacity,
                      }}
                    >
                      {block.label}
                    </div>
                  );
                });
              })}
            </div>

            {/* Concurrent time arrow */}
            <div
              style={{
                marginTop: 6,
                marginLeft: 12,
                display: "flex",
                alignItems: "center",
              }}
            >
              <div
                style={{
                  width: concurrentArrowWidth,
                  height: 2,
                  backgroundColor: THEME.colors.nvidiaGreen,
                }}
              />
              <span
                style={{
                  fontSize: 11,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                  marginLeft: 6,
                  whiteSpace: "nowrap",
                }}
              >
                ~2.5 units
              </span>
            </div>
          </div>

          {/* Speedup badge */}
          <div
            style={{
              marginTop: 20,
              display: "flex",
              alignItems: "center",
              gap: 12,
              opacity: speedupOpacity,
              width: 400,
            }}
          >
            <div
              style={{
                padding: "8px 20px",
                backgroundColor: "rgba(118,185,0,0.15)",
                border: `2px solid ${THEME.colors.nvidiaGreen}`,
                borderRadius: 20,
                fontSize: 24,
                fontWeight: 800,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
              }}
            >
              ~2.4x
            </div>
            <span
              style={{
                fontSize: 16,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
              }}
            >
              speedup from overlapping
            </span>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 60, width: 420 }}>
          <BulletPoint
            index={0}
            delay={2 * fps}
            text="Serial execution wastes GPU resources"
            icon="1"
          />
          <BulletPoint
            index={1}
            delay={2 * fps}
            text="Streams enable concurrent execution of independent ops"
            icon="2"
            highlight
          />
          <BulletPoint
            index={2}
            delay={2 * fps}
            text="Overlap data transfer with computation"
            icon="3"
          />
          <BulletPoint
            index={3}
            delay={2 * fps}
            text="2-3x speedup for transfer-heavy workloads"
            icon="4"
            highlight
          />

          {/* Insight box */}
          <div
            style={{
              marginTop: 28,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: interpolate(
                frame - 8 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
              width: 400,
            }}
          >
            <span
              style={{
                fontSize: 17,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              The GPU has{" "}
              <span style={{ color: THEME.colors.nvidiaGreen }}>
                separate engines
              </span>{" "}
              for copy and compute — use them simultaneously.
            </span>
          </div>
        </div>
      }
    />
  );
};
