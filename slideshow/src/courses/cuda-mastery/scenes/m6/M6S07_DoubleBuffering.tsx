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

const BLOCK_H = 34;
const BLOCK_R = 6;
const TRACK_H = 44;

interface BufferBlock {
  label: string;
  width: number;
  color: string;
  x: number;
  delay: number;
}

const bufferABlocks: BufferBlock[] = [
  { label: "Fill A", width: 90, color: THEME.colors.accentBlue, x: 0, delay: 2 },
  { label: "Process A", width: 110, color: THEME.colors.nvidiaGreen, x: 98, delay: 3 },
  { label: "Fill A", width: 90, color: THEME.colors.accentBlue, x: 318, delay: 5.5 },
  { label: "Process A", width: 110, color: THEME.colors.nvidiaGreen, x: 416, delay: 6.5 },
];

const bufferBBlocks: BufferBlock[] = [
  { label: "Fill B", width: 90, color: THEME.colors.accentPurple, x: 98, delay: 3 },
  { label: "Process B", width: 110, color: THEME.colors.accentOrange, x: 208, delay: 4.2 },
  { label: "Fill B", width: 90, color: THEME.colors.accentPurple, x: 416, delay: 6.5 },
];

const codeLines = [
  { text: "float *buf[2];  // double buffers", color: THEME.colors.textCode },
  { text: "cudaMallocHost(&buf[0], size);", color: THEME.colors.syntaxFunction },
  { text: "cudaMallocHost(&buf[1], size);", color: THEME.colors.syntaxFunction },
  { text: "", color: "transparent" },
  { text: "// Initial fill", color: THEME.colors.syntaxComment },
  { text: "fillBuffer(buf[0]);", color: THEME.colors.textCode },
  { text: "", color: "transparent" },
  { text: "for (int i = 0; i < N; i++) {", color: THEME.colors.syntaxKeyword },
  { text: "  int cur = i % 2;", color: THEME.colors.textCode },
  { text: "  int nxt = (i + 1) % 2;", color: THEME.colors.textCode },
  { text: "", color: "transparent" },
  { text: "  // Overlap: fill next + process current", color: THEME.colors.syntaxComment },
  { text: "  cudaMemcpyAsync(d, buf[cur],", color: THEME.colors.syntaxFunction },
  { text: "    size, H2D, stream1);", color: THEME.colors.textCode },
  { text: "  kernel<<<g, b, 0, stream1>>>();", color: THEME.colors.syntaxFunction },
  { text: "  fillBuffer(buf[nxt]);  // CPU", color: THEME.colors.accentCyan },
  { text: "}", color: THEME.colors.syntaxKeyword },
];

export const M6S07_DoubleBuffering: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Swap arrows animation
  const swapArrowOpacity = interpolate(
    frame - 4.5 * fps,
    [0, 0.5 * fps],
    [0, 0.7],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Code block opacity
  const codeOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={6}
      leftWidth="52%"
      left={
        <div style={{ width: 580 }}>
          <SlideTitle
            title="Double Buffering"
            subtitle="Ping-pong between two buffers for continuous overlap"
          />

          {/* Buffer A track */}
          <div style={{ marginTop: 12, width: 560 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 6,
              }}
            >
              <div
                style={{
                  width: 14,
                  height: 14,
                  borderRadius: 3,
                  backgroundColor: THEME.colors.accentBlue,
                }}
              />
              <span
                style={{
                  fontSize: 15,
                  fontWeight: 700,
                  color: THEME.colors.accentBlue,
                  fontFamily: fontFamilyBody,
                }}
              >
                Buffer A
              </span>
            </div>

            <div
              style={{
                position: "relative",
                height: TRACK_H,
                backgroundColor: "rgba(79,195,247,0.04)",
                borderRadius: 6,
                border: `1px solid ${THEME.colors.accentBlue}20`,
                width: 550,
              }}
            >
              {bufferABlocks.map((block, i) => {
                const bSpring = spring({
                  frame: frame - block.delay * fps,
                  fps,
                  config: { damping: 150 },
                });
                const bOpacity = interpolate(bSpring, [0, 1], [0, 1]);
                const bX = interpolate(bSpring, [0, 1], [-20, 0]);

                return (
                  <div
                    key={`a-${i}`}
                    style={{
                      position: "absolute",
                      left: block.x + 4,
                      top: (TRACK_H - BLOCK_H) / 2,
                      width: block.width,
                      height: BLOCK_H,
                      backgroundColor: `${block.color}cc`,
                      borderRadius: BLOCK_R,
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
                    {block.label}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Swap arrows between tracks */}
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              gap: 60,
              opacity: swapArrowOpacity,
              margin: "6px 0",
              width: 550,
            }}
          >
            {[0, 1, 2].map((i) => (
              <div
                key={`swap-${i}`}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 0,
                }}
              >
                <span
                  style={{
                    fontSize: 18,
                    color: THEME.colors.accentYellow,
                  }}
                >
                  {"\u21C5"}
                </span>
                <span
                  style={{
                    fontSize: 10,
                    color: THEME.colors.accentYellow,
                    fontFamily: fontFamilyBody,
                    fontWeight: 600,
                  }}
                >
                  swap
                </span>
              </div>
            ))}
          </div>

          {/* Buffer B track */}
          <div style={{ width: 560 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 6,
              }}
            >
              <div
                style={{
                  width: 14,
                  height: 14,
                  borderRadius: 3,
                  backgroundColor: THEME.colors.accentPurple,
                }}
              />
              <span
                style={{
                  fontSize: 15,
                  fontWeight: 700,
                  color: THEME.colors.accentPurple,
                  fontFamily: fontFamilyBody,
                }}
              >
                Buffer B
              </span>
            </div>

            <div
              style={{
                position: "relative",
                height: TRACK_H,
                backgroundColor: "rgba(179,136,255,0.04)",
                borderRadius: 6,
                border: `1px solid ${THEME.colors.accentPurple}20`,
                width: 550,
              }}
            >
              {bufferBBlocks.map((block, i) => {
                const bSpring = spring({
                  frame: frame - block.delay * fps,
                  fps,
                  config: { damping: 150 },
                });
                const bOpacity = interpolate(bSpring, [0, 1], [0, 1]);
                const bX = interpolate(bSpring, [0, 1], [-20, 0]);

                return (
                  <div
                    key={`b-${i}`}
                    style={{
                      position: "absolute",
                      left: block.x + 4,
                      top: (TRACK_H - BLOCK_H) / 2,
                      width: block.width,
                      height: BLOCK_H,
                      backgroundColor: `${block.color}cc`,
                      borderRadius: BLOCK_R,
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
                    {block.label}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Time arrow */}
          <div
            style={{
              marginTop: 10,
              display: "flex",
              alignItems: "center",
              width: 550,
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

          {/* Code snippet */}
          <div
            style={{
              marginTop: 16,
              backgroundColor: THEME.colors.bgCode,
              borderRadius: 8,
              padding: "10px 14px",
              border: `1px solid rgba(255,255,255,0.08)`,
              opacity: codeOpacity,
              width: 540,
              maxHeight: 240,
              overflow: "hidden",
            }}
          >
            {codeLines.map((line, i) => (
              <div
                key={i}
                style={{
                  fontSize: 12,
                  fontFamily: fontFamilyCode,
                  color: line.color,
                  lineHeight: 1.55,
                  minHeight: line.text === "" ? 6 : "auto",
                  whiteSpace: "nowrap",
                }}
              >
                {line.text}
              </div>
            ))}
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 60, width: 420 }}>
          <BulletPoint
            index={0}
            delay={2 * fps}
            text="Simplest overlap pattern — just 2 buffers"
            icon="1"
          />
          <BulletPoint
            index={1}
            delay={2 * fps}
            text="While GPU processes buffer A, CPU fills buffer B"
            icon="2"
            highlight
          />
          <BulletPoint
            index={2}
            delay={2 * fps}
            text="Swap buffers each iteration"
            icon="3"
          />
          <BulletPoint
            index={3}
            delay={2 * fps}
            text="Works great for streaming data (training, inference)"
            icon="4"
            highlight
          />

          {/* Pattern box */}
          <div
            style={{
              marginTop: 28,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: interpolate(
                frame - 9 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
              width: 400,
            }}
          >
            <span
              style={{
                fontSize: 16,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              Double buffering is the go-to pattern for{" "}
              <span style={{ color: THEME.colors.nvidiaGreen }}>
                real-time ML inference
              </span>{" "}
              — process one batch while loading the next.
            </span>
          </div>
        </div>
      }
    />
  );
};
