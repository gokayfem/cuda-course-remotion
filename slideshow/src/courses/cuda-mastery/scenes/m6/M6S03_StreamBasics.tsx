import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

const BLOCK_H = 26;
const TRACK_H = 34;
const STREAM_GAP = 10;

interface StreamOp {
  label: string;
  width: number;
  color: string;
  x: number;
}

const streams: { name: string; ops: StreamOp[] }[] = [
  {
    name: "Stream 0 (default)",
    ops: [
      { label: "A", width: 80, color: THEME.colors.accentBlue, x: 0 },
      { label: "B", width: 70, color: THEME.colors.nvidiaGreen, x: 88 },
      { label: "C", width: 60, color: THEME.colors.accentOrange, x: 166 },
    ],
  },
  {
    name: "Stream 1",
    ops: [
      { label: "D", width: 90, color: THEME.colors.accentPurple, x: 0 },
      { label: "E", width: 80, color: THEME.colors.accentCyan, x: 98 },
    ],
  },
  {
    name: "Stream 2",
    ops: [
      { label: "F", width: 70, color: THEME.colors.accentYellow, x: 0 },
      { label: "G", width: 100, color: THEME.colors.accentRed, x: 78 },
    ],
  },
];

const codeLines = [
  { text: "cudaStream_t stream1, stream2;", color: THEME.colors.textCode },
  { text: "cudaStreamCreate(&stream1);", color: THEME.colors.syntaxFunction },
  { text: "cudaStreamCreate(&stream2);", color: THEME.colors.syntaxFunction },
  { text: "", color: "transparent" },
  {
    text: "// Launch on separate streams",
    color: THEME.colors.syntaxComment,
  },
  {
    text: "kernel_A<<<grid, block, 0, stream1>>>(..);",
    color: THEME.colors.textCode,
  },
  {
    text: "kernel_B<<<grid, block, 0, stream2>>>(..);",
    color: THEME.colors.textCode,
  },
  { text: "", color: "transparent" },
  {
    text: "cudaStreamSynchronize(stream1);",
    color: THEME.colors.syntaxKeyword,
  },
  {
    text: "cudaStreamDestroy(stream1);",
    color: THEME.colors.accentRed,
  },
];

export const M6S03_StreamBasics: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Arrow annotations
  const withinArrowOpacity = interpolate(
    frame - 5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const acrossArrowOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Tip at bottom
  const tipOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={6}
      leftWidth="50%"
      left={
        <div style={{ width: 540 }}>
          <SlideTitle
            title="CUDA Streams — The Basics"
            subtitle="Ordered queues of GPU operations"
          />

          {/* Stream diagram */}
          <div style={{ position: "relative", marginTop: 16, width: 520 }}>
            {streams.map((stream, si) => {
              const streamDelay = (1 + si * 0.8) * fps;
              const labelSpring = spring({
                frame: frame - streamDelay,
                fps,
                config: { damping: 200 },
              });
              const labelOpacity = interpolate(labelSpring, [0, 1], [0, 1]);

              return (
                <div
                  key={stream.name}
                  style={{
                    marginBottom: STREAM_GAP + 6,
                    opacity: labelOpacity,
                    width: 520,
                  }}
                >
                  {/* Stream label */}
                  <div
                    style={{
                      fontSize: 14,
                      fontWeight: 700,
                      color:
                        si === 0
                          ? THEME.colors.textSecondary
                          : THEME.colors.accentCyan,
                      fontFamily: fontFamilyBody,
                      marginBottom: 4,
                    }}
                  >
                    {stream.name}
                  </div>

                  {/* Track */}
                  <div
                    style={{
                      position: "relative",
                      height: TRACK_H,
                      backgroundColor: "rgba(255,255,255,0.04)",
                      borderRadius: 6,
                      border: `1px solid rgba(255,255,255,0.06)`,
                      width: 500,
                    }}
                  >
                    {stream.ops.map((op, oi) => {
                      const opDelay =
                        streamDelay + (0.3 + oi * 0.25) * fps;
                      const opSpring = spring({
                        frame: frame - opDelay,
                        fps,
                        config: { damping: 150 },
                      });
                      const opOpacity = interpolate(
                        opSpring,
                        [0, 1],
                        [0, 1]
                      );
                      const opX = interpolate(opSpring, [0, 1], [-20, 0]);

                      return (
                        <div
                          key={`${si}-${oi}`}
                          style={{
                            position: "absolute",
                            left: op.x + 4,
                            top: (TRACK_H - BLOCK_H) / 2,
                            width: op.width,
                            height: BLOCK_H,
                            backgroundColor: `${op.color}cc`,
                            borderRadius: 5,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: 13,
                            fontWeight: 700,
                            fontFamily: fontFamilyCode,
                            color: "#000",
                            opacity: opOpacity,
                            transform: `translateX(${opX}px)`,
                          }}
                        >
                          {op.label}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}

            {/* Annotation: ops within a stream = ordered */}
            <div
              style={{
                marginTop: 16,
                display: "flex",
                alignItems: "center",
                gap: 10,
                opacity: withinArrowOpacity,
                width: 500,
              }}
            >
              <div
                style={{
                  width: 20,
                  height: 2,
                  backgroundColor: THEME.colors.accentBlue,
                }}
              />
              <span
                style={{
                  fontSize: 14,
                  color: THEME.colors.accentBlue,
                  fontFamily: fontFamilyBody,
                  fontWeight: 600,
                }}
              >
                Ops within a stream = ordered (FIFO)
              </span>
            </div>

            {/* Annotation: ops across streams = concurrent */}
            <div
              style={{
                marginTop: 8,
                display: "flex",
                alignItems: "center",
                gap: 10,
                opacity: acrossArrowOpacity,
                width: 500,
              }}
            >
              <div
                style={{
                  width: 20,
                  height: 2,
                  backgroundColor: THEME.colors.nvidiaGreen,
                }}
              />
              <span
                style={{
                  fontSize: 14,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                  fontWeight: 600,
                }}
              >
                Ops across streams = concurrent (if HW allows)
              </span>
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 40, width: 480 }}>
          <FadeInText
            text="Stream API"
            delay={2 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 16, width: 460 }}
          />

          {/* Code block */}
          <div
            style={{
              backgroundColor: THEME.colors.bgCode,
              borderRadius: 10,
              padding: "16px 20px",
              border: `1px solid rgba(255,255,255,0.08)`,
              width: 460,
            }}
          >
            {codeLines.map((line, i) => {
              const lineDelay = (2.5 + i * 0.3) * fps;
              const lineOpacity = interpolate(
                frame - lineDelay,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              return (
                <div
                  key={i}
                  style={{
                    fontSize: 14,
                    fontFamily: fontFamilyCode,
                    color: line.color,
                    opacity: lineOpacity,
                    lineHeight: 1.7,
                    minHeight: line.text === "" ? 10 : "auto",
                    whiteSpace: "nowrap",
                  }}
                >
                  {line.text}
                </div>
              );
            })}
          </div>

          {/* Tip box */}
          <div
            style={{
              marginTop: 24,
              padding: "12px 18px",
              backgroundColor: "rgba(255,215,64,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.accentYellow}40`,
              opacity: tipOpacity,
              width: 440,
            }}
          >
            <span
              style={{
                fontSize: 15,
                color: THEME.colors.accentYellow,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
              }}
            >
              Tip:{" "}
            </span>
            <span
              style={{
                fontSize: 15,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
              }}
            >
              The 4th argument to{" "}
              <span
                style={{
                  fontFamily: fontFamilyCode,
                  color: THEME.colors.accentCyan,
                }}
              >
                {"<<<>>>"}
              </span>{" "}
              is the stream
            </span>
          </div>
        </div>
      }
    />
  );
};
