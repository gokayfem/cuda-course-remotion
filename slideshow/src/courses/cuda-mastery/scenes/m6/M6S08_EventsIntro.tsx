import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint, FadeInText } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

const BLOCK_H = 30;
const TRACK_H = 40;
const EVENT_SIZE = 14;

const codeLines = [
  { text: "cudaEvent_t event;", color: THEME.colors.textCode },
  { text: "cudaEventCreate(&event);", color: THEME.colors.syntaxFunction },
  { text: "", color: "transparent" },
  { text: "// Record event on stream1", color: THEME.colors.syntaxComment },
  { text: "cudaEventRecord(event, stream1);", color: THEME.colors.syntaxFunction },
  { text: "", color: "transparent" },
  { text: "// stream2 waits for event", color: THEME.colors.syntaxComment },
  { text: "cudaStreamWaitEvent(stream2, event);", color: THEME.colors.accentOrange },
  { text: "", color: "transparent" },
  { text: "cudaEventDestroy(event);", color: THEME.colors.accentRed },
];

export const M6S08_EventsIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Stream 1 kernel
  const stream1KernelSpring = spring({
    frame: frame - 1.5 * fps,
    fps,
    config: { damping: 150 },
  });
  const stream1KernelOpacity = interpolate(stream1KernelSpring, [0, 1], [0, 1]);
  const stream1KernelX = interpolate(stream1KernelSpring, [0, 1], [-20, 0]);

  // Event dot on stream 1
  const eventDelay = 3 * fps;
  const eventOpacity = interpolate(
    frame - eventDelay,
    [0, 0.3 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Event glow pulse
  const glowPulse = interpolate(
    frame % (1.2 * fps),
    [0, 0.6 * fps, 1.2 * fps],
    [0.4, 1, 0.4],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Dependency arrow
  const arrowDelay = 4 * fps;
  const arrowProgress = interpolate(
    frame - arrowDelay,
    [0, 0.8 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Stream 2 wait + kernel
  const stream2Delay = 5 * fps;
  const stream2Spring = spring({
    frame: frame - stream2Delay,
    fps,
    config: { damping: 150 },
  });
  const stream2Opacity = interpolate(stream2Spring, [0, 1], [0, 1]);
  const stream2X = interpolate(stream2Spring, [0, 1], [-20, 0]);

  // Wait label on stream 2
  const waitOpacity = interpolate(
    frame - 4.5 * fps,
    [0, 0.3 * fps],
    [0, 0.6],
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
            title="CUDA Events"
            subtitle="Synchronization markers between streams"
          />

          {/* Timeline diagram */}
          <div
            style={{
              position: "relative",
              width: 520,
              height: 260,
              marginTop: 20,
            }}
          >
            {/* Stream 1 */}
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: 520,
              }}
            >
              <div
                style={{
                  fontSize: 14,
                  fontWeight: 700,
                  color: THEME.colors.accentBlue,
                  fontFamily: fontFamilyBody,
                  marginBottom: 6,
                }}
              >
                Stream 1
              </div>
              <div
                style={{
                  position: "relative",
                  height: TRACK_H,
                  backgroundColor: "rgba(79,195,247,0.04)",
                  borderRadius: 6,
                  border: `1px solid ${THEME.colors.accentBlue}20`,
                  width: 500,
                }}
              >
                {/* Kernel A block */}
                <div
                  style={{
                    position: "absolute",
                    left: 10,
                    top: (TRACK_H - BLOCK_H) / 2,
                    width: 140,
                    height: BLOCK_H,
                    backgroundColor: `${THEME.colors.accentBlue}cc`,
                    borderRadius: 5,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 13,
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                    color: "#000",
                    opacity: stream1KernelOpacity,
                    transform: `translateX(${stream1KernelX}px)`,
                  }}
                >
                  Kernel_A
                </div>

                {/* Event dot */}
                <div
                  style={{
                    position: "absolute",
                    left: 160,
                    top: (TRACK_H - EVENT_SIZE) / 2,
                    width: EVENT_SIZE,
                    height: EVENT_SIZE,
                    borderRadius: "50%",
                    backgroundColor: THEME.colors.accentYellow,
                    opacity: eventOpacity,
                    boxShadow:
                      frame > eventDelay
                        ? `0 0 ${8 + glowPulse * 8}px ${THEME.colors.accentYellow}80`
                        : "none",
                  }}
                />

                {/* Event label */}
                <div
                  style={{
                    position: "absolute",
                    left: 180,
                    top: 2,
                    fontSize: 12,
                    fontWeight: 700,
                    color: THEME.colors.accentYellow,
                    fontFamily: fontFamilyBody,
                    opacity: eventOpacity,
                  }}
                >
                  event
                </div>

                {/* Continuation after event */}
                <div
                  style={{
                    position: "absolute",
                    left: 190,
                    top: (TRACK_H - 2) / 2,
                    width: 300,
                    height: 2,
                    backgroundColor: `${THEME.colors.accentBlue}30`,
                    opacity: eventOpacity,
                  }}
                />
              </div>
            </div>

            {/* Dependency arrow (SVG) */}
            <svg
              width={520}
              height={100}
              style={{
                position: "absolute",
                top: 50,
                left: 0,
              }}
            >
              {/* Curved arrow from event to stream 2 */}
              <path
                d={`M 167 10 C 167 50, 120 60, 120 85`}
                fill="none"
                stroke={THEME.colors.accentYellow}
                strokeWidth={2}
                strokeDasharray="6 3"
                opacity={arrowProgress}
              />
              {/* Arrowhead */}
              <polygon
                points="115,80 120,90 125,80"
                fill={THEME.colors.accentYellow}
                opacity={arrowProgress}
              />
            </svg>

            {/* Stream 2 */}
            <div
              style={{
                position: "absolute",
                top: 130,
                left: 0,
                width: 520,
              }}
            >
              <div
                style={{
                  fontSize: 14,
                  fontWeight: 700,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                  marginBottom: 6,
                }}
              >
                Stream 2
              </div>
              <div
                style={{
                  position: "relative",
                  height: TRACK_H,
                  backgroundColor: "rgba(118,185,0,0.04)",
                  borderRadius: 6,
                  border: `1px solid ${THEME.colors.nvidiaGreen}20`,
                  width: 500,
                }}
              >
                {/* Wait region */}
                <div
                  style={{
                    position: "absolute",
                    left: 10,
                    top: (TRACK_H - BLOCK_H) / 2,
                    width: 100,
                    height: BLOCK_H,
                    backgroundColor: "rgba(255,215,64,0.08)",
                    borderRadius: 5,
                    border: `1px dashed ${THEME.colors.accentYellow}40`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 11,
                    fontFamily: fontFamilyBody,
                    fontWeight: 600,
                    color: THEME.colors.accentYellow,
                    opacity: waitOpacity,
                  }}
                >
                  waits...
                </div>

                {/* Event dot on stream 2 */}
                <div
                  style={{
                    position: "absolute",
                    left: 115,
                    top: (TRACK_H - EVENT_SIZE) / 2,
                    width: EVENT_SIZE,
                    height: EVENT_SIZE,
                    borderRadius: "50%",
                    backgroundColor: THEME.colors.accentYellow,
                    opacity: stream2Opacity,
                    boxShadow:
                      frame > stream2Delay
                        ? `0 0 ${8 + glowPulse * 8}px ${THEME.colors.accentYellow}80`
                        : "none",
                  }}
                />

                {/* Kernel B block */}
                <div
                  style={{
                    position: "absolute",
                    left: 140,
                    top: (TRACK_H - BLOCK_H) / 2,
                    width: 140,
                    height: BLOCK_H,
                    backgroundColor: `${THEME.colors.nvidiaGreen}cc`,
                    borderRadius: 5,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 13,
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                    color: "#000",
                    opacity: stream2Opacity,
                    transform: `translateX(${stream2X}px)`,
                  }}
                >
                  Kernel_B
                </div>
              </div>
            </div>

            {/* Explanation label */}
            <div
              style={{
                position: "absolute",
                bottom: 0,
                left: 0,
                width: 500,
                opacity: interpolate(
                  frame - 6 * fps,
                  [0, 0.5 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <FadeInText
                text="Stream 2 blocks until Stream 1's event is recorded"
                delay={6 * fps}
                fontSize={14}
                fontWeight={600}
                color={THEME.colors.textSecondary}
                style={{ width: 500 }}
              />
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 30, width: 480 }}>
          <FadeInText
            text="Event API"
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
              padding: "14px 18px",
              border: `1px solid rgba(255,255,255,0.08)`,
              width: 460,
            }}
          >
            {codeLines.map((line, i) => {
              const lineDelay = (3 + i * 0.25) * fps;
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
                    lineHeight: 1.65,
                    minHeight: line.text === "" ? 8 : "auto",
                    whiteSpace: "nowrap",
                  }}
                >
                  {line.text}
                </div>
              );
            })}
          </div>

          {/* Bullet points */}
          <div style={{ marginTop: 24, width: 460 }}>
            <BulletPoint
              index={0}
              delay={6 * fps}
              text="Events mark points in a stream's execution"
              icon="1"
            />
            <BulletPoint
              index={1}
              delay={6 * fps}
              text="cudaStreamWaitEvent creates cross-stream dependencies"
              icon="2"
              highlight
            />
            <BulletPoint
              index={2}
              delay={6 * fps}
              text="cudaEventSynchronize blocks CPU until event completes"
              icon="3"
            />
          </div>

          {/* Use case box */}
          <div
            style={{
              marginTop: 20,
              padding: "12px 18px",
              backgroundColor: "rgba(179,136,255,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.accentPurple}40`,
              opacity: interpolate(
                frame - 10 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
              width: 440,
            }}
          >
            <span
              style={{
                fontSize: 15,
                color: THEME.colors.accentPurple,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              Use case: Kernel B depends on Kernel A's output, but they run on different streams for other overlapping work.
            </span>
          </div>
        </div>
      }
    />
  );
};
