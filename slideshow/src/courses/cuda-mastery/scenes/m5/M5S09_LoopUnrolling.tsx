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
import { fontFamilyBody, fontFamilyCode, fontFamilyHeading } from "../../../../styles/fonts";

export const M5S09_LoopUnrolling: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Before code
  const beforeCode = [
    { text: "// Standard loop", color: THEME.colors.syntaxComment },
    { text: "for (int i = 0; i < N; i++) {", color: THEME.colors.syntaxKeyword },
    { text: "    sum += data[i];", color: THEME.colors.textCode },
    { text: "}", color: THEME.colors.syntaxKeyword },
  ];

  // After code (unrolled 4x)
  const afterCode = [
    { text: "// Unrolled 4x", color: THEME.colors.syntaxComment },
    { text: "for (int i = 0; i < N; i += 4) {", color: THEME.colors.syntaxKeyword },
    { text: "    sum += data[i];", color: THEME.colors.nvidiaGreen },
    { text: "    sum += data[i+1];", color: THEME.colors.nvidiaGreen },
    { text: "    sum += data[i+2];", color: THEME.colors.nvidiaGreen },
    { text: "    sum += data[i+3];", color: THEME.colors.nvidiaGreen },
    { text: "}", color: THEME.colors.syntaxKeyword },
  ];

  // Animated highlight for unrolled lines
  const highlightDelay = 4 * fps;
  const highlightPulse = interpolate(
    frame - highlightDelay,
    [0, 0.5 * fps, 1 * fps],
    [0, 0.3, 0.15],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Speedup chart data
  const speedups = [
    { factor: "1x", speedup: 1.0, delay: 8 * fps },
    { factor: "2x", speedup: 1.35, delay: 8.3 * fps },
    { factor: "4x", speedup: 1.72, delay: 8.6 * fps },
    { factor: "8x", speedup: 1.85, delay: 8.9 * fps },
  ];
  const MAX_BAR_HEIGHT = 120;
  const maxSpeedup = 2.0;

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={5}
      leftWidth="50%"
      left={
        <div>
          <SlideTitle
            title="Loop Unrolling"
            subtitle="Reduce overhead, expose more ILP to the scheduler"
          />

          {/* BEFORE code */}
          <div style={{ width: 520 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 8,
                opacity: interpolate(
                  frame - 1 * fps,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <div
                style={{
                  padding: "3px 12px",
                  backgroundColor: "rgba(255,171,64,0.15)",
                  borderRadius: 6,
                  fontSize: 13,
                  fontWeight: 700,
                  color: THEME.colors.accentOrange,
                  fontFamily: fontFamilyBody,
                }}
              >
                BEFORE
              </div>
              <span
                style={{
                  fontSize: 14,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                }}
              >
                Standard loop
              </span>
            </div>
            <div
              style={{
                padding: "16px 20px",
                backgroundColor: THEME.colors.bgCode,
                borderRadius: 8,
                border: "1px solid rgba(255,171,64,0.15)",
                width: 500,
              }}
            >
              {beforeCode.map((line, i) => {
                const lineDelay = 1.5 * fps + i * 0.2 * fps;
                const lineOpacity = interpolate(
                  frame - lineDelay,
                  [0, 0.2 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                );
                return (
                  <div
                    key={i}
                    style={{
                      fontSize: 15,
                      color: line.color,
                      fontFamily: fontFamilyCode,
                      lineHeight: 1.7,
                      opacity: lineOpacity,
                    }}
                  >
                    {line.text}
                  </div>
                );
              })}
            </div>

            {/* Arrow between code blocks */}
            <div
              style={{
                display: "flex",
                justifyContent: "center",
                padding: "8px 0",
                opacity: interpolate(
                  frame - 3 * fps,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                <div
                  style={{
                    width: 2,
                    height: 14,
                    backgroundColor: THEME.colors.nvidiaGreen,
                  }}
                />
                <div
                  style={{
                    width: 0,
                    height: 0,
                    borderLeft: "6px solid transparent",
                    borderRight: "6px solid transparent",
                    borderTop: `8px solid ${THEME.colors.nvidiaGreen}`,
                  }}
                />
              </div>
            </div>

            {/* AFTER code */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 8,
                opacity: interpolate(
                  frame - 3 * fps,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <div
                style={{
                  padding: "3px 12px",
                  backgroundColor: "rgba(118,185,0,0.15)",
                  borderRadius: 6,
                  fontSize: 13,
                  fontWeight: 700,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                }}
              >
                AFTER
              </div>
              <span
                style={{
                  fontSize: 14,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                }}
              >
                Unrolled 4x
              </span>
            </div>
            <div
              style={{
                padding: "16px 20px",
                backgroundColor: THEME.colors.bgCode,
                borderRadius: 8,
                border: "1px solid rgba(118,185,0,0.15)",
                width: 500,
              }}
            >
              {afterCode.map((line, i) => {
                const lineDelay = 3.5 * fps + i * 0.15 * fps;
                const lineOpacity = interpolate(
                  frame - lineDelay,
                  [0, 0.2 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                );
                // Highlight unrolled body lines
                const isUnrolledLine = i >= 2 && i <= 5;
                return (
                  <div
                    key={i}
                    style={{
                      fontSize: 15,
                      color: line.color,
                      fontFamily: fontFamilyCode,
                      lineHeight: 1.7,
                      opacity: lineOpacity,
                      backgroundColor: isUnrolledLine
                        ? `rgba(118,185,0,${highlightPulse})`
                        : "transparent",
                      borderRadius: 3,
                      padding: isUnrolledLine ? "0 4px" : 0,
                      margin: isUnrolledLine ? "0 -4px" : 0,
                    }}
                  >
                    {line.text}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 40, width: 480 }}>
          <BulletPoint
            index={0}
            delay={2 * fps}
            text="#pragma unroll — compiler hint for auto unrolling"
            subtext="Tell nvcc to unroll a loop at compile time."
            icon="1"
          />
          <BulletPoint
            index={1}
            delay={2 * fps}
            text="Reduces loop overhead (branch, increment, compare)"
            subtext="Fewer instructions per iteration = less wasted work."
            icon="2"
          />
          <BulletPoint
            index={2}
            delay={2 * fps}
            text="Exposes more ILP to the instruction scheduler"
            subtext="Multiple independent loads/stores can be issued together."
            icon="3"
            highlight
          />
          <BulletPoint
            index={3}
            delay={2 * fps}
            text="Trade-off: more registers used per thread"
            subtext="Unrolling too much can reduce occupancy."
            icon="4"
          />

          {/* Speedup chart */}
          <div
            style={{
              marginTop: 28,
              width: 460,
            }}
          >
            <div
              style={{
                fontSize: 15,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                marginBottom: 12,
                opacity: interpolate(
                  frame - 7.5 * fps,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              Relative Speedup by Unroll Factor
            </div>
            <div
              style={{
                display: "flex",
                gap: 20,
                alignItems: "flex-end",
                height: MAX_BAR_HEIGHT + 30,
                width: 460,
              }}
            >
              {speedups.map((s, i) => {
                const colSpring = spring({
                  frame: frame - s.delay,
                  fps,
                  config: { damping: 100, stiffness: 80 },
                });
                const colHeight = interpolate(
                  colSpring,
                  [0, 1],
                  [0, (s.speedup / maxSpeedup) * MAX_BAR_HEIGHT]
                );
                const colOpacity = interpolate(colSpring, [0, 1], [0, 1]);

                const barColor = i === 0
                  ? THEME.colors.textMuted
                  : i === speedups.length - 1
                    ? THEME.colors.nvidiaGreen
                    : THEME.colors.accentBlue;

                return (
                  <div
                    key={s.factor}
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: 4,
                      flex: 1,
                      opacity: colOpacity,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 14,
                        color: barColor,
                        fontFamily: fontFamilyCode,
                        fontWeight: 700,
                      }}
                    >
                      {s.speedup.toFixed(2)}x
                    </span>
                    <div
                      style={{
                        width: 56,
                        height: colHeight,
                        backgroundColor: `${barColor}40`,
                        borderRadius: 6,
                        border: `1px solid ${barColor}80`,
                      }}
                    />
                    <span
                      style={{
                        fontSize: 13,
                        color: THEME.colors.textSecondary,
                        fontFamily: fontFamilyCode,
                      }}
                    >
                      {s.factor}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Axis label */}
            <div
              style={{
                textAlign: "center",
                fontSize: 12,
                color: THEME.colors.textMuted,
                fontFamily: fontFamilyBody,
                marginTop: 8,
                opacity: interpolate(
                  frame - 9.5 * fps,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              Unroll Factor
            </div>
          </div>
        </div>
      }
    />
  );
};
