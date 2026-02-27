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
import { fontFamilyBody, fontFamilyCode, fontFamilyHeading } from "../../../../styles/fonts";

export const M5S08_ILP: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Pipeline stages
  const stages = ["Fetch", "Decode", "Execute", "Write"];
  const STAGE_W = 100;
  const STAGE_H = 32;
  const STAGE_GAP = 8;
  const ROW_GAP = 6;

  // Single-instruction pipeline (no ILP)
  const singlePipeDelay = 1.5 * fps;

  // ILP pipeline (3 instructions overlapping)
  const ilpDelay = 4 * fps;

  // Code comparison
  const badCode = [
    { text: "// Sequential dependency chain", color: THEME.colors.syntaxComment },
    { text: "a = x[i];", color: THEME.colors.textCode },
    { text: "b = a * 2;    // depends on a", color: THEME.colors.accentRed },
    { text: "c = b + 1;    // depends on b", color: THEME.colors.accentRed },
  ];

  const goodCode = [
    { text: "// Independent operations", color: THEME.colors.syntaxComment },
    { text: "a = x[i];", color: THEME.colors.textCode },
    { text: "b = y[i];     // independent!", color: THEME.colors.nvidiaGreen },
    { text: "c = z[i];     // independent!", color: THEME.colors.nvidiaGreen },
    { text: "a *= 2; b *= 2; c *= 2;", color: THEME.colors.textCode },
  ];

  // Tip animation
  const tipDelay = 9 * fps;
  const tipOpacity = interpolate(
    frame - tipDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Pipeline row colors for ILP visualization
  const instrColors = [
    THEME.colors.accentBlue,
    THEME.colors.accentPurple,
    THEME.colors.accentOrange,
  ];

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={5}
      leftWidth="50%"
      left={
        <div>
          <SlideTitle
            title="Instruction-Level Parallelism"
            subtitle="Overlap independent instructions in the GPU pipeline"
          />

          {/* Single-instruction pipeline */}
          <div style={{ width: 540 }}>
            <FadeInText
              text="Without ILP: instructions wait in line"
              delay={1 * fps}
              fontSize={16}
              fontWeight={600}
              color={THEME.colors.accentRed}
              style={{ marginBottom: 10 }}
            />

            {/* Stage headers */}
            <div style={{ display: "flex", gap: STAGE_GAP, marginBottom: 6, marginLeft: 48 }}>
              {stages.map((s) => (
                <div
                  key={s}
                  style={{
                    width: STAGE_W,
                    textAlign: "center",
                    fontSize: 12,
                    color: THEME.colors.textMuted,
                    fontFamily: fontFamilyCode,
                    fontWeight: 600,
                  }}
                >
                  {s}
                </div>
              ))}
            </div>

            {/* Sequential rows - each instruction occupies one stage at a time */}
            {["Inst A", "Inst B", "Inst C"].map((inst, row) => {
              const rowDelay = singlePipeDelay + row * 0.6 * fps;
              const rowOpacity = interpolate(
                frame - rowDelay,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              return (
                <div
                  key={inst}
                  style={{
                    display: "flex",
                    gap: STAGE_GAP,
                    marginBottom: ROW_GAP,
                    opacity: rowOpacity,
                    alignItems: "center",
                  }}
                >
                  <span
                    style={{
                      width: 44,
                      fontSize: 12,
                      color: THEME.colors.textMuted,
                      fontFamily: fontFamilyCode,
                      textAlign: "right",
                      flexShrink: 0,
                    }}
                  >
                    {inst}
                  </span>
                  {stages.map((_, col) => {
                    // In sequential, inst i is at stage col during cycle (i * stages + col)
                    // Show filled if this is the active stage for this row
                    const isActive = col === 0; // simplified: highlight Fetch to show one-at-a-time
                    return (
                      <div
                        key={col}
                        style={{
                          width: STAGE_W,
                          height: STAGE_H,
                          borderRadius: 4,
                          backgroundColor: row === col
                            ? "rgba(255,82,82,0.25)"
                            : "rgba(255,255,255,0.03)",
                          border: row === col
                            ? `1px solid ${THEME.colors.accentRed}60`
                            : "1px solid rgba(255,255,255,0.06)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 11,
                          color: row === col ? THEME.colors.accentRed : THEME.colors.textMuted,
                          fontFamily: fontFamilyCode,
                          fontWeight: 600,
                        }}
                      >
                        {row === col ? inst : "idle"}
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>

          {/* ILP pipeline */}
          <div style={{ marginTop: 24, width: 540 }}>
            <FadeInText
              text="With ILP: instructions overlap in pipeline"
              delay={3.5 * fps}
              fontSize={16}
              fontWeight={600}
              color={THEME.colors.nvidiaGreen}
              style={{ marginBottom: 10 }}
            />

            {/* Stage headers */}
            <div style={{ display: "flex", gap: STAGE_GAP, marginBottom: 6, marginLeft: 48 }}>
              {stages.map((s) => (
                <div
                  key={`ilp-${s}`}
                  style={{
                    width: STAGE_W,
                    textAlign: "center",
                    fontSize: 12,
                    color: THEME.colors.textMuted,
                    fontFamily: fontFamilyCode,
                    fontWeight: 600,
                  }}
                >
                  {s}
                </div>
              ))}
            </div>

            {/* Overlapped rows - each instruction starts one cycle after the previous */}
            {["Inst A", "Inst B", "Inst C"].map((inst, row) => {
              const rowDelay = ilpDelay + row * 0.3 * fps;
              const rowOpacity = interpolate(
                frame - rowDelay,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              const color = instrColors[row];

              return (
                <div
                  key={`ilp-${inst}`}
                  style={{
                    display: "flex",
                    gap: STAGE_GAP,
                    marginBottom: ROW_GAP,
                    opacity: rowOpacity,
                    alignItems: "center",
                  }}
                >
                  <span
                    style={{
                      width: 44,
                      fontSize: 12,
                      color: THEME.colors.textMuted,
                      fontFamily: fontFamilyCode,
                      textAlign: "right",
                      flexShrink: 0,
                    }}
                  >
                    {inst}
                  </span>
                  {stages.map((stage, col) => {
                    // In pipelined mode, inst i is at stage (col - i)
                    // Active when col matches the instruction's pipeline position
                    const isActive = col === row;
                    return (
                      <div
                        key={col}
                        style={{
                          width: STAGE_W,
                          height: STAGE_H,
                          borderRadius: 4,
                          backgroundColor: isActive
                            ? `${color}30`
                            : "rgba(255,255,255,0.03)",
                          border: isActive
                            ? `1px solid ${color}60`
                            : "1px solid rgba(255,255,255,0.06)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 11,
                          color: isActive ? color : THEME.colors.textMuted,
                          fontFamily: fontFamilyCode,
                          fontWeight: 600,
                        }}
                      >
                        {isActive ? stage : ""}
                      </div>
                    );
                  })}
                </div>
              );
            })}

            {/* Arrow showing all 3 run simultaneously */}
            <FadeInText
              text="All 3 instructions in-flight simultaneously!"
              delay={5.5 * fps}
              fontSize={14}
              fontWeight={700}
              color={THEME.colors.nvidiaGreen}
              style={{ marginTop: 8, textAlign: "center", width: 540 }}
            />
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 40, width: 480 }}>
          {/* BAD code */}
          <div style={{ width: 460 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 8,
                opacity: interpolate(
                  frame - 5 * fps,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <div
                style={{
                  padding: "3px 10px",
                  backgroundColor: "rgba(255,82,82,0.15)",
                  borderRadius: 6,
                  fontSize: 13,
                  fontWeight: 700,
                  color: THEME.colors.accentRed,
                  fontFamily: fontFamilyBody,
                }}
              >
                BAD
              </div>
              <span
                style={{
                  fontSize: 14,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                }}
              >
                Sequential dependencies
              </span>
            </div>
            <div
              style={{
                padding: "14px 18px",
                backgroundColor: THEME.colors.bgCode,
                borderRadius: 8,
                border: "1px solid rgba(255,82,82,0.15)",
                width: 440,
              }}
            >
              {badCode.map((line, i) => {
                const lineDelay = 5.5 * fps + i * 0.2 * fps;
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
                      fontSize: 14,
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
          </div>

          {/* GOOD code */}
          <div style={{ marginTop: 24, width: 460 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 8,
                opacity: interpolate(
                  frame - 7 * fps,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <div
                style={{
                  padding: "3px 10px",
                  backgroundColor: "rgba(118,185,0,0.15)",
                  borderRadius: 6,
                  fontSize: 13,
                  fontWeight: 700,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                }}
              >
                GOOD
              </div>
              <span
                style={{
                  fontSize: 14,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                }}
              >
                Independent operations
              </span>
            </div>
            <div
              style={{
                padding: "14px 18px",
                backgroundColor: THEME.colors.bgCode,
                borderRadius: 8,
                border: "1px solid rgba(118,185,0,0.15)",
                width: 440,
              }}
            >
              {goodCode.map((line, i) => {
                const lineDelay = 7.5 * fps + i * 0.2 * fps;
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
                      fontSize: 14,
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
          </div>

          {/* Bottom tip */}
          <div
            style={{
              marginTop: 24,
              padding: "14px 18px",
              backgroundColor: "rgba(79,195,247,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.accentBlue}40`,
              opacity: tipOpacity,
              width: 440,
            }}
          >
            <span
              style={{
                fontSize: 16,
                color: THEME.colors.accentBlue,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
              }}
            >
              Process multiple elements per thread to expose ILP
            </span>
            <div
              style={{
                fontSize: 14,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                marginTop: 4,
              }}
            >
              Each thread loads 2-4 elements and processes them independently.
            </div>
          </div>
        </div>
      }
    />
  );
};
