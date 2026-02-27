import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode, fontFamilyHeading } from "../../../../styles/fonts";

export const M5S05_LaunchConfig: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Decision tree steps
  const steps = [
    { text: "Start with 128 or 256 threads", icon: "1", color: THEME.colors.accentBlue },
    { text: "Need shared memory? Check bank conflicts at 32-wide", icon: "2", color: THEME.colors.accentPurple },
    { text: "Measure! Profile with different sizes", icon: "3", color: THEME.colors.accentOrange },
    { text: "Use cudaOccupancyMaxPotentialBlockSize()", icon: "4", color: THEME.colors.nvidiaGreen },
  ];

  // Code snippet lines
  const codeLines = [
    { text: "int blockSize, minGridSize;", color: THEME.colors.textCode },
    { text: "cudaOccupancyMaxPotentialBlockSize(", color: THEME.colors.syntaxFunction },
    { text: "    &minGridSize, &blockSize,", color: THEME.colors.textCode },
    { text: "    myKernel, 0, 0", color: THEME.colors.textCode },
    { text: ");", color: THEME.colors.textCode },
    { text: "", color: THEME.colors.textCode },
    { text: "int gridSize =", color: THEME.colors.syntaxKeyword },
    { text: "    (N + blockSize - 1) / blockSize;", color: THEME.colors.textCode },
  ];

  // Performance table
  const perfData = [
    { blockSize: 32, throughput: 45, color: THEME.colors.accentRed },
    { blockSize: 64, throughput: 62, color: THEME.colors.accentOrange },
    { blockSize: 128, throughput: 85, color: THEME.colors.accentBlue },
    { blockSize: 256, throughput: 92, color: THEME.colors.nvidiaGreen },
    { blockSize: 512, throughput: 88, color: THEME.colors.accentBlue },
    { blockSize: 1024, throughput: 72, color: THEME.colors.accentOrange },
  ];

  // Table animation
  const tableDelay = 6 * fps;

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={5}
      leftWidth="48%"
      left={
        <div>
          <SlideTitle
            title="Choosing Block & Grid Size"
            subtitle="A systematic approach to launch configuration"
          />

          {/* Decision tree */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 0,
              marginTop: 8,
              width: 520,
            }}
          >
            {steps.map((step, i) => {
              const stepSpring = spring({
                frame: frame - (1.5 * fps + i * 0.8 * fps),
                fps,
                config: { damping: 200 },
              });
              const stepOpacity = interpolate(stepSpring, [0, 1], [0, 1]);
              const stepX = interpolate(stepSpring, [0, 1], [-20, 0]);

              return (
                <div key={step.text} style={{ width: 520 }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 14,
                      opacity: stepOpacity,
                      transform: `translateX(${stepX}px)`,
                      padding: "10px 0",
                    }}
                  >
                    {/* Step number */}
                    <div
                      style={{
                        width: 32,
                        height: 32,
                        borderRadius: 16,
                        backgroundColor: `${step.color}20`,
                        border: `2px solid ${step.color}80`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 15,
                        fontWeight: 700,
                        color: step.color,
                        fontFamily: fontFamilyCode,
                        flexShrink: 0,
                      }}
                    >
                      {step.icon}
                    </div>

                    <span
                      style={{
                        fontSize: 18,
                        color: THEME.colors.textPrimary,
                        fontFamily: fontFamilyBody,
                        fontWeight: 500,
                        lineHeight: 1.4,
                      }}
                    >
                      {step.text}
                    </span>
                  </div>

                  {/* Connector line */}
                  {i < steps.length - 1 && (
                    <div
                      style={{
                        width: 2,
                        height: 16,
                        backgroundColor: `${step.color}30`,
                        marginLeft: 15,
                        opacity: stepOpacity,
                      }}
                    />
                  )}
                </div>
              );
            })}
          </div>

          {/* Performance table */}
          <div
            style={{
              marginTop: 20,
              width: 520,
            }}
          >
            <div
              style={{
                fontSize: 15,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                marginBottom: 10,
                opacity: interpolate(
                  frame - tableDelay,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              Throughput vs Block Size (example kernel)
            </div>
            <div
              style={{
                display: "flex",
                gap: 8,
                alignItems: "flex-end",
                height: 100,
                width: 520,
              }}
            >
              {perfData.map((d, i) => {
                const colDelay = tableDelay + i * 0.15 * fps;
                const colSpring = spring({
                  frame: frame - colDelay,
                  fps,
                  config: { damping: 100, stiffness: 80 },
                });
                const colHeight = interpolate(colSpring, [0, 1], [0, (d.throughput / 100) * 80]);
                const colOpacity = interpolate(colSpring, [0, 1], [0, 1]);

                return (
                  <div
                    key={d.blockSize}
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
                        fontSize: 11,
                        color: d.color,
                        fontFamily: fontFamilyCode,
                        fontWeight: 700,
                      }}
                    >
                      {d.throughput}%
                    </span>
                    <div
                      style={{
                        width: "100%",
                        maxWidth: 60,
                        height: colHeight,
                        backgroundColor: `${d.color}40`,
                        borderRadius: 4,
                        border: `1px solid ${d.color}80`,
                      }}
                    />
                    <span
                      style={{
                        fontSize: 11,
                        color: THEME.colors.textMuted,
                        fontFamily: fontFamilyCode,
                      }}
                    >
                      {d.blockSize}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 40, width: 480 }}>
          {/* Code snippet */}
          <div
            style={{
              fontSize: 15,
              color: THEME.colors.accentCyan,
              fontFamily: fontFamilyBody,
              fontWeight: 600,
              marginBottom: 10,
              opacity: interpolate(
                frame - 3 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            CUDA Occupancy API
          </div>
          <div
            style={{
              padding: "20px 24px",
              backgroundColor: THEME.colors.bgCode,
              borderRadius: 10,
              border: `1px solid rgba(255,255,255,0.08)`,
              width: 460,
            }}
          >
            {codeLines.map((line, i) => {
              const lineDelay = 3.5 * fps + i * 0.2 * fps;
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
                    fontSize: 15,
                    color: line.color,
                    fontFamily: fontFamilyCode,
                    lineHeight: 1.7,
                    opacity: lineOpacity,
                    minHeight: line.text === "" ? 12 : undefined,
                  }}
                >
                  {line.text}
                </div>
              );
            })}
          </div>

          {/* Explanation */}
          <div
            style={{
              marginTop: 24,
              padding: "14px 18px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: interpolate(
                frame - 6 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
              width: 440,
            }}
          >
            <span
              style={{
                fontSize: 16,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              This API queries the device and returns the block size that maximizes occupancy for your kernel.
            </span>
          </div>

          {/* Pro tip */}
          <div
            style={{
              marginTop: 16,
              padding: "12px 18px",
              backgroundColor: "rgba(79,195,247,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.accentBlue}40`,
              opacity: interpolate(
                frame - 8 * fps,
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
                color: THEME.colors.accentBlue,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              Pro tip: 256 threads is a safe default for most kernels.
            </span>
            <div
              style={{
                fontSize: 13,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                marginTop: 4,
              }}
            >
              It evenly divides into warps (8 warps) and works well on all architectures.
            </div>
          </div>
        </div>
      }
    />
  );
};
