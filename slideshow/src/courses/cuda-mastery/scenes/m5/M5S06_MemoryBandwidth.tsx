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

export const M5S06_MemoryBandwidth: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Formula animation
  const formulaSpring = spring({
    frame: frame - 1 * fps,
    fps,
    config: { damping: 200 },
  });
  const formulaOpacity = interpolate(formulaSpring, [0, 1], [0, 1]);
  const formulaScale = interpolate(formulaSpring, [0, 1], [0.9, 1]);

  // Pipe widths for memory hierarchy diagram
  const pipes = [
    { label: "Global Memory (HBM)", width: 120, color: THEME.colors.accentBlue, delay: 3 * fps },
    { label: "L2 Cache", width: 80, color: THEME.colors.accentPurple, delay: 4 * fps },
    { label: "L1 / Shared Memory", width: 50, color: THEME.colors.nvidiaGreen, delay: 5 * fps },
    { label: "SM Registers", width: 30, color: THEME.colors.accentOrange, delay: 6 * fps },
  ];

  // Key numbers
  const gpuStats = [
    { gpu: "A100", bw: "2,039 GB/s", l2: "40 MB L2", color: THEME.colors.accentBlue, delay: 5 * fps },
    { gpu: "H100", bw: "3,350 GB/s", l2: "50 MB L2", color: THEME.colors.nvidiaGreen, delay: 6 * fps },
  ];

  // Data flow animation - pulsing dots
  const flowPulse = interpolate(
    frame % (1.5 * fps),
    [0, 0.75 * fps, 1.5 * fps],
    [0, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Warning animation
  const warningOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={5}
      leftWidth="52%"
      left={
        <div>
          <SlideTitle
            title="Memory Bandwidth"
            subtitle="The real bottleneck for most GPU kernels"
          />

          {/* Bandwidth equation */}
          <div
            style={{
              display: "flex",
              justifyContent: "flex-start",
              marginBottom: 24,
              opacity: formulaOpacity,
              transform: `scale(${formulaScale})`,
              transformOrigin: "left center",
              width: 560,
            }}
          >
            <div
              style={{
                padding: "14px 28px",
                backgroundColor: "rgba(79,195,247,0.08)",
                borderRadius: 12,
                border: `2px solid ${THEME.colors.accentBlue}40`,
                display: "flex",
                alignItems: "center",
                gap: 12,
              }}
            >
              <span
                style={{
                  fontSize: 20,
                  fontWeight: 700,
                  color: THEME.colors.accentBlue,
                  fontFamily: fontFamilyHeading,
                }}
              >
                Effective BW
              </span>
              <span
                style={{
                  fontSize: 20,
                  color: THEME.colors.textPrimary,
                  fontFamily: fontFamilyHeading,
                }}
              >
                =
              </span>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                <span
                  style={{
                    fontSize: 16,
                    color: THEME.colors.nvidiaGreen,
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                  }}
                >
                  Bytes Read + Bytes Written
                </span>
                <div
                  style={{
                    width: 260,
                    height: 2,
                    backgroundColor: THEME.colors.textSecondary,
                    margin: "3px 0",
                  }}
                />
                <span
                  style={{
                    fontSize: 16,
                    color: THEME.colors.accentOrange,
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                  }}
                >
                  Kernel Time
                </span>
              </div>
            </div>
          </div>

          {/* Pipe diagram: memory hierarchy with narrowing widths */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 0,
              width: 560,
            }}
          >
            {pipes.map((pipe, i) => {
              const pipeSpring = spring({
                frame: frame - pipe.delay,
                fps,
                config: { damping: 200 },
              });
              const pipeOpacity = interpolate(pipeSpring, [0, 1], [0, 1]);
              const pipeScale = interpolate(pipeSpring, [0, 1], [0.8, 1]);

              // Data flow dots
              const dotOpacity = frame > pipe.delay + 0.5 * fps ? flowPulse : 0;

              return (
                <div key={pipe.label} style={{ width: 560 }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 16,
                      opacity: pipeOpacity,
                      transform: `scale(${pipeScale})`,
                      transformOrigin: "left center",
                    }}
                  >
                    {/* Label */}
                    <div
                      style={{
                        width: 200,
                        textAlign: "right",
                        fontSize: 14,
                        color: pipe.color,
                        fontFamily: fontFamilyBody,
                        fontWeight: 600,
                        flexShrink: 0,
                      }}
                    >
                      {pipe.label}
                    </div>

                    {/* Pipe visualization */}
                    <div
                      style={{
                        width: pipe.width,
                        height: 28,
                        borderRadius: 6,
                        backgroundColor: `${pipe.color}20`,
                        border: `2px solid ${pipe.color}60`,
                        position: "relative",
                        overflow: "hidden",
                      }}
                    >
                      {/* Flowing data animation */}
                      <div
                        style={{
                          position: "absolute",
                          top: 0,
                          left: 0,
                          right: 0,
                          bottom: 0,
                          background: `linear-gradient(90deg, transparent, ${pipe.color}40, transparent)`,
                          opacity: dotOpacity,
                        }}
                      />
                    </div>

                    {/* Bandwidth label */}
                    <div
                      style={{
                        fontSize: 12,
                        color: THEME.colors.textMuted,
                        fontFamily: fontFamilyCode,
                        flexShrink: 0,
                      }}
                    >
                      {i === 0 ? "~2-3 TB/s" : i === 1 ? "~6 TB/s" : i === 2 ? "~12 TB/s" : "~80 TB/s"}
                    </div>
                  </div>

                  {/* Connector between pipes */}
                  {i < pipes.length - 1 && (
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "center",
                        padding: "2px 0",
                        marginLeft: 216,
                        opacity: pipeOpacity * 0.5,
                      }}
                    >
                      <div
                        style={{
                          width: 0,
                          height: 0,
                          borderLeft: "6px solid transparent",
                          borderRight: "6px solid transparent",
                          borderTop: `8px solid ${pipe.color}60`,
                        }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Bottleneck label */}
          <FadeInText
            text="Data flows through a narrowing pipeline"
            delay={7 * fps}
            fontSize={15}
            fontWeight={600}
            color={THEME.colors.textSecondary}
            style={{ marginTop: 12, textAlign: "center", width: 560 }}
          />
        </div>
      }
      right={
        <div style={{ paddingTop: 50, width: 460 }}>
          <FadeInText
            text="Key GPU Bandwidth Numbers"
            delay={4 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 20, width: 460 }}
          />

          {/* GPU stats cards */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 16,
              width: 460,
            }}
          >
            {gpuStats.map((gpu, i) => {
              const cardSpring = spring({
                frame: frame - gpu.delay,
                fps,
                config: { damping: 200 },
              });
              const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
              const cardX = interpolate(cardSpring, [0, 1], [20, 0]);

              return (
                <div
                  key={gpu.gpu}
                  style={{
                    padding: "16px 22px",
                    backgroundColor: `${gpu.color}08`,
                    borderRadius: 10,
                    border: `1px solid ${gpu.color}40`,
                    opacity: cardOpacity,
                    transform: `translateX(${cardX}px)`,
                    width: 420,
                  }}
                >
                  <div
                    style={{
                      fontSize: 22,
                      fontWeight: 800,
                      color: gpu.color,
                      fontFamily: fontFamilyHeading,
                      marginBottom: 4,
                    }}
                  >
                    {gpu.gpu}
                  </div>
                  <div
                    style={{
                      display: "flex",
                      gap: 20,
                      fontSize: 16,
                      fontFamily: fontFamilyCode,
                    }}
                  >
                    <span style={{ color: THEME.colors.textPrimary, fontWeight: 600 }}>
                      {gpu.bw} HBM
                    </span>
                    <span style={{ color: THEME.colors.textSecondary }}>
                      {gpu.l2}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Typical utilization */}
          <FadeInText
            text="Most kernels achieve 60-80% of peak bandwidth"
            delay={7 * fps}
            fontSize={17}
            fontWeight={600}
            color={THEME.colors.accentOrange}
            style={{ marginTop: 24, width: 460 }}
          />

          {/* Warning box */}
          <div
            style={{
              marginTop: 24,
              padding: "14px 18px",
              backgroundColor: "rgba(255,82,82,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.accentRed}40`,
              opacity: warningOpacity,
              width: 420,
            }}
          >
            <span
              style={{
                fontSize: 16,
                color: THEME.colors.accentRed,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
              }}
            >
              If your kernel is memory-bound, no amount of compute optimization helps
            </span>
            <div
              style={{
                fontSize: 14,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                marginTop: 4,
              }}
            >
              First determine if you are compute-bound or memory-bound, then optimize accordingly.
            </div>
          </div>
        </div>
      }
    />
  );
};
