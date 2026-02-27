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

type KernelBlock = {
  label: string;
  color: string;
  width: number;
};

const ArrowRight: React.FC<{ color: string }> = ({ color }) => (
  <div
    style={{
      width: 28,
      height: 2,
      backgroundColor: color,
      position: "relative",
      flexShrink: 0,
      alignSelf: "center",
    }}
  >
    <div
      style={{
        position: "absolute",
        right: 0,
        top: -4,
        width: 0,
        height: 0,
        borderLeft: `6px solid ${color}`,
        borderTop: "5px solid transparent",
        borderBottom: "5px solid transparent",
      }}
    />
  </div>
);

const HBMBlock: React.FC<{ opacity: number }> = ({ opacity }) => (
  <div
    style={{
      padding: "4px 8px",
      backgroundColor: "rgba(255,82,82,0.12)",
      border: `1px solid ${THEME.colors.accentRed}40`,
      borderRadius: 4,
      fontSize: 11,
      color: THEME.colors.accentRed,
      fontFamily: fontFamilyCode,
      fontWeight: 600,
      flexShrink: 0,
      opacity,
    }}
  >
    HBM
  </div>
);

export const M9S10_FusionIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const separateBlocks: KernelBlock[] = [
    { label: "Kernel A", color: THEME.colors.accentRed, width: 80 },
    { label: "Kernel B", color: THEME.colors.accentRed, width: 80 },
    { label: "Kernel C", color: THEME.colors.accentRed, width: 80 },
  ];

  const diagramDelay = 1 * fps;
  const separateOpacity = interpolate(
    frame - diagramDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const fusedDelay = 3.5 * fps;
  const fusedOpacity = interpolate(
    frame - fusedDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const tripCountDelay = 5 * fps;
  const tripCountOpacity = interpolate(
    frame - tripCountDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={9}
      leftWidth="52%"
      left={
        <div style={{ width: 800 }}>
          <SlideTitle title="Kernel Fusion -- Why It Matters" />

          {/* Separate Kernels Diagram */}
          <div
            style={{
              marginBottom: 24,
              padding: "16px 20px",
              backgroundColor: "rgba(255,82,82,0.05)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.accentRed}20`,
              opacity: separateOpacity,
            }}
          >
            <div
              style={{
                fontSize: 14,
                fontWeight: 700,
                color: THEME.colors.accentRed,
                fontFamily: fontFamilyBody,
                marginBottom: 12,
              }}
            >
              Separate Kernels (3 HBM round-trips)
            </div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                flexWrap: "wrap",
              }}
            >
              {separateBlocks.map((block, i) => {
                const blockDelay = diagramDelay + (i * 0.6 * fps);
                const blockOpacity = interpolate(
                  frame - blockDelay,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                );

                return (
                  <React.Fragment key={block.label}>
                    <div
                      style={{
                        padding: "8px 14px",
                        backgroundColor: `${block.color}18`,
                        border: `1px solid ${block.color}50`,
                        borderRadius: 6,
                        fontSize: 13,
                        color: block.color,
                        fontFamily: fontFamilyCode,
                        fontWeight: 600,
                        opacity: blockOpacity,
                        flexShrink: 0,
                      }}
                    >
                      {block.label}
                    </div>
                    {i < separateBlocks.length - 1 && (
                      <>
                        <ArrowRight color={THEME.colors.accentRed} />
                        <HBMBlock opacity={blockOpacity} />
                        <ArrowRight color={THEME.colors.accentRed} />
                      </>
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>

          {/* Fused Kernel Diagram */}
          <div
            style={{
              marginBottom: 24,
              padding: "16px 20px",
              backgroundColor: "rgba(118,185,0,0.05)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}20`,
              opacity: fusedOpacity,
            }}
          >
            <div
              style={{
                fontSize: 14,
                fontWeight: 700,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                marginBottom: 12,
              }}
            >
              Fused Kernel (1 HBM round-trip)
            </div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <div
                style={{
                  padding: "4px 8px",
                  backgroundColor: "rgba(118,185,0,0.12)",
                  border: `1px solid ${THEME.colors.nvidiaGreen}40`,
                  borderRadius: 4,
                  fontSize: 11,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyCode,
                  fontWeight: 600,
                  flexShrink: 0,
                }}
              >
                HBM read
              </div>
              <ArrowRight color={THEME.colors.nvidiaGreen} />
              <div
                style={{
                  padding: "8px 20px",
                  backgroundColor: `${THEME.colors.nvidiaGreen}18`,
                  border: `1px solid ${THEME.colors.nvidiaGreen}50`,
                  borderRadius: 6,
                  fontSize: 14,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyCode,
                  fontWeight: 700,
                  flexShrink: 0,
                }}
              >
                Fused A+B+C
              </div>
              <ArrowRight color={THEME.colors.nvidiaGreen} />
              <div
                style={{
                  padding: "4px 8px",
                  backgroundColor: "rgba(118,185,0,0.12)",
                  border: `1px solid ${THEME.colors.nvidiaGreen}40`,
                  borderRadius: 4,
                  fontSize: 11,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyCode,
                  fontWeight: 600,
                  flexShrink: 0,
                }}
              >
                HBM write
              </div>
            </div>
          </div>

          {/* Round-trip comparison */}
          <div
            style={{
              display: "flex",
              gap: 16,
              opacity: tripCountOpacity,
            }}
          >
            <div
              style={{
                flex: 1,
                padding: "10px 14px",
                backgroundColor: "rgba(255,82,82,0.08)",
                borderRadius: 8,
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: 28, fontWeight: 800, color: THEME.colors.accentRed, fontFamily: fontFamilyCode }}>
                6
              </div>
              <div style={{ fontSize: 12, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody }}>
                HBM accesses (separate)
              </div>
            </div>
            <div
              style={{
                flex: 1,
                padding: "10px 14px",
                backgroundColor: "rgba(118,185,0,0.08)",
                borderRadius: 8,
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: 28, fontWeight: 800, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyCode }}>
                2
              </div>
              <div style={{ fontSize: 12, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody }}>
                HBM accesses (fused)
              </div>
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ width: 500, marginTop: 80 }}>
          <BulletPoint
            text="Eliminate intermediate global memory writes"
            index={0}
            delay={6 * fps}
            highlight
          />
          <BulletPoint
            text="Keep data in registers/shared memory"
            index={1}
            delay={6 * fps}
          />
          <BulletPoint
            text="2-5x speedup for memory-bound ops"
            index={2}
            delay={6 * fps}
            highlight
          />
          <BulletPoint
            text="Essential for transformer inference latency"
            index={3}
            delay={6 * fps}
          />

          {/* Bottom callout */}
          <div
            style={{
              marginTop: 32,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}30`,
              opacity: interpolate(
                frame - 9 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div
              style={{
                fontSize: 16,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                lineHeight: 1.5,
              }}
            >
              Most transformer ops are memory-bound -- fusion gives huge wins
            </div>
          </div>
        </div>
      }
    />
  );
};
