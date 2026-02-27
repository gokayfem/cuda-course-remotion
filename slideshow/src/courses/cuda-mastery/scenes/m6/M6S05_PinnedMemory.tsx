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

const BOX_W = 100;
const BOX_H = 36;
const ARROW_COLOR = "rgba(255,255,255,0.3)";

interface BarData {
  label: string;
  value: number;
  maxValue: number;
  color: string;
  delay: number;
}

const bars: BarData[] = [
  { label: "Pageable H2D", value: 12, maxValue: 30, color: THEME.colors.accentRed, delay: 5 },
  { label: "Pinned H2D", value: 25, maxValue: 30, color: THEME.colors.nvidiaGreen, delay: 6 },
  { label: "Pageable D2H", value: 13, maxValue: 30, color: THEME.colors.accentRed, delay: 7 },
  { label: "Pinned D2H", value: 26, maxValue: 30, color: THEME.colors.nvidiaGreen, delay: 8 },
];

const BAR_MAX_W = 280;
const BAR_H = 30;

export const M6S05_PinnedMemory: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Pageable path animation
  const pageableDelay = 1 * fps;
  const pageableOpacity = interpolate(
    frame - pageableDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Pinned path animation
  const pinnedDelay = 3 * fps;
  const pinnedOpacity = interpolate(
    frame - pinnedDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Bottleneck flash for pageable
  const flashCycle = interpolate(
    frame % (1 * fps),
    [0, 0.5 * fps, 1 * fps],
    [0, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );
  const showFlash = frame > 2 * fps && frame < 4.5 * fps;

  // Warning box
  const warningOpacity = interpolate(
    frame - 10 * fps,
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
            title="Pinned vs Pageable Memory"
            subtitle="Why pinned memory is required for async transfers"
          />

          {/* Pageable path */}
          <div style={{ marginTop: 12, opacity: pageableOpacity, width: 520 }}>
            <div
              style={{
                fontSize: 15,
                fontWeight: 700,
                color: THEME.colors.accentRed,
                fontFamily: fontFamilyBody,
                marginBottom: 10,
              }}
            >
              Pageable (two hops)
            </div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                width: 510,
              }}
            >
              {/* CPU Mem */}
              <div
                style={{
                  width: BOX_W,
                  height: BOX_H,
                  backgroundColor: "rgba(79,195,247,0.12)",
                  border: `1px solid ${THEME.colors.accentBlue}60`,
                  borderRadius: 6,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 12,
                  fontWeight: 700,
                  color: THEME.colors.accentBlue,
                  fontFamily: fontFamilyBody,
                  flexShrink: 0,
                }}
              >
                CPU Mem
              </div>

              {/* Arrow 1 */}
              <div
                style={{
                  fontSize: 16,
                  color: ARROW_COLOR,
                }}
              >
                {"\u2192"}
              </div>

              {/* Page Table / Staging */}
              <div
                style={{
                  width: 120,
                  height: BOX_H,
                  backgroundColor: showFlash
                    ? `rgba(255,82,82,${0.1 + flashCycle * 0.15})`
                    : "rgba(255,82,82,0.1)",
                  border: `1px solid ${THEME.colors.accentRed}60`,
                  borderRadius: 6,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 11,
                  fontWeight: 700,
                  color: THEME.colors.accentRed,
                  fontFamily: fontFamilyBody,
                  flexShrink: 0,
                }}
              >
                Staging Buffer
              </div>

              {/* Arrow 2 */}
              <div
                style={{
                  fontSize: 16,
                  color: ARROW_COLOR,
                }}
              >
                {"\u2192"}
              </div>

              {/* GPU */}
              <div
                style={{
                  width: BOX_W,
                  height: BOX_H,
                  backgroundColor: "rgba(118,185,0,0.12)",
                  border: `1px solid ${THEME.colors.nvidiaGreen}60`,
                  borderRadius: 6,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 12,
                  fontWeight: 700,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                  flexShrink: 0,
                }}
              >
                GPU Mem
              </div>
            </div>

            {/* Extra hop label */}
            <div
              style={{
                marginTop: 6,
                fontSize: 12,
                color: THEME.colors.accentRed,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                opacity: showFlash ? 1 : 0.6,
                width: 400,
              }}
            >
              Extra copy through pinned staging buffer (bottleneck)
            </div>
          </div>

          {/* Pinned path */}
          <div style={{ marginTop: 28, opacity: pinnedOpacity, width: 520 }}>
            <div
              style={{
                fontSize: 15,
                fontWeight: 700,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                marginBottom: 10,
              }}
            >
              Pinned (direct DMA)
            </div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                width: 510,
              }}
            >
              {/* CPU Pinned Mem */}
              <div
                style={{
                  width: 120,
                  height: BOX_H,
                  backgroundColor: "rgba(118,185,0,0.12)",
                  border: `2px solid ${THEME.colors.nvidiaGreen}80`,
                  borderRadius: 6,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 12,
                  fontWeight: 700,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                  flexShrink: 0,
                }}
              >
                Pinned Mem
              </div>

              {/* Direct DMA arrow */}
              <div
                style={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  maxWidth: 180,
                }}
              >
                <div
                  style={{
                    width: "100%",
                    height: 3,
                    background: `linear-gradient(90deg, ${THEME.colors.nvidiaGreen}80, ${THEME.colors.nvidiaGreen})`,
                    borderRadius: 2,
                  }}
                />
                <span
                  style={{
                    fontSize: 11,
                    color: THEME.colors.nvidiaGreen,
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                    marginTop: 4,
                  }}
                >
                  DMA (Direct)
                </span>
              </div>

              {/* GPU */}
              <div
                style={{
                  width: BOX_W,
                  height: BOX_H,
                  backgroundColor: "rgba(118,185,0,0.12)",
                  border: `2px solid ${THEME.colors.nvidiaGreen}80`,
                  borderRadius: 6,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 12,
                  fontWeight: 700,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                  flexShrink: 0,
                }}
              >
                GPU Mem
              </div>
            </div>

            <div
              style={{
                marginTop: 6,
                fontSize: 12,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                width: 400,
              }}
            >
              Single hop via DMA — no staging needed
            </div>
          </div>

          {/* Warning */}
          <div
            style={{
              marginTop: 28,
              padding: "12px 18px",
              backgroundColor: "rgba(255,82,82,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.accentRed}40`,
              opacity: warningOpacity,
              width: 500,
            }}
          >
            <span
              style={{
                fontSize: 14,
                color: THEME.colors.accentRed,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
              }}
            >
              Warning:{" "}
            </span>
            <span
              style={{
                fontSize: 14,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
              }}
            >
              Don't over-allocate pinned memory — it reduces OS pageable pool and can hurt system performance.
            </span>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 40, width: 460 }}>
          <FadeInText
            text="Performance Comparison"
            delay={4 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 20, width: 460 }}
          />

          {/* Bar chart */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 16,
              width: 440,
            }}
          >
            {bars.map((bar) => {
              const barSpring = spring({
                frame: frame - bar.delay * fps,
                fps,
                config: { damping: 100, stiffness: 80 },
              });
              const barWidth = interpolate(
                barSpring,
                [0, 1],
                [0, (bar.value / bar.maxValue) * BAR_MAX_W]
              );
              const barOpacity = interpolate(barSpring, [0, 1], [0, 1]);

              const labelOpacity = interpolate(
                frame - (bar.delay + 0.3) * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              return (
                <div key={bar.label} style={{ opacity: barOpacity, width: 440 }}>
                  <div
                    style={{
                      fontSize: 14,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyBody,
                      marginBottom: 4,
                      fontWeight: 600,
                    }}
                  >
                    {bar.label}
                  </div>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 12,
                    }}
                  >
                    <div
                      style={{
                        width: BAR_MAX_W,
                        height: BAR_H,
                        backgroundColor: "rgba(255,255,255,0.05)",
                        borderRadius: 5,
                        overflow: "hidden",
                      }}
                    >
                      <div
                        style={{
                          width: barWidth,
                          height: BAR_H,
                          background: `linear-gradient(90deg, ${bar.color}80, ${bar.color})`,
                          borderRadius: 5,
                        }}
                      />
                    </div>
                    <span
                      style={{
                        fontSize: 18,
                        fontWeight: 700,
                        color: bar.color,
                        fontFamily: fontFamilyCode,
                        opacity: labelOpacity,
                        minWidth: 80,
                      }}
                    >
                      ~{bar.value} GB/s
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* 2x speedup callout */}
          <div
            style={{
              marginTop: 28,
              display: "flex",
              alignItems: "center",
              gap: 14,
              opacity: interpolate(
                frame - 9.5 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
              width: 440,
            }}
          >
            <div
              style={{
                padding: "8px 20px",
                backgroundColor: "rgba(118,185,0,0.15)",
                border: `2px solid ${THEME.colors.nvidiaGreen}`,
                borderRadius: 20,
                fontSize: 22,
                fontWeight: 800,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyHeading,
              }}
            >
              ~2x
            </div>
            <span
              style={{
                fontSize: 16,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
              }}
            >
              transfer bandwidth with pinned memory
            </span>
          </div>
        </div>
      }
    />
  );
};
