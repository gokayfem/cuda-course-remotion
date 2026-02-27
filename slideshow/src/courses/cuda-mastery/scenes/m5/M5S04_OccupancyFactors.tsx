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

export const M5S04_OccupancyFactors: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const resources = [
    {
      name: "Registers per Thread",
      detail: "SM has 65,536 registers total",
      example: "32 regs/thread x 256 threads = 8,192 per block",
      used: 0.65,
      limit: 1.0,
      color: THEME.colors.accentBlue,
      delay: 1.5 * fps,
    },
    {
      name: "Shared Memory per Block",
      detail: "SM has 48-164 KB shared memory",
      example: "16 KB/block x 4 blocks = 64 KB needed",
      used: 0.8,
      limit: 1.0,
      color: THEME.colors.accentPurple,
      delay: 3.5 * fps,
    },
    {
      name: "Threads per Block",
      detail: "SM supports max 48 warps (1,536 threads)",
      example: "256 threads/block x 6 blocks = 1,536 threads",
      used: 0.5,
      limit: 1.0,
      color: THEME.colors.accentOrange,
      delay: 5.5 * fps,
    },
  ];

  const BAR_WIDTH = 480;
  const BAR_HEIGHT = 28;

  // Bottom callout
  const calloutOpacity = interpolate(
    frame - 8 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={5}
      leftWidth="56%"
      left={
        <div>
          <SlideTitle
            title="What Limits Occupancy?"
            subtitle="Three resources compete for limited SM capacity"
          />

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 28,
              marginTop: 16,
              width: 560,
            }}
          >
            {resources.map((res, i) => {
              const barSpring = spring({
                frame: frame - res.delay,
                fps,
                config: { damping: 100, stiffness: 60 },
              });
              const usedWidth = interpolate(barSpring, [0, 1], [0, res.used * BAR_WIDTH]);
              const fadeIn = interpolate(barSpring, [0, 1], [0, 1]);

              // Wasted region width
              const wastedWidth = (res.limit - res.used) * BAR_WIDTH;

              return (
                <div
                  key={res.name}
                  style={{
                    opacity: fadeIn,
                    width: 560,
                  }}
                >
                  {/* Resource name */}
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
                        width: 10,
                        height: 10,
                        borderRadius: 5,
                        backgroundColor: res.color,
                        flexShrink: 0,
                      }}
                    />
                    <span
                      style={{
                        fontSize: 18,
                        fontWeight: 700,
                        color: res.color,
                        fontFamily: fontFamilyBody,
                      }}
                    >
                      {res.name}
                    </span>
                  </div>

                  {/* Bar */}
                  <div
                    style={{
                      width: BAR_WIDTH,
                      height: BAR_HEIGHT,
                      backgroundColor: "rgba(255,255,255,0.04)",
                      borderRadius: 6,
                      overflow: "hidden",
                      display: "flex",
                      border: "1px solid rgba(255,255,255,0.08)",
                    }}
                  >
                    {/* Used section */}
                    <div
                      style={{
                        width: usedWidth,
                        height: BAR_HEIGHT,
                        backgroundColor: `${res.color}60`,
                        borderRight: `2px solid ${res.color}`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <span
                        style={{
                          fontSize: 12,
                          color: THEME.colors.textPrimary,
                          fontFamily: fontFamilyCode,
                          fontWeight: 600,
                        }}
                      >
                        Used
                      </span>
                    </div>
                    {/* Wasted/Available section */}
                    <div
                      style={{
                        flex: 1,
                        height: BAR_HEIGHT,
                        backgroundColor: "rgba(255,82,82,0.12)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <span
                        style={{
                          fontSize: 12,
                          color: THEME.colors.accentRed,
                          fontFamily: fontFamilyCode,
                          fontWeight: 600,
                          opacity: 0.8,
                        }}
                      >
                        Available
                      </span>
                    </div>
                  </div>

                  {/* Detail text */}
                  <div
                    style={{
                      fontSize: 13,
                      color: THEME.colors.textMuted,
                      fontFamily: fontFamilyCode,
                      marginTop: 4,
                    }}
                  >
                    {res.detail}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Bottom callout */}
          <div
            style={{
              marginTop: 24,
              padding: "12px 18px",
              backgroundColor: "rgba(255,82,82,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.accentRed}40`,
              opacity: calloutOpacity,
              width: 540,
            }}
          >
            <span
              style={{
                fontSize: 17,
                color: THEME.colors.accentRed,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
              }}
            >
              The most constraining resource determines your occupancy
            </span>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 60, width: 420 }}>
          <BulletPoint
            index={0}
            delay={2 * fps}
            text="More registers per thread = fewer threads per SM"
            subtext="Each thread's registers are private. Using 64 regs halves your max threads."
            icon="R"
          />
          <BulletPoint
            index={1}
            delay={4 * fps}
            text="More shared memory per block = fewer blocks per SM"
            subtext="Blocks compete for a fixed shared memory pool on each SM."
            icon="S"
          />
          <BulletPoint
            index={2}
            delay={6 * fps}
            text="Block size affects warp count"
            subtext="Threads are grouped into 32-thread warps. Each SM supports up to 48 warps."
            icon="T"
          />

          {/* Trade-off insight */}
          <div
            style={{
              marginTop: 24,
              padding: "14px 18px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: interpolate(
                frame - 8.5 * fps,
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
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              Use --ptxas-options=-v to see per-kernel resource usage at compile time.
            </span>
          </div>
        </div>
      }
    />
  );
};
