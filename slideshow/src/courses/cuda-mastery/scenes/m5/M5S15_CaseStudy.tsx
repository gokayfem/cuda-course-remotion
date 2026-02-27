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

type OptStep = {
  label: string;
  bandwidth: number;
  color: string;
};

const optimizations: OptStep[] = [
  { label: "Naive (1 element/thread)", bandwidth: 120, color: THEME.colors.accentRed },
  { label: "+ Coalesced access", bandwidth: 280, color: THEME.colors.accentOrange },
  { label: "+ float4 vectorized", bandwidth: 450, color: THEME.colors.accentYellow },
  { label: "+ Grid-stride + unroll", bandwidth: 520, color: THEME.colors.nvidiaGreen },
];

const PEAK_BW = 600;
const BAR_MAX_WIDTH = 580;

export const M5S15_CaseStudy: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={5}
      leftWidth="55%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle title="Case Study: Optimizing Vector Scale" />

          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {optimizations.map((opt, i) => {
              const barDelay = 1 * fps + i * 1.2 * fps;
              const barProgress = interpolate(
                frame - barDelay,
                [0, 1 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              const labelSpring = spring({
                frame: frame - barDelay,
                fps,
                config: { damping: 200 },
              });
              const labelOpacity = interpolate(labelSpring, [0, 1], [0, 1]);

              const barWidth = (opt.bandwidth / PEAK_BW) * BAR_MAX_WIDTH * barProgress;

              return (
                <div key={opt.label} style={{ opacity: labelOpacity }}>
                  {/* Step label */}
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "baseline",
                      marginBottom: 6,
                      width: BAR_MAX_WIDTH + 100,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 16,
                        color: THEME.colors.textPrimary,
                        fontFamily: fontFamilyBody,
                        fontWeight: 500,
                      }}
                    >
                      <span style={{ color: opt.color, fontWeight: 700, marginRight: 8 }}>
                        {i + 1}.
                      </span>
                      {opt.label}
                    </div>
                    <div
                      style={{
                        fontSize: 16,
                        color: opt.color,
                        fontFamily: fontFamilyCode,
                        fontWeight: 700,
                      }}
                    >
                      {Math.round(opt.bandwidth * barProgress)} GB/s
                    </div>
                  </div>

                  {/* Bar container */}
                  <div
                    style={{
                      width: BAR_MAX_WIDTH + 100,
                      height: 28,
                      backgroundColor: "rgba(255,255,255,0.04)",
                      borderRadius: 6,
                      overflow: "hidden",
                      position: "relative",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: barWidth,
                        backgroundColor: `${opt.color}CC`,
                        borderRadius: 6,
                        boxShadow: `0 0 12px ${opt.color}30`,
                      }}
                    />
                  </div>
                </div>
              );
            })}

            {/* Peak bandwidth dashed line */}
            <div
              style={{
                marginTop: 8,
                opacity: interpolate(
                  frame - 6 * fps,
                  [0, 0.5 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  width: BAR_MAX_WIDTH + 100,
                }}
              >
                <div
                  style={{
                    flex: 1,
                    height: 2,
                    borderTop: `2px dashed ${THEME.colors.textMuted}`,
                  }}
                />
                <span
                  style={{
                    fontSize: 14,
                    color: THEME.colors.textMuted,
                    fontFamily: fontFamilyCode,
                    flexShrink: 0,
                  }}
                >
                  Peak: {PEAK_BW} GB/s
                </span>
              </div>
            </div>

            {/* Speedup annotation */}
            <div
              style={{
                marginTop: 12,
                padding: "10px 16px",
                backgroundColor: "rgba(118,185,0,0.08)",
                borderRadius: 8,
                border: `1px solid ${THEME.colors.nvidiaGreen}30`,
                width: BAR_MAX_WIDTH + 100,
                opacity: interpolate(
                  frame - 7 * fps,
                  [0, 0.5 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <div style={{ fontSize: 18, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyBody, fontWeight: 700 }}>
                4.3x speedup -- 87% of peak bandwidth
              </div>
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ width: 460, marginTop: 80 }}>
          <div
            style={{
              fontSize: 20,
              fontWeight: 700,
              color: THEME.colors.accentCyan,
              fontFamily: fontFamilyBody,
              marginBottom: 20,
              opacity: interpolate(
                frame - 7 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            Key Takeaways
          </div>

          <BulletPoint
            text="4.3x improvement from systematic optimization"
            index={0}
            delay={7.5 * fps}
            highlight
          />
          <BulletPoint
            text="Each step targeted a specific bottleneck"
            index={1}
            delay={7.5 * fps}
          />
          <BulletPoint
            text="Profiling guided every decision"
            index={2}
            delay={7.5 * fps}
          />
          <BulletPoint
            text="Achieved 87% of peak -- excellent!"
            index={3}
            delay={7.5 * fps}
            highlight
          />

          {/* What each optimization addressed */}
          <div
            style={{
              marginTop: 32,
              padding: "16px 20px",
              backgroundColor: "rgba(255,255,255,0.03)",
              borderRadius: 10,
              border: `1px solid rgba(255,255,255,0.08)`,
              opacity: interpolate(
                frame - 10 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div style={{ fontSize: 14, fontWeight: 700, color: THEME.colors.accentPurple, fontFamily: fontFamilyBody, marginBottom: 10 }}>
              Bottleneck Addressed
            </div>
            {[
              { step: "Coalescing", fix: "Reduced memory transactions" },
              { step: "float4", fix: "Fewer load instructions" },
              { step: "Grid-stride", fix: "Better warp utilization + ILP" },
            ].map((item, i) => (
              <div key={item.step} style={{ display: "flex", gap: 8, marginBottom: 6 }}>
                <span style={{ fontSize: 13, color: THEME.colors.accentPurple, fontFamily: fontFamilyCode, fontWeight: 600, width: 100, flexShrink: 0 }}>
                  {item.step}
                </span>
                <span style={{ fontSize: 13, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody }}>
                  {item.fix}
                </span>
              </div>
            ))}
          </div>
        </div>
      }
    />
  );
};
