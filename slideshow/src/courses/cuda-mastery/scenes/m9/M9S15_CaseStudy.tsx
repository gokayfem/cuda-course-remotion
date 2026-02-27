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
  latency: string;
  latencyMs: number;
  color: string;
};

const optimizations: OptStep[] = [
  { label: "Naive PyTorch", latency: "100ms/token", latencyMs: 100, color: THEME.colors.accentRed },
  { label: "+ cuBLAS GEMM", latency: "40ms", latencyMs: 40, color: THEME.colors.accentOrange },
  { label: "+ Fused LayerNorm", latency: "32ms", latencyMs: 32, color: THEME.colors.accentYellow },
  { label: "+ Flash Attention", latency: "18ms", latencyMs: 18, color: THEME.colors.nvidiaGreen },
  { label: "+ KV Cache + Quantization", latency: "8ms", latencyMs: 8, color: THEME.colors.nvidiaGreenLight },
];

const MAX_LATENCY = 100;
const BAR_MAX_WIDTH = 620;

export const M9S15_CaseStudy: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={9}
      leftWidth="55%"
      left={
        <div style={{ width: 820 }}>
          <SlideTitle title="Case Study: LLaMA-Style Inference" />

          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
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

              const barWidth = (opt.latencyMs / MAX_LATENCY) * BAR_MAX_WIDTH * barProgress;

              return (
                <div key={opt.label} style={{ opacity: labelOpacity }}>
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
                        fontSize: 15,
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
                        fontSize: 15,
                        color: opt.color,
                        fontFamily: fontFamilyCode,
                        fontWeight: 700,
                      }}
                    >
                      {opt.latency}
                    </div>
                  </div>

                  <div
                    style={{
                      width: BAR_MAX_WIDTH + 100,
                      height: 26,
                      backgroundColor: "rgba(255,255,255,0.04)",
                      borderRadius: 6,
                      overflow: "hidden",
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

            {/* Speedup annotation */}
            <div
              style={{
                marginTop: 10,
                padding: "10px 16px",
                backgroundColor: "rgba(118,185,0,0.08)",
                borderRadius: 8,
                border: `1px solid ${THEME.colors.nvidiaGreen}30`,
                width: BAR_MAX_WIDTH + 100,
                opacity: interpolate(
                  frame - 8 * fps,
                  [0, 0.5 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <div
                style={{
                  fontSize: 18,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                  fontWeight: 700,
                }}
              >
                12.5x total speedup -- 100ms down to 8ms
              </div>
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ width: 480, marginTop: 80 }}>
          <BulletPoint
            text="12.5x total speedup"
            index={0}
            delay={8.5 * fps}
            highlight
          />
          <BulletPoint
            text="Flash Attention: single biggest win (44% reduction)"
            index={1}
            delay={8.5 * fps}
          />
          <BulletPoint
            text="Fusion: cumulative 20% improvement"
            index={2}
            delay={8.5 * fps}
          />
          <BulletPoint
            text="Quantization: 2x from INT8/INT4"
            index={3}
            delay={8.5 * fps}
          />

          {/* Impact breakdown */}
          <div
            style={{
              marginTop: 28,
              padding: "16px 20px",
              backgroundColor: "rgba(255,255,255,0.03)",
              borderRadius: 10,
              border: "1px solid rgba(255,255,255,0.08)",
              opacity: interpolate(
                frame - 11 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div
              style={{
                fontSize: 14,
                fontWeight: 700,
                color: THEME.colors.accentPurple,
                fontFamily: fontFamilyBody,
                marginBottom: 10,
              }}
            >
              Impact Breakdown
            </div>
            {[
              { opt: "cuBLAS", impact: "60% reduction", color: THEME.colors.accentOrange },
              { opt: "Fusion", impact: "20% reduction", color: THEME.colors.accentYellow },
              { opt: "Flash Attn", impact: "44% reduction", color: THEME.colors.nvidiaGreen },
              { opt: "Quantization", impact: "56% reduction", color: THEME.colors.nvidiaGreenLight },
            ].map((item) => (
              <div
                key={item.opt}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "4px 0",
                }}
              >
                <span
                  style={{
                    fontSize: 14,
                    color: item.color,
                    fontFamily: fontFamilyCode,
                    fontWeight: 600,
                  }}
                >
                  {item.opt}
                </span>
                <span
                  style={{
                    fontSize: 14,
                    color: THEME.colors.textSecondary,
                    fontFamily: fontFamilyBody,
                  }}
                >
                  {item.impact}
                </span>
              </div>
            ))}
          </div>
        </div>
      }
    />
  );
};
