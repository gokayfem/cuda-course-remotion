import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

type PipelineBlock = {
  label: string;
  sublabel: string;
  color: string;
  isCustom: boolean;
};

const pipeline: PipelineBlock[] = [
  {
    label: "Fused Residual + RMSNorm",
    sublabel: "Custom kernel",
    color: THEME.colors.nvidiaGreen,
    isCustom: true,
  },
  {
    label: "cuBLAS: QKV Projection",
    sublabel: "GEMM",
    color: THEME.colors.textMuted,
    isCustom: false,
  },
  {
    label: "Flash Attention",
    sublabel: "Fused softmax + matmul",
    color: THEME.colors.nvidiaGreenLight,
    isCustom: true,
  },
  {
    label: "cuBLAS: Output Projection",
    sublabel: "GEMM",
    color: THEME.colors.textMuted,
    isCustom: false,
  },
  {
    label: "Fused Residual + RMSNorm",
    sublabel: "Custom kernel",
    color: THEME.colors.nvidiaGreen,
    isCustom: true,
  },
  {
    label: "cuBLAS: Up Projection",
    sublabel: "GEMM",
    color: THEME.colors.textMuted,
    isCustom: false,
  },
  {
    label: "Fused Bias + SiLU + Gate",
    sublabel: "Custom kernel",
    color: THEME.colors.nvidiaGreen,
    isCustom: true,
  },
  {
    label: "cuBLAS: Down Projection",
    sublabel: "GEMM",
    color: THEME.colors.textMuted,
    isCustom: false,
  },
];

type TimingSlice = {
  label: string;
  pct: number;
  color: string;
};

const timing: TimingSlice[] = [
  { label: "GEMM", pct: 70, color: THEME.colors.accentBlue },
  { label: "Custom Kernels", pct: 20, color: THEME.colors.nvidiaGreen },
  { label: "Overhead", pct: 10, color: THEME.colors.textMuted },
];

export const M9S14_TransformerOptimizations: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const timingDelay = 8 * fps;
  const timingOpacity = interpolate(
    frame - timingDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="dark" moduleNumber={9}>
      <SlideTitle
        title="Putting It All Together: Optimized Transformer"
        subtitle="Full pipeline of a modern optimized transformer block"
      />

      {/* Pipeline flow */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 10,
          alignItems: "center",
          width: 1776,
          marginBottom: 28,
        }}
      >
        {pipeline.map((block, i) => {
          const blockDelay = 1 * fps + i * 0.4 * fps;
          const blockSpring = spring({
            frame: frame - blockDelay,
            fps,
            config: { damping: 200 },
          });
          const blockOpacity = interpolate(blockSpring, [0, 1], [0, 1]);
          const blockScale = interpolate(blockSpring, [0, 1], [0.9, 1]);

          return (
            <React.Fragment key={`${block.label}-${i}`}>
              <div
                style={{
                  padding: "12px 18px",
                  backgroundColor: block.isCustom
                    ? `${block.color}12`
                    : "rgba(255,255,255,0.03)",
                  border: `1px solid ${block.color}40`,
                  borderRadius: 8,
                  opacity: blockOpacity,
                  transform: `scale(${blockScale})`,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  minWidth: 160,
                }}
              >
                <div
                  style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color: block.isCustom ? block.color : THEME.colors.textPrimary,
                    fontFamily: fontFamilyBody,
                    textAlign: "center",
                    lineHeight: 1.3,
                  }}
                >
                  {block.label}
                </div>
                <div
                  style={{
                    fontSize: 11,
                    color: block.isCustom
                      ? `${block.color}CC`
                      : THEME.colors.textSecondary,
                    fontFamily: fontFamilyCode,
                    marginTop: 4,
                  }}
                >
                  {block.sublabel}
                </div>
              </div>

              {/* Arrow between blocks */}
              {i < pipeline.length - 1 && (
                <div
                  style={{
                    width: 20,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    opacity: blockOpacity,
                  }}
                >
                  <div
                    style={{
                      fontSize: 16,
                      color: THEME.colors.textMuted,
                    }}
                  >
                    {"\u2192"}
                  </div>
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Timing breakdown */}
      <div
        style={{
          width: 1776,
          opacity: timingOpacity,
        }}
      >
        <div
          style={{
            fontSize: 16,
            fontWeight: 700,
            color: THEME.colors.accentCyan,
            fontFamily: fontFamilyBody,
            marginBottom: 12,
          }}
        >
          Timing Breakdown
        </div>

        {/* Stacked bar */}
        <div
          style={{
            display: "flex",
            width: 1776,
            height: 40,
            borderRadius: 8,
            overflow: "hidden",
            marginBottom: 12,
          }}
        >
          {timing.map((slice) => {
            const barWidth = interpolate(
              frame - timingDelay,
              [0, 1 * fps],
              [0, (slice.pct / 100) * 1776],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );

            return (
              <div
                key={slice.label}
                style={{
                  width: barWidth,
                  height: "100%",
                  backgroundColor: `${slice.color}60`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <span
                  style={{
                    fontSize: 14,
                    fontWeight: 700,
                    color: THEME.colors.textPrimary,
                    fontFamily: fontFamilyBody,
                    whiteSpace: "nowrap",
                  }}
                >
                  {slice.label}: {slice.pct}%
                </span>
              </div>
            );
          })}
        </div>

        {/* Legend */}
        <div style={{ display: "flex", gap: 28 }}>
          {timing.map((slice) => (
            <div
              key={slice.label}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <div
                style={{
                  width: 14,
                  height: 14,
                  borderRadius: 3,
                  backgroundColor: `${slice.color}60`,
                }}
              />
              <span
                style={{
                  fontSize: 14,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                }}
              >
                {slice.label} ({slice.pct}%)
              </span>
            </div>
          ))}
        </div>
      </div>
    </SlideLayout>
  );
};
