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
import { fontFamilyBody } from "../../../../styles/fonts";

type TakeawayCard = {
  title: string;
  text: string;
  color: string;
};

const takeaways: TakeawayCard[] = [
  {
    title: "Naive",
    text: "1 element/thread, 0.3% peak. Zero data reuse.",
    color: THEME.colors.accentRed,
  },
  {
    title: "Tiling",
    text: "Shared memory tiles, 3% peak. TILE x data reuse.",
    color: THEME.colors.accentYellow,
  },
  {
    title: "Register Tiling",
    text: "TM x TN per thread, 30% peak. Outer product pattern.",
    color: THEME.colors.accentBlue,
  },
  {
    title: "Vectorized",
    text: "float4 + double buffer, 85% peak. Hide latency.",
    color: THEME.colors.accentOrange,
  },
  {
    title: "Tensor Cores",
    text: "4x4x4 MMA, 16x throughput. cuBLAS uses automatically.",
    color: THEME.colors.nvidiaGreen,
  },
  {
    title: "CUTLASS",
    text: "Template library for custom high-performance GEMM.",
    color: THEME.colors.accentPurple,
  },
];

export const M8S18_Summary: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const nextOpacity = interpolate(
    frame - 8 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout moduleNumber={8} variant="accent">
      <SlideTitle
        title="Module 8 Summary"
        subtitle="Matrix Multiplication Deep Dive -- from naive to near-peak"
      />

      {/* 2x3 grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: 16,
          flex: 1,
          width: 1776,
        }}
      >
        {takeaways.map((item, i) => {
          const itemDelay = 0.8 * fps + i * 0.3 * fps;
          const itemSpring = spring({
            frame: frame - itemDelay,
            fps,
            config: { damping: 200 },
          });
          const itemOpacity = interpolate(itemSpring, [0, 1], [0, 1]);
          const itemScale = interpolate(itemSpring, [0, 1], [0.92, 1]);

          return (
            <div
              key={item.title}
              style={{
                padding: "20px 24px",
                backgroundColor: `${item.color}08`,
                borderLeft: `4px solid ${item.color}`,
                borderRadius: 10,
                opacity: itemOpacity,
                transform: `scale(${itemScale})`,
                display: "flex",
                flexDirection: "column",
              }}
            >
              <div
                style={{
                  fontSize: 20,
                  fontWeight: 700,
                  color: item.color,
                  fontFamily: fontFamilyBody,
                  marginBottom: 10,
                }}
              >
                {item.title}
              </div>
              <div
                style={{
                  fontSize: 15,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  lineHeight: 1.5,
                }}
              >
                {item.text}
              </div>
            </div>
          );
        })}
      </div>

      {/* Next module teaser */}
      <div
        style={{
          marginTop: 20,
          padding: "18px 28px",
          background:
            "linear-gradient(135deg, rgba(24,255,255,0.08), rgba(79,195,247,0.06))",
          borderRadius: 12,
          border: `1px solid ${THEME.colors.accentCyan}30`,
          opacity: nextOpacity,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          width: 1776,
        }}
      >
        <div>
          <div
            style={{
              fontSize: 14,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
              marginBottom: 4,
            }}
          >
            Coming Next
          </div>
          <div
            style={{
              fontSize: 24,
              fontWeight: 700,
              color: THEME.colors.accentCyan,
              fontFamily: fontFamilyBody,
            }}
          >
            Module 9: Attention & Transformer Kernels
          </div>
          <div
            style={{
              fontSize: 15,
              color: THEME.colors.textSecondary,
              fontFamily: fontFamilyBody,
              marginTop: 4,
            }}
          >
            FlashAttention, fused softmax, KV-cache optimization, and multi-head attention kernels
          </div>
        </div>
        <div
          style={{
            fontSize: 44,
            color: THEME.colors.accentCyan,
            opacity: 0.5,
          }}
        >
          {"\u2192"}
        </div>
      </div>
    </SlideLayout>
  );
};
