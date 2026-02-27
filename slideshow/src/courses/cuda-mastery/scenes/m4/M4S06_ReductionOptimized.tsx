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

interface OptCard {
  title: string;
  tagColor: string;
  tagText: string;
  description: string;
  code: string;
  speedup: string;
}

export const M4S06_ReductionOptimized: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const optimizations: OptCard[] = [
    {
      title: "First Add During Load",
      tagColor: THEME.colors.accentCyan,
      tagText: "2x fewer blocks",
      description: "Add two elements when loading from global memory. Each thread loads and sums two values before the reduction loop begins.",
      code: `int i = blockIdx.x * (blockDim.x * 2) + tid;
sdata[tid] = data[i] + data[i + blockDim.x];`,
      speedup: "~1.8x",
    },
    {
      title: "Warp Unrolling (Last 32)",
      tagColor: THEME.colors.nvidiaGreen,
      tagText: "No sync overhead",
      description: "When <= 32 threads remain, they are all in one warp. Warp-synchronous execution means __syncthreads() is unnecessary.",
      code: `if (tid < 32) {  // Last warp — no sync needed
    volatile float *vmem = sdata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
}`,
      speedup: "~1.4x",
    },
    {
      title: "Complete Loop Unrolling",
      tagColor: THEME.colors.accentPurple,
      tagText: "Zero loop overhead",
      description: "Template metaprogramming with compile-time block size. Compiler fully unrolls the loop, eliminating branch overhead.",
      code: `template <unsigned int blockSize>
__global__ void reduce(float *data, float *out) {
    // ...
    if (blockSize >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid+256];
        __syncthreads();
    }
    if (blockSize >= 256) { /* ... */ }
    // Compiler removes dead branches
}`,
      speedup: "~1.2x",
    },
  ];

  return (
    <SlideLayout variant="gradient" moduleNumber={4} slideNumber={6} totalSlides={18}>
      <SlideTitle
        title="Advanced Reduction Optimizations"
        subtitle="Three techniques that compound for massive speedups"
      />

      <div style={{ display: "flex", gap: 24, flex: 1 }}>
        {optimizations.map((opt, i) => {
          const cardDelay = 1 * fps + i * 1.2 * fps;
          const cardSpring = spring({
            frame: frame - cardDelay,
            fps,
            config: { damping: 200 },
          });
          const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
          const cardY = interpolate(cardSpring, [0, 1], [20, 0]);

          const lines = opt.code.split("\n");

          return (
            <div
              key={i}
              style={{
                flex: 1,
                opacity: cardOpacity,
                transform: `translateY(${cardY}px)`,
                backgroundColor: "rgba(255,255,255,0.03)",
                borderRadius: 12,
                border: `1px solid ${opt.tagColor}30`,
                padding: 20,
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
              }}
            >
              {/* Header */}
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10, flexShrink: 0 }}>
                <div
                  style={{
                    fontSize: 18,
                    fontWeight: 700,
                    color: THEME.colors.textPrimary,
                    fontFamily: fontFamilyBody,
                    flex: 1,
                  }}
                >
                  {opt.title}
                </div>
                <div
                  style={{
                    padding: "4px 10px",
                    backgroundColor: `${opt.tagColor}15`,
                    border: `1.5px solid ${opt.tagColor}60`,
                    borderRadius: 12,
                    fontSize: 12,
                    fontWeight: 700,
                    color: opt.tagColor,
                    fontFamily: fontFamilyBody,
                    flexShrink: 0,
                  }}
                >
                  {opt.tagText}
                </div>
              </div>

              {/* Description */}
              <div
                style={{
                  fontSize: 13,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  lineHeight: 1.5,
                  marginBottom: 12,
                  flexShrink: 0,
                }}
              >
                {opt.description}
              </div>

              {/* Code snippet */}
              <div
                style={{
                  backgroundColor: "#0d1117",
                  borderRadius: 8,
                  padding: "10px 14px",
                  flex: 1,
                  overflow: "hidden",
                }}
              >
                {lines.map((line, li) => (
                  <pre
                    key={li}
                    style={{
                      margin: 0,
                      fontFamily: fontFamilyCode,
                      fontSize: 11,
                      lineHeight: 1.5,
                      color: THEME.colors.textCode,
                      whiteSpace: "pre",
                    }}
                  >
                    {line}
                  </pre>
                ))}
              </div>

              {/* Speedup badge */}
              <div
                style={{
                  marginTop: 12,
                  textAlign: "center",
                  flexShrink: 0,
                }}
              >
                <span
                  style={{
                    fontSize: 24,
                    fontWeight: 800,
                    color: opt.tagColor,
                    fontFamily: fontFamilyBody,
                  }}
                >
                  {opt.speedup}
                </span>
                <span
                  style={{
                    fontSize: 13,
                    color: THEME.colors.textMuted,
                    fontFamily: fontFamilyBody,
                    marginLeft: 8,
                  }}
                >
                  speedup
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Bottom combined speedup */}
      <div
        style={{
          marginTop: 14,
          padding: "12px 24px",
          backgroundColor: "rgba(118,185,0,0.08)",
          borderRadius: 10,
          border: `1px solid ${THEME.colors.nvidiaGreen}40`,
          textAlign: "center",
          opacity: interpolate(
            frame - 5.5 * fps,
            [0, 0.5 * fps],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          ),
        }}
      >
        <span
          style={{
            fontSize: 18,
            color: THEME.colors.textPrimary,
            fontFamily: fontFamilyBody,
            fontWeight: 600,
          }}
        >
          Combined:{" "}
          <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 800, fontSize: 22 }}>
            ~3x total speedup
          </span>
          {" "}over naive reduction (V1 to fully optimized)
        </span>
      </div>
    </SlideLayout>
  );
};
