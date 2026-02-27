import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

const CELL_SIZE = 18;
const CELL_GAP = 2;
const THREADS_PER_WARP = 32;

type PatternConfig = {
  label: string;
  rating: string;
  ratingColor: string;
  code: string;
  getActive: (threadIdx: number, warpIdx: number) => boolean;
  description: string;
};

const PATTERNS: PatternConfig[] = [
  {
    label: "Pattern 1 (BAD)",
    rating: "BAD",
    ratingColor: THEME.colors.accentRed,
    code: "if (threadIdx.x % 2 == 0)",
    getActive: (threadIdx) => threadIdx % 2 === 0,
    description: "Every warp diverges. 50% utilization on each pass.",
  },
  {
    label: "Pattern 2 (BETTER)",
    rating: "BETTER",
    ratingColor: THEME.colors.accentOrange,
    code: "if (threadIdx.x < 16)",
    getActive: (threadIdx) => threadIdx < 16,
    description:
      "First half active, second half masked. Still diverges within every warp.",
  },
  {
    label: "Pattern 3 (BEST)",
    rating: "BEST",
    ratingColor: THEME.colors.nvidiaGreen,
    code: "if (threadIdx.x / 32 < N)",
    getActive: (_threadIdx, warpIdx) => warpIdx < 2,
    description:
      "Entire warps take same path. No divergence within any warp!",
  },
];

export const M3S05_DivergencePatterns: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const renderWarpRow = (
    warpIdx: number,
    getActive: (threadIdx: number, warpIdx: number) => boolean,
    activeColor: string,
    delay: number
  ) => {
    const rowSpring = spring({
      frame: frame - delay,
      fps,
      config: { damping: 200 },
    });
    const rowOpacity = interpolate(rowSpring, [0, 1], [0, 1]);

    const threads = Array.from({ length: THREADS_PER_WARP }, (_, i) => {
      const globalIdx = warpIdx * THREADS_PER_WARP + i;
      return getActive(globalIdx, warpIdx);
    });

    const activeCount = threads.filter(Boolean).length;
    const allSame = activeCount === 0 || activeCount === THREADS_PER_WARP;

    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          opacity: rowOpacity,
        }}
      >
        <span
          style={{
            fontSize: 11,
            color: THEME.colors.textMuted,
            fontFamily: fontFamilyCode,
            width: 44,
            flexShrink: 0,
          }}
        >
          W{warpIdx}
        </span>
        <div style={{ display: "flex", gap: CELL_GAP }}>
          {threads.map((active, i) => (
            <div
              key={i}
              style={{
                width: CELL_SIZE,
                height: CELL_SIZE,
                borderRadius: 2,
                backgroundColor: active
                  ? `${activeColor}40`
                  : "rgba(255,255,255,0.03)",
                border: `1px solid ${active ? `${activeColor}80` : "rgba(255,255,255,0.06)"}`,
                fontSize: 8,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: active ? activeColor : "rgba(255,255,255,0.12)",
                fontFamily: fontFamilyCode,
              }}
            />
          ))}
        </div>
        <span
          style={{
            fontSize: 12,
            color: allSame ? THEME.colors.nvidiaGreen : THEME.colors.accentRed,
            fontFamily: fontFamilyCode,
            fontWeight: 700,
            flexShrink: 0,
            width: 80,
          }}
        >
          {allSame ? "No div." : "DIVERGES"}
        </span>
      </div>
    );
  };

  const renderPattern = (pattern: PatternConfig, patternIdx: number) => {
    const baseDelay = 1 * fps + patternIdx * 2 * fps;
    const patternSpring = spring({
      frame: frame - baseDelay,
      fps,
      config: { damping: 200 },
    });
    const patternOpacity = interpolate(patternSpring, [0, 1], [0, 1]);

    return (
      <div
        key={patternIdx}
        style={{
          opacity: patternOpacity,
          padding: "12px 16px",
          backgroundColor: "rgba(255,255,255,0.02)",
          border: `1px solid ${pattern.ratingColor}30`,
          borderRadius: 10,
          display: "flex",
          flexDirection: "column",
          gap: 6,
        }}
      >
        {/* Header */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            marginBottom: 2,
          }}
        >
          <div
            style={{
              padding: "3px 10px",
              backgroundColor: `${pattern.ratingColor}15`,
              border: `1px solid ${pattern.ratingColor}50`,
              borderRadius: 4,
              fontSize: 13,
              color: pattern.ratingColor,
              fontFamily: fontFamilyBody,
              fontWeight: 700,
            }}
          >
            {pattern.rating}
          </div>
          <code
            style={{
              fontSize: 14,
              color: THEME.colors.textCode,
              fontFamily: fontFamilyCode,
              backgroundColor: "rgba(255,255,255,0.05)",
              padding: "2px 8px",
              borderRadius: 4,
            }}
          >
            {pattern.code}
          </code>
        </div>

        {/* Warp visualizations (4 warps = 128 threads block) */}
        <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
          {Array.from({ length: 4 }).map((_, warpIdx) =>
            renderWarpRow(
              warpIdx,
              pattern.getActive,
              pattern.ratingColor,
              baseDelay + 0.3 * fps + warpIdx * 0.15 * fps
            )
          )}
        </div>

        {/* Description */}
        <div
          style={{
            fontSize: 14,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
            marginTop: 2,
          }}
        >
          {pattern.description}
        </div>
      </div>
    );
  };

  return (
    <SlideLayout variant="gradient" moduleNumber={3}>
      <SlideTitle
        title="Divergence Patterns"
        subtitle="Structure branches so entire warps take the same path"
      />

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 14,
          flex: 1,
        }}
      >
        {PATTERNS.map((pattern, i) => renderPattern(pattern, i))}
      </div>

      {/* Key takeaway */}
      <div
        style={{
          marginTop: 8,
          padding: "10px 24px",
          backgroundColor: "rgba(118,185,0,0.10)",
          borderRadius: 10,
          border: `2px solid ${THEME.colors.nvidiaGreen}50`,
          textAlign: "center",
          opacity: interpolate(
            frame - 7 * fps,
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
            fontWeight: 700,
          }}
        >
          Rule of thumb:{" "}
          <span style={{ color: THEME.colors.nvidiaGreen }}>
            branch on warp boundaries
          </span>
          , not on individual thread indices.
        </span>
      </div>
    </SlideLayout>
  );
};
