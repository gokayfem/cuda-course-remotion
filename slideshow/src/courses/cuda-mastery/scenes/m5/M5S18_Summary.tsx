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

type TakeawayCard = {
  title: string;
  text: string;
  color: string;
  icon: string;
};

const takeaways: TakeawayCard[] = [
  {
    title: "Occupancy",
    text: "Active warps / max warps. Use API to find optimal block size",
    color: THEME.colors.accentBlue,
    icon: "\u25CB",
  },
  {
    title: "Bandwidth",
    text: "Most kernels are memory-bound. Vectorize loads, ensure coalescing",
    color: THEME.colors.accentCyan,
    icon: "\u2261",
  },
  {
    title: "ILP",
    text: "Multiple elements per thread exposes instruction-level parallelism",
    color: THEME.colors.accentOrange,
    icon: "\u26A1",
  },
  {
    title: "Unrolling",
    text: "#pragma unroll reduces loop overhead, exposes ILP",
    color: THEME.colors.accentPurple,
    icon: "\u21BB",
  },
  {
    title: "Roofline",
    text: "AI = FLOPs/byte. Determines if memory or compute bound",
    color: THEME.colors.nvidiaGreen,
    icon: "\u25B3",
  },
  {
    title: "Profiling",
    text: "Always profile before optimizing. ncu for kernels, nsys for timeline",
    color: THEME.colors.accentYellow,
    icon: "\uD83D\uDD0D",
  },
];

export const M5S18_Summary: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const nextOpacity = interpolate(
    frame - 8 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout moduleNumber={5} variant="accent">
      <SlideTitle
        title="Module 5 Summary"
        subtitle="Performance Optimization -- squeezing every FLOP from the GPU"
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
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
                <span
                  style={{
                    fontSize: 24,
                    width: 36,
                    textAlign: "center",
                    flexShrink: 0,
                  }}
                >
                  {item.icon}
                </span>
                <div
                  style={{
                    fontSize: 20,
                    fontWeight: 700,
                    color: item.color,
                    fontFamily: fontFamilyBody,
                  }}
                >
                  {item.title}
                </div>
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
          background: "linear-gradient(135deg, rgba(24,255,255,0.08), rgba(79,195,247,0.06))",
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
            Module 6: Streams & Concurrency
          </div>
          <div
            style={{
              fontSize: 15,
              color: THEME.colors.textSecondary,
              fontFamily: fontFamilyBody,
              marginTop: 4,
            }}
          >
            Overlapping compute and data transfer, multi-stream execution, and async operations
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
