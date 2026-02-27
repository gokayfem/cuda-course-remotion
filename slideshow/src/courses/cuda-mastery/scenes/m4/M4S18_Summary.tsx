import React from "react";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

const takeaways = [
  {
    icon: "\u2B07",
    title: "Reduction",
    text: "O(log N) steps, tree pattern, warp unrolling for the last 32 elements",
    color: THEME.colors.accentBlue,
  },
  {
    icon: "\u2192",
    title: "Scan",
    text: "Blelloch algorithm is work-efficient -- the building block for many algorithms",
    color: THEME.colors.accentCyan,
  },
  {
    icon: "\u2581\u2583\u2585\u2587",
    title: "Histogram",
    text: "Shared memory privatization for speed -- merge to global at the end",
    color: THEME.colors.accentOrange,
  },
  {
    icon: "\u2702",
    title: "Compaction",
    text: "Predicate \u2192 Scan \u2192 Scatter -- three-kernel pipeline for filtering",
    color: THEME.colors.accentPurple,
  },
  {
    icon: "\u26A1",
    title: "Bandwidth",
    text: "Well-optimized reduction reaches ~90% of peak memory bandwidth",
    color: THEME.colors.nvidiaGreen,
  },
  {
    icon: "\u2699",
    title: "Composability",
    text: "These patterns combine: dot product = multiply + reduce, softmax = reduce + reduce + map",
    color: THEME.colors.accentYellow,
  },
  {
    icon: "\u2699",
    title: "ML Applications",
    text: "Softmax, loss computation, sparse attention, quantization calibration",
    color: THEME.colors.accentPink,
  },
  {
    icon: "\u2B06",
    title: "Next Up",
    text: "Module 5: Performance Optimization -- occupancy, roofline model, profiling",
    color: THEME.colors.accentCyan,
  },
];

export const M4S18_Summary: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const congratsOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const nextOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout moduleNumber={4} variant="accent">
      <SlideTitle
        title="Module 4: Complete"
        subtitle="Parallel Patterns -- Reduction, Scan, Histogram & Compaction"
      />

      {/* Key Takeaways - 2x4 grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, flex: 1 }}>
        {takeaways.map((item, i) => {
          const itemDelay = 0.8 * fps + i * 0.25 * fps;
          const itemSpring = spring({
            frame: frame - itemDelay,
            fps,
            config: { damping: 200 },
          });
          const itemOpacity = interpolate(itemSpring, [0, 1], [0, 1]);
          const itemX = interpolate(itemSpring, [0, 1], [-20, 0]);

          return (
            <div
              key={i}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 14,
                padding: "12px 18px",
                backgroundColor: `${item.color}08`,
                borderLeft: `4px solid ${item.color}`,
                borderRadius: 8,
                opacity: itemOpacity,
                transform: `translateX(${itemX}px)`,
              }}
            >
              <span
                style={{
                  fontSize: 24,
                  flexShrink: 0,
                  width: 36,
                  textAlign: "center",
                }}
              >
                {item.icon}
              </span>
              <div>
                <div
                  style={{
                    fontSize: 16,
                    fontWeight: 700,
                    color: item.color,
                    fontFamily: fontFamilyBody,
                    marginBottom: 2,
                  }}
                >
                  {item.title}
                </div>
                <div
                  style={{
                    fontSize: 14,
                    color: THEME.colors.textSecondary,
                    fontFamily: fontFamilyBody,
                    lineHeight: 1.4,
                  }}
                >
                  {item.text}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Congratulations + Next Module */}
      <div style={{ marginTop: 16, display: "flex", gap: 16 }}>
        {/* Congrats */}
        <div
          style={{
            flex: 1,
            padding: "16px 24px",
            background: "linear-gradient(135deg, rgba(118,185,0,0.1), rgba(79,195,247,0.06))",
            borderRadius: 12,
            border: `1px solid ${THEME.colors.nvidiaGreen}30`,
            opacity: congratsOpacity,
            display: "flex",
            alignItems: "center",
            gap: 16,
          }}
        >
          <span style={{ fontSize: 36 }}>{"\uD83C\uDF89"}</span>
          <div>
            <div style={{ fontSize: 20, fontWeight: 700, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyBody }}>
              Congratulations!
            </div>
            <div style={{ fontSize: 15, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, marginTop: 2 }}>
              You now have the fundamental parallel algorithm toolkit for GPU programming.
            </div>
          </div>
        </div>

        {/* Next module teaser */}
        <div
          style={{
            flex: 1,
            padding: "16px 24px",
            background: "linear-gradient(135deg, rgba(24,255,255,0.08), rgba(79,195,247,0.06))",
            borderRadius: 12,
            border: `1px solid ${THEME.colors.accentCyan}30`,
            opacity: nextOpacity,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div>
            <div style={{ fontSize: 14, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginBottom: 4 }}>
              Coming Next
            </div>
            <div style={{ fontSize: 22, fontWeight: 700, color: THEME.colors.accentCyan, fontFamily: fontFamilyBody }}>
              Module 5: Performance Optimization
            </div>
            <div style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, marginTop: 3 }}>
              Occupancy, roofline model, Nsight profiling -- squeezing every FLOP from the GPU
            </div>
          </div>
          <div
            style={{
              fontSize: 40,
              color: THEME.colors.accentCyan,
              opacity: 0.5,
            }}
          >
            {"\u2192"}
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
