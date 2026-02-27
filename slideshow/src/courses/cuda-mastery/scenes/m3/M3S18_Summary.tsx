import React from "react";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

const takeaways = [
  {
    icon: "\u26A1",
    title: "Warps",
    text: "32 threads execute in lockstep -- the fundamental unit of GPU execution",
    color: THEME.colors.accentBlue,
  },
  {
    icon: "\u26D4",
    title: "Divergence",
    text: "Branches within a warp serialize execution -- keep warps uniform",
    color: THEME.colors.accentRed,
  },
  {
    icon: "\u23F8",
    title: "Barriers",
    text: "__syncthreads() for block-level sync -- ALL threads must reach it",
    color: THEME.colors.accentOrange,
  },
  {
    icon: "\u269B",
    title: "Atomics",
    text: "Thread-safe RMW operations -- use shared memory atomics when possible",
    color: THEME.colors.accentYellow,
  },
  {
    icon: "\u21C4",
    title: "Shuffle",
    text: "Direct register-to-register lane exchange -- fastest intra-warp communication",
    color: THEME.colors.nvidiaGreen,
  },
  {
    icon: "\u2714",
    title: "Vote",
    text: "Warp-level ballot, any, all -- collective boolean decisions in one cycle",
    color: THEME.colors.accentCyan,
  },
  {
    icon: "\u2699",
    title: "Cooperative Groups",
    text: "Modern, type-safe synchronization API -- flexible granularity",
    color: THEME.colors.accentPurple,
  },
  {
    icon: "\u2B06",
    title: "Block Reduction",
    text: "Warp shuffle + shared memory = fastest reduction pattern",
    color: THEME.colors.accentPink,
  },
];

export const M3S18_Summary: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleSpring = spring({
    frame,
    fps,
    config: { damping: 200 },
  });
  const titleOpacity = interpolate(titleSpring, [0, 1], [0, 1]);

  const lineWidth = interpolate(frame, [0, 1 * fps], [0, 300], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

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
    <SlideLayout moduleNumber={3} variant="accent">
      <SlideTitle
        title="Module 3: Complete"
        subtitle="Thread Synchronization & Execution Model"
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
              You now understand how the GPU really executes your code at the hardware level.
            </div>
          </div>
        </div>

        {/* Next module teaser */}
        <div
          style={{
            flex: 1,
            padding: "16px 24px",
            background: "linear-gradient(135deg, rgba(179,136,255,0.08), rgba(255,171,64,0.06))",
            borderRadius: 12,
            border: `1px solid ${THEME.colors.accentPurple}30`,
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
            <div style={{ fontSize: 22, fontWeight: 700, color: THEME.colors.accentPurple, fontFamily: fontFamilyBody }}>
              Module 4: Parallel Patterns
            </div>
            <div style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, marginTop: 3 }}>
              Reduction, scan, sort, histogram -- the building blocks of GPU algorithms
            </div>
          </div>
          <div
            style={{
              fontSize: 40,
              color: THEME.colors.accentPurple,
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
