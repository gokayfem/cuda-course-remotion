import React from "react";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText, BulletPoint } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

const GroupBox: React.FC<{
  label: string;
  sublabel: string;
  color: string;
  delay: number;
  icon: string;
}> = ({ label, sublabel, color, delay, icon }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const boxSpring = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });
  const opacity = interpolate(boxSpring, [0, 1], [0, 1]);
  const scale = interpolate(boxSpring, [0, 1], [0.9, 1]);

  return (
    <div
      style={{
        flex: 1,
        padding: "14px 16px",
        backgroundColor: `${color}08`,
        border: `2px solid ${color}40`,
        borderRadius: 10,
        opacity,
        transform: `scale(${scale})`,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 6,
      }}
    >
      <span style={{ fontSize: 28 }}>{icon}</span>
      <span
        style={{
          fontSize: 16,
          fontWeight: 700,
          color,
          fontFamily: fontFamilyCode,
          textAlign: "center",
        }}
      >
        {label}
      </span>
      <span
        style={{
          fontSize: 13,
          color: THEME.colors.textSecondary,
          fontFamily: fontFamilyBody,
          textAlign: "center",
          lineHeight: 1.4,
        }}
      >
        {sublabel}
      </span>
    </div>
  );
};

export const M3S13_CooperativeGroups: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const groups = [
    {
      label: "thread_block",
      sublabel: "All threads in a block",
      color: THEME.colors.accentBlue,
      icon: "\u25A6",
    },
    {
      label: "coalesced_threads",
      sublabel: "Active threads in current warp",
      color: THEME.colors.nvidiaGreen,
      icon: "\u25C9",
    },
    {
      label: "thread_block_tile<N>",
      sublabel: "Fixed-size sub-warp groups",
      color: THEME.colors.accentOrange,
      icon: "\u25A3",
    },
    {
      label: "grid_group",
      sublabel: "All threads across entire grid",
      color: THEME.colors.accentPurple,
      icon: "\u25C8",
    },
  ];

  const whyBetterOpacity = interpolate(
    frame - 8 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout moduleNumber={3}>
      <SlideTitle
        title="Cooperative Groups"
        subtitle="A modern, flexible API for thread synchronization (CUDA 9+)"
      />

      <div style={{ display: "flex", gap: 32, flex: 1 }}>
        {/* Left: overview + groups */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text="Group Types"
            delay={0.5 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 12 }}
          />

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 20 }}>
            {groups.map((g, i) => (
              <GroupBox
                key={i}
                label={g.label}
                sublabel={g.sublabel}
                color={g.color}
                delay={(1 + i * 0.4) * fps}
                icon={g.icon}
              />
            ))}
          </div>

          <CodeBlock
            delay={3 * fps}
            title="cooperative_groups.cu"
            fontSize={14}
            code={`#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel(float* data, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp =
        cg::tiled_partition<32>(block);

    // Flexible sync -- only sync the group you need
    float val = data[threadIdx.x];
    val += warp.shfl_down(val, 16);
    val += warp.shfl_down(val, 8);
    val += warp.shfl_down(val, 4);
    val += warp.shfl_down(val, 2);
    val += warp.shfl_down(val, 1);

    block.sync();  // replaces __syncthreads()
}`}
            highlightLines={[5, 6, 7, 11, 17]}
          />
        </div>

        {/* Right: why better + benefits */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text="Why Better Than Raw __syncthreads()?"
            delay={5 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentOrange}
            style={{ marginBottom: 14 }}
          />

          <div style={{ display: "flex", gap: 16, marginBottom: 20 }}>
            {/* Old way */}
            <div style={{ flex: 1 }}>
              <FadeInText
                text="Old Way"
                delay={5.5 * fps}
                fontSize={16}
                fontWeight={600}
                color={THEME.colors.accentRed}
                style={{ marginBottom: 8 }}
              />
              <CodeBlock
                delay={5.8 * fps}
                title=""
                fontSize={13}
                showLineNumbers={false}
                code={`__syncthreads();
// syncs ALL threads
// must be hit by ALL
// can't sync subsets
// no composition`}
              />
            </div>

            {/* New way */}
            <div style={{ flex: 1 }}>
              <FadeInText
                text="Cooperative Groups"
                delay={6 * fps}
                fontSize={16}
                fontWeight={600}
                color={THEME.colors.nvidiaGreen}
                style={{ marginBottom: 8 }}
              />
              <CodeBlock
                delay={6.3 * fps}
                title=""
                fontSize={13}
                showLineNumbers={false}
                code={`group.sync();
// syncs only the group
// flexible granularity
// composable
// type-safe API`}
              />
            </div>
          </div>

          {/* Benefits */}
          <div style={{ opacity: whyBetterOpacity }}>
            <FadeInText
              text="Benefits"
              delay={8 * fps}
              fontSize={18}
              fontWeight={700}
              color={THEME.colors.nvidiaGreen}
              style={{ marginBottom: 10 }}
            />

            <BulletPoint
              text="Type safety -- compiler catches sync errors"
              index={0}
              delay={8.3 * fps}
              icon="\u2713"
              highlight
            />
            <BulletPoint
              text="Flexible granularity -- sync warps, tiles, or blocks"
              index={1}
              delay={8.3 * fps}
              icon="\u2713"
            />
            <BulletPoint
              text="Composable -- pass groups to device functions"
              index={2}
              delay={8.3 * fps}
              icon="\u2713"
            />
            <BulletPoint
              text="Grid-wide sync -- enables multi-block cooperation"
              index={3}
              delay={8.3 * fps}
              icon="\u2713"
            />
            <BulletPoint
              text="Works with divergent code -- coalesced_threads()"
              index={4}
              delay={8.3 * fps}
              icon="\u2713"
            />
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
