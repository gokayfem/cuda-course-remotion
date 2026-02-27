import React from "react";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText, BulletPoint } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

const StepBadge: React.FC<{
  step: number;
  label: string;
  color: string;
  delay: number;
}> = ({ step, label, color, delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const badgeSpring = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });
  const opacity = interpolate(badgeSpring, [0, 1], [0, 1]);

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        marginBottom: 8,
        opacity,
        transform: `translateX(${interpolate(badgeSpring, [0, 1], [-15, 0])}px)`,
      }}
    >
      <div
        style={{
          width: 28,
          height: 28,
          borderRadius: 14,
          backgroundColor: `${color}25`,
          border: `2px solid ${color}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 14,
          fontWeight: 800,
          color,
          fontFamily: fontFamilyBody,
          flexShrink: 0,
        }}
      >
        {step}
      </div>
      <span
        style={{
          fontSize: 16,
          color: THEME.colors.textPrimary,
          fontFamily: fontFamilyBody,
          fontWeight: 500,
        }}
      >
        {label}
      </span>
    </div>
  );
};

export const M3S14_PracticalPatterns: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <TwoColumnLayout
      moduleNumber={3}
      leftWidth="55%"
      left={
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
          <SlideTitle
            title="Block-Level Reduction"
            subtitle="The fastest pattern: warp shuffle + shared memory combined"
          />

          {/* Step indicators */}
          <div style={{ marginBottom: 12 }}>
            <StepBadge
              step={1}
              label="Each warp reduces its 32 values with __shfl_down_sync"
              color={THEME.colors.accentBlue}
              delay={0.5 * fps}
            />
            <StepBadge
              step={2}
              label="Lane 0 of each warp writes result to shared memory"
              color={THEME.colors.accentOrange}
              delay={1 * fps}
            />
            <StepBadge
              step={3}
              label="First warp loads shared memory and reduces again"
              color={THEME.colors.nvidiaGreen}
              delay={1.5 * fps}
            />
          </div>

          <CodeBlock
            delay={2 * fps}
            title="block_reduce.cu"
            fontSize={14}
            code={`__device__ float warpReduce(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, off);
    return val;
}

__device__ float blockReduce(float val) {
    __shared__ float warpSums[32]; // max 32 warps/block
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    // Step 1: warp-level reduction
    val = warpReduce(val);

    // Step 2: lane 0 writes to shared memory
    if (lane == 0)
        warpSums[wid] = val;
    __syncthreads();

    // Step 3: first warp reduces warp results
    val = (threadIdx.x < blockDim.x / 32)
        ? warpSums[lane] : 0.0f;
    if (wid == 0)
        val = warpReduce(val);

    return val; // thread 0 has block total
}`}
            highlightLines={[3, 13, 16, 17, 23, 24]}
          />
        </div>
      }
      right={
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <FadeInText
            text="Why This Is the Fastest Pattern"
            delay={5 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.nvidiaGreen}
            style={{ marginBottom: 4 }}
          />

          <BulletPoint
            text="Warp shuffles are single-cycle operations"
            index={0}
            delay={5.5 * fps}
            highlight
            subtext="No shared memory latency for intra-warp communication"
          />
          <BulletPoint
            text="Only ONE __syncthreads() call"
            index={1}
            delay={5.5 * fps}
            subtext="Minimizes barrier overhead vs. tree reduction in shared memory"
          />
          <BulletPoint
            text="Only 32 shared memory writes (one per warp)"
            index={2}
            delay={5.5 * fps}
            subtext="vs. N/2 reads + writes per step in naive shared memory reduction"
          />
          <BulletPoint
            text="No bank conflicts possible"
            index={3}
            delay={5.5 * fps}
            subtext="Each warp writes to a unique shared memory location"
          />
          <BulletPoint
            text="Works for any block size up to 1024 threads"
            index={4}
            delay={5.5 * fps}
            subtext="32 warps max = 1024 threads, all fits in shared memory"
          />

          {/* Performance comparison */}
          <div
            style={{
              marginTop: "auto",
              padding: "14px 18px",
              background: "linear-gradient(135deg, rgba(118,185,0,0.08), rgba(79,195,247,0.06))",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}30`,
              opacity: interpolate(
                frame - 9 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div style={{ fontSize: 14, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginBottom: 6 }}>
              Performance (1024 threads, reduce to scalar)
            </div>
            {[
              { label: "Naive shared memory", time: "~40 cycles", pct: "100%", color: THEME.colors.accentRed },
              { label: "Optimized shared mem", time: "~25 cycles", pct: "62%", color: THEME.colors.accentOrange },
              { label: "Warp shuffle + shared", time: "~10 cycles", pct: "25%", color: THEME.colors.nvidiaGreen },
            ].map((bar, i) => {
              const barSpring = spring({
                frame: frame - (9.5 * fps + i * 0.3 * fps),
                fps,
                config: { damping: 200 },
              });
              const barProgress = interpolate(barSpring, [0, 1], [0, 1]);

              return (
                <div key={i} style={{ marginBottom: 8, opacity: interpolate(barSpring, [0, 1], [0, 1]) }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                    <span style={{ fontSize: 13, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody }}>
                      {bar.label}
                    </span>
                    <span style={{ fontSize: 13, color: bar.color, fontFamily: fontFamilyCode, fontWeight: 700 }}>
                      {bar.time}
                    </span>
                  </div>
                  <div
                    style={{
                      height: 10,
                      backgroundColor: "rgba(255,255,255,0.06)",
                      borderRadius: 5,
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: `calc(${bar.pct} * ${barProgress})`,
                        backgroundColor: bar.color,
                        borderRadius: 5,
                        opacity: 0.8,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      }
    />
  );
};
