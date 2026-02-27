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
import { CodeBlock } from "../../../../components/CodeBlock";

export const M9S05_SoftmaxCode: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const code = `__device__ float warp_softmax(
    float val, int lane_id) {
  // Step 1: Find max across warp
  float max_val = val;
  for (int offset = 16; offset > 0;
       offset >>= 1) {
    max_val = fmaxf(max_val,
      __shfl_down_sync(0xffffffff,
                       max_val, offset));
  }
  // Broadcast max to all lanes
  max_val = __shfl_sync(
    0xffffffff, max_val, 0);

  // Step 2: Compute exp and sum
  float exp_val = expf(val - max_val);
  float sum = exp_val;
  for (int offset = 16; offset > 0;
       offset >>= 1) {
    sum += __shfl_down_sync(
      0xffffffff, sum, offset);
  }
  sum = __shfl_sync(
    0xffffffff, sum, 0);

  // Step 3: Normalize
  return exp_val / sum;
}`;

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={9}
      leftWidth="55%"
      left={
        <div style={{ width: 860 }}>
          <SlideTitle
            title="Softmax Implementation"
            subtitle="Warp-level softmax using shuffle intrinsics"
          />
          <CodeBlock
            code={code}
            title="warp_softmax.cu"
            delay={0.5 * fps}
            fontSize={15}
            highlightLines={[3, 5, 6, 7, 8, 9, 14, 15, 18, 22, 23]}
          />
        </div>
      }
      right={
        <div style={{ width: 580, marginTop: 80 }}>
          <BulletPoint
            text="__shfl_down_sync: exchange data within warp"
            index={0}
            delay={2 * fps}
            highlight
            subtext="No shared memory needed — registers only, near-zero latency"
          />
          <BulletPoint
            text="Works for seq_len up to 32 (one warp)"
            index={1}
            delay={2 * fps}
            subtext="Each lane holds one element, reduction covers all 32 values"
          />
          <BulletPoint
            text="Block-level for longer sequences"
            index={2}
            delay={2 * fps}
            subtext="Use shared memory + multiple warps for seq_len > 32"
          />
          <BulletPoint
            text="Two reductions: max then sum"
            index={3}
            delay={2 * fps}
            subtext="Each takes log2(32) = 5 steps via butterfly reduction"
          />
          <BulletPoint
            text="Broadcast result to all lanes"
            index={4}
            delay={2 * fps}
            subtext="__shfl_sync(mask, val, 0) shares lane 0's value"
          />

          {/* Bottom callout */}
          <div
            style={{
              marginTop: 36,
              padding: "14px 24px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              borderLeft: `4px solid ${THEME.colors.nvidiaGreen}`,
              opacity: interpolate(
                frame - 8 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <span
              style={{
                fontSize: 20,
                color: THEME.colors.textPrimary,
                fontFamily: "Inter, sans-serif",
              }}
            >
              Warp softmax:{" "}
              <span
                style={{
                  color: THEME.colors.nvidiaGreen,
                  fontWeight: 700,
                }}
              >
                3x faster
              </span>{" "}
              than cuDNN for short sequences
            </span>
          </div>
        </div>
      }
    />
  );
};
