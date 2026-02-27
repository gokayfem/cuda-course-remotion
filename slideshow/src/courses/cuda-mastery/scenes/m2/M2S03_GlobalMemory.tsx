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
import { fontFamilyBody } from "../../../../styles/fonts";

export const M2S03_GlobalMemory: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const noteOpacity = interpolate(
    frame - 6 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={2}
      slideNumber={3}
      totalSlides={18}
      leftWidth="50%"
      left={
        <>
          <SlideTitle
            title="Global Memory — The Main GPU Memory"
            subtitle="VRAM / HBM: where all your data starts"
          />

          <BulletPoint
            index={0}
            delay={1 * fps}
            text="Large capacity: 24-80 GB"
            subtext="Enough for most workloads (VRAM on consumer, HBM on datacenter)"
            icon="1"
          />
          <BulletPoint
            index={1}
            delay={1 * fps}
            text="High latency: 400-800 cycles"
            subtext="Each access pays a heavy penalty if not cached"
            icon="2"
          />
          <BulletPoint
            index={2}
            delay={1 * fps}
            text="High bandwidth — if coalesced"
            subtext="1-3 TB/s peak, but only with consecutive access patterns"
            icon="3"
            highlight
          />
          <BulletPoint
            index={3}
            delay={1 * fps}
            text="Default memory space for all allocations"
            subtext="cudaMalloc() returns global memory pointers"
            icon="4"
          />
        </>
      }
      right={
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <CodeBlock
            delay={2.5 * fps}
            title="global_memory.cu"
            fontSize={16}
            code={`// Allocate global memory
float *d_data;
cudaMalloc(&d_data, N * sizeof(float));

// Simple kernel reading/writing global
__global__ void scale(float *data,
                      float factor, int N) {
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;
    if (idx < N) {
        // Read from global (400+ cycles)
        float val = data[idx];
        // Write to global (400+ cycles)
        data[idx] = val * factor;
    }
}

// Launch
scale<<<(N+255)/256, 256>>>(d_data, 2.0f, N);`}
            highlightLines={[11, 13]}
          />

          <div
            style={{
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              borderLeft: `4px solid ${THEME.colors.nvidiaGreen}`,
              opacity: noteOpacity,
            }}
          >
            <span
              style={{
                fontSize: 17,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
                lineHeight: 1.5,
              }}
            >
              Global memory is where most of your data lives. The art is{" "}
              <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>
                minimizing how often you access it
              </span>
              .
            </span>
          </div>
        </div>
      }
    />
  );
};
