import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody } from "../../../../styles/fonts";

const multiGPUCode = `int deviceCount;
cudaGetDeviceCount(&deviceCount);

for (int i = 0; i < deviceCount; i++) {
    cudaSetDevice(i);
    cudaMalloc(&d_data[i], size);
    kernel<<<grid, block>>>(d_data[i]);
}
// Synchronize all
for (int i = 0; i < deviceCount; i++) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
}`;

export const M6S11_MultiGPUCode: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const tipOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={6}
      leftWidth="55%"
      left={
        <div style={{ width: 820 }}>
          <SlideTitle title="Multi-GPU Programming" />

          <CodeBlock
            delay={0.8 * fps}
            title="multi_gpu.cu"
            fontSize={16}
            code={multiGPUCode}
            highlightLines={[5, 6, 7, 11, 12]}
          />

          {/* Tip box */}
          <div
            style={{
              marginTop: 20,
              padding: "14px 20px",
              backgroundColor: "rgba(79,195,247,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.accentBlue}30`,
              opacity: tipOpacity,
              width: 750,
            }}
          >
            <div
              style={{
                fontSize: 14,
                fontWeight: 700,
                color: THEME.colors.accentBlue,
                fontFamily: fontFamilyBody,
                marginBottom: 4,
              }}
            >
              Tip: NCCL Library
            </div>
            <div
              style={{
                fontSize: 14,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                lineHeight: 1.5,
              }}
            >
              Use NVIDIA NCCL (Nickel) for optimized multi-GPU collective
              communication in ML -- AllReduce, Broadcast, AllGather with
              topology-aware routing
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ width: 460, marginTop: 80 }}>
          <BulletPoint
            text="Each device has its own default stream"
            index={0}
            delay={2 * fps}
          />
          <BulletPoint
            text="Allocations are device-local"
            index={1}
            delay={2 * fps}
          />
          <BulletPoint
            text="Enable P2P: cudaDeviceEnablePeerAccess()"
            index={2}
            delay={2 * fps}
            highlight
          />
          <BulletPoint
            text="NVLink: 600 GB/s vs PCIe: 64 GB/s"
            index={3}
            delay={2 * fps}
            highlight
          />
        </div>
      }
    />
  );
};
