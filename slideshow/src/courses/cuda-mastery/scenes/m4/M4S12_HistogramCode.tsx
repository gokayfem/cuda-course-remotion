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

export const M4S12_HistogramCode: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const insightOpacity = interpolate(
    frame - 8 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={4}
      slideNumber={12}
      totalSlides={18}
      leftWidth="55%"
      left={
        <>
          <SlideTitle
            title="Histogram Implementation"
            subtitle="Shared memory privatization -- the workhorse approach"
          />

          <CodeBlock
            delay={0.5 * fps}
            title="histogram_shared.cu"
            fontSize={14}
            code={`__global__ void histogram_shared(
    const int *data, int *hist, int N, int numBins
) {
    // Phase 1: Zero shared memory histogram
    extern __shared__ int s_hist[];
    int tid = threadIdx.x;
    for (int b = tid; b < numBins; b += blockDim.x)
        s_hist[b] = 0;
    __syncthreads();

    // Phase 2: Accumulate into shared memory
    int i = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    for (int idx = i; idx < N; idx += stride)
        atomicAdd(&s_hist[data[idx]], 1);
    __syncthreads();

    // Phase 3: Merge shared -> global
    for (int b = tid; b < numBins; b += blockDim.x)
        atomicAdd(&hist[b], s_hist[b]);
}`}
            highlightLines={[8, 9, 15, 16, 20]}
          />
        </>
      }
      right={
        <>
          <div style={{ marginTop: 100 }}>
            <BulletPoint
              index={0}
              delay={2 * fps}
              text="Phase 1: Initialize"
              subtext="Zero out shared memory bins. Each thread handles multiple bins if numBins > blockDim."
            />
            <BulletPoint
              index={1}
              delay={2 * fps}
              text="Phase 2: Local accumulation"
              subtext="atomicAdd to shared memory -- ~100x lower latency than global. Uses grid-stride loop for large arrays."
              highlight
            />
            <BulletPoint
              index={2}
              delay={2 * fps}
              text="Phase 3: Merge to global"
              subtext="Only numBins atomic adds per block to global memory. For 256 bins and 1000 blocks: 256K vs millions of global atomics."
            />
            <BulletPoint
              index={3}
              delay={2 * fps}
              text="__syncthreads is critical"
              subtext="Between each phase -- ensures all shared memory writes are visible before the next phase reads."
            />
          </div>

          {/* Application insight box */}
          <div
            style={{
              marginTop: 20,
              padding: "14px 18px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.nvidiaGreen}30`,
              opacity: insightOpacity,
            }}
          >
            <div
              style={{
                fontSize: 15,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
                marginBottom: 4,
              }}
            >
              ML Application
            </div>
            <div
              style={{
                fontSize: 14,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                lineHeight: 1.5,
              }}
            >
              Quantization calibration computes value distributions using histograms.
              Shared memory privatization makes this practical for large tensors.
            </div>
          </div>
        </>
      }
    />
  );
};
