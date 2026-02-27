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

export const M9S07_LayerNormCode: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const code = `__global__ void fused_layernorm(
    float* out, const float* in,
    const float* gamma, const float* beta,
    int D) {
  int row = blockIdx.x;
  const float* x = in + row * D;

  // Welford's online algorithm
  float mean = 0.0f, M2 = 0.0f;
  int count = 0;
  for (int i = threadIdx.x; i < D;
       i += blockDim.x) {
    count++;
    float delta = x[i] - mean;
    mean += delta / count;
    M2 += delta * (x[i] - mean);
  }
  // Warp-level reduction of mean, M2
  // + shared memory across warps
  // (reduction code omitted for clarity)

  float var = M2 / D;
  float inv_std = rsqrtf(var + 1e-5f);

  // Normalize + scale + shift
  for (int i = threadIdx.x; i < D;
       i += blockDim.x) {
    float norm = (x[i] - mean) * inv_std;
    out[row*D + i] = norm*gamma[i] + beta[i];
  }
}`;

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={9}
      leftWidth="55%"
      left={
        <div style={{ width: 860 }}>
          <SlideTitle
            title="Fused LayerNorm with Welford's"
            subtitle="Single kernel: statistics + normalization + affine transform"
          />
          <CodeBlock
            code={code}
            title="fused_layernorm.cu"
            delay={0.5 * fps}
            fontSize={15}
            highlightLines={[8, 9, 10, 11, 13, 14, 15, 16, 23, 28, 29]}
          />
        </div>
      }
      right={
        <div style={{ width: 580, marginTop: 80 }}>
          <BulletPoint
            text="Welford's: numerically stable online stats"
            index={0}
            delay={2 * fps}
            highlight
            subtext="Avoids catastrophic cancellation in variance computation"
          />
          <BulletPoint
            text="Warp + shared memory reduction"
            index={1}
            delay={2 * fps}
            subtext="Each warp reduces locally, then warps merge via shared memory"
          />
          <BulletPoint
            text="float4 vectorized loads for bandwidth"
            index={2}
            delay={2 * fps}
            subtext="Load 4 floats at once = 4x fewer memory transactions"
          />
          <BulletPoint
            text="RMSNorm variant: skip mean, just variance"
            index={3}
            delay={2 * fps}
            subtext="Used in LLaMA/Mistral: RMSNorm(x) = x / RMS(x) * gamma"
          />
          <BulletPoint
            text="One kernel replaces three"
            index={4}
            delay={2 * fps}
            subtext="Naive: mean kernel + var kernel + normalize kernel"
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
                fontFamily: fontFamilyBody,
              }}
            >
              Fused LayerNorm:{" "}
              <span
                style={{
                  color: THEME.colors.nvidiaGreen,
                  fontWeight: 700,
                }}
              >
                2-3x faster
              </span>{" "}
              than naive 3-kernel approach
            </span>
          </div>
        </div>
      }
    />
  );
};
