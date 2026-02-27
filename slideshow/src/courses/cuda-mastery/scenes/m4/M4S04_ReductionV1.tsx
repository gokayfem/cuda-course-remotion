import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M4S04_ReductionV1: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Interleaved addressing: stride 1, 2, 4
  // At stride=1: threads 0,2,4,6 active (even threads)
  // At stride=2: threads 0,4 active
  // At stride=4: thread 0 active
  const NUM_THREADS = 8;
  const steps = [
    { stride: 1, label: "stride=1", active: [0, 2, 4, 6] },
    { stride: 2, label: "stride=2", active: [0, 4] },
    { stride: 4, label: "stride=4", active: [0] },
  ];

  const CELL_W = 52;
  const CELL_H = 32;
  const CELL_GAP = 4;
  const ROW_GAP = 14;
  const DIAGRAM_W = NUM_THREADS * (CELL_W + CELL_GAP);

  const warningDelay = 6 * fps;
  const warningSpring = spring({
    frame: frame - warningDelay,
    fps,
    config: { damping: 200 },
  });
  const warningOpacity = interpolate(warningSpring, [0, 1], [0, 1]);
  const warningScale = interpolate(warningSpring, [0, 1], [0.9, 1]);

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={4}
      slideNumber={4}
      totalSlides={18}
      leftWidth="48%"
      left={
        <div>
          <SlideTitle
            title="Reduction V1: Interleaved"
            subtitle="Stride starts at 1, doubles each step"
          />

          <CodeBlock
            delay={0.5 * fps}
            title="reduce_interleaved.cu"
            fontSize={14}
            code={`__global__ void reduce_v1(
    float *data, float *out, int n
) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? data[i] : 0.0f;
    __syncthreads();

    // Interleaved addressing
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}`}
            highlightLines={[11, 12, 13]}
          />
        </div>
      }
      right={
        <div style={{ paddingTop: 60 }}>
          {/* Thread activity diagram per step */}
          <div
            style={{
              fontSize: 18,
              color: THEME.colors.textPrimary,
              fontFamily: fontFamilyBody,
              fontWeight: 700,
              marginBottom: 16,
              opacity: interpolate(
                frame - 2 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            Thread Activity (8 threads)
          </div>

          {/* Thread header row */}
          <div
            style={{
              display: "flex",
              gap: CELL_GAP,
              marginBottom: 6,
              width: DIAGRAM_W,
              opacity: interpolate(
                frame - 2.2 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            {Array.from({ length: NUM_THREADS }).map((_, i) => (
              <div
                key={i}
                style={{
                  width: CELL_W,
                  fontSize: 12,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyCode,
                  textAlign: "center",
                }}
              >
                T{i}
              </div>
            ))}
          </div>

          {/* Steps */}
          {steps.map((step, si) => {
            const stepDelay = 2.5 * fps + si * 1.2 * fps;
            const stepSpring = spring({
              frame: frame - stepDelay,
              fps,
              config: { damping: 200 },
            });
            const stepOpacity = interpolate(stepSpring, [0, 1], [0, 1]);

            return (
              <div key={si} style={{ marginBottom: ROW_GAP, opacity: stepOpacity }}>
                <div
                  style={{
                    fontSize: 13,
                    color: THEME.colors.accentCyan,
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                    marginBottom: 4,
                  }}
                >
                  {step.label}
                </div>
                <div style={{ display: "flex", gap: CELL_GAP, width: DIAGRAM_W }}>
                  {Array.from({ length: NUM_THREADS }).map((_, tid) => {
                    const isActive = step.active.includes(tid);
                    return (
                      <div
                        key={tid}
                        style={{
                          width: CELL_W,
                          height: CELL_H,
                          borderRadius: 4,
                          backgroundColor: isActive
                            ? "rgba(255,171,64,0.20)"
                            : "rgba(255,255,255,0.03)",
                          border: `1.5px solid ${isActive ? THEME.colors.accentOrange : "rgba(255,255,255,0.08)"}`,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 12,
                          fontWeight: 700,
                          color: isActive ? THEME.colors.accentOrange : THEME.colors.textMuted,
                          fontFamily: fontFamilyCode,
                        }}
                      >
                        {isActive ? "WORK" : "idle"}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}

          {/* Warp divergence highlight */}
          <div
            style={{
              marginTop: 12,
              display: "flex",
              flexDirection: "column",
              gap: 10,
              opacity: warningOpacity,
              transform: `scale(${warningScale})`,
            }}
          >
            {/* BAD badge */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
              }}
            >
              <div
                style={{
                  padding: "6px 16px",
                  backgroundColor: "rgba(255,82,82,0.15)",
                  border: `2px solid ${THEME.colors.accentRed}`,
                  borderRadius: 6,
                  fontSize: 16,
                  fontWeight: 800,
                  color: THEME.colors.accentRed,
                  fontFamily: fontFamilyBody,
                }}
              >
                BAD: Warp Divergence
              </div>
            </div>

            <div
              style={{
                fontSize: 14,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                lineHeight: 1.6,
                maxWidth: 420,
              }}
            >
              Threads 0,2,4,6 are in the{" "}
              <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>same warp</span>{" "}
              as threads 1,3,5,7. The inactive threads still occupy execution slots,
              causing{" "}
              <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>50% wasted cycles</span>{" "}
              at every step.
            </div>
          </div>
        </div>
      }
    />
  );
};
