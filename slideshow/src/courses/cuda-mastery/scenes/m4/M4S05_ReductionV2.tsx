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

export const M4S05_ReductionV2: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Sequential addressing: stride starts at N/2, halves
  // stride=4: threads 0,1,2,3 active (contiguous!)
  // stride=2: threads 0,1 active
  // stride=1: thread 0 active
  const NUM_THREADS = 8;
  const steps = [
    { stride: 4, label: "stride=4 (blockDim/2)", active: [0, 1, 2, 3] },
    { stride: 2, label: "stride=2", active: [0, 1] },
    { stride: 1, label: "stride=1", active: [0] },
  ];

  const CELL_W = 52;
  const CELL_H = 32;
  const CELL_GAP = 4;
  const ROW_GAP = 14;
  const DIAGRAM_W = NUM_THREADS * (CELL_W + CELL_GAP);

  const goodDelay = 6 * fps;
  const goodSpring = spring({
    frame: frame - goodDelay,
    fps,
    config: { damping: 200 },
  });
  const goodOpacity = interpolate(goodSpring, [0, 1], [0, 1]);
  const goodScale = interpolate(goodSpring, [0, 1], [0.9, 1]);

  const perfDelay = 7.5 * fps;
  const perfOpacity = interpolate(
    frame - perfDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={4}
      slideNumber={5}
      totalSlides={18}
      leftWidth="48%"
      left={
        <div>
          <SlideTitle
            title="Reduction V2: Sequential"
            subtitle="Stride starts at blockDim.x/2, halves each step"
          />

          <CodeBlock
            delay={0.5 * fps}
            title="reduce_sequential.cu"
            fontSize={14}
            code={`__global__ void reduce_v2(
    float *data, float *out, int n
) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? data[i] : 0.0f;
    __syncthreads();

    // Sequential addressing
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}`}
            highlightLines={[11, 12, 13]}
          />

          {/* Performance comparison */}
          <div
            style={{
              marginTop: 16,
              padding: "12px 18px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: perfOpacity,
            }}
          >
            <span
              style={{
                fontSize: 16,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
              }}
            >
              ~2x faster
            </span>
            <span
              style={{
                fontSize: 14,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                marginLeft: 10,
              }}
            >
              just by changing the indexing pattern!
            </span>
          </div>
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
                            ? "rgba(118,185,0,0.20)"
                            : "rgba(255,255,255,0.03)",
                          border: `1.5px solid ${isActive ? THEME.colors.nvidiaGreen : "rgba(255,255,255,0.08)"}`,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 12,
                          fontWeight: 700,
                          color: isActive ? THEME.colors.nvidiaGreen : THEME.colors.textMuted,
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

          {/* GOOD badge */}
          <div
            style={{
              marginTop: 12,
              display: "flex",
              flexDirection: "column",
              gap: 10,
              opacity: goodOpacity,
              transform: `scale(${goodScale})`,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div
                style={{
                  padding: "6px 16px",
                  backgroundColor: "rgba(118,185,0,0.15)",
                  border: `2px solid ${THEME.colors.nvidiaGreen}`,
                  borderRadius: 6,
                  fontSize: 16,
                  fontWeight: 800,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                }}
              >
                GOOD: No Divergence
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
              First half of threads do work,
              second half are{" "}
              <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>cleanly idle</span>.
              Active threads are{" "}
              <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>contiguous</span>{" "}
              within each warp, so no lanes are wasted.
            </div>
          </div>
        </div>
      }
    />
  );
};
