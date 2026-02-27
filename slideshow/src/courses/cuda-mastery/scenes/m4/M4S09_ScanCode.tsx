import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M4S09_ScanCode: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Blelloch scan step-by-step visualization on [3,1,7,0,4,1,6,3]
  // Up-sweep (reduce phase)
  // Step d=0: add pairs at distance 1 into odd indices
  //   idx 1: 3+1=4, idx 3: 7+0=7, idx 5: 4+1=5, idx 7: 6+3=9
  //   => [3,4,7,7,4,5,6,9]
  // Step d=1: add pairs at distance 2 into indices 3,7
  //   idx 3: 4+7=11, idx 7: 5+9=14
  //   => [3,4,7,11,4,5,6,14]
  // Step d=2: add pairs at distance 4 into index 7
  //   idx 7: 11+14=25
  //   => [3,4,7,11,4,5,6,25]
  // Set last to 0: [3,4,7,11,4,5,6,0]
  // Down-sweep
  // Step d=2: at index 7,3: swap and add
  //   temp=arr[3]=11, arr[3]=arr[7]=0, arr[7]=0+11=11
  //   => [3,4,7,0,4,5,6,11]
  // Step d=1: at indices 1,3,5,7:
  //   pair(1,3): temp=arr[1]=4, arr[1]=arr[3]=0, arr[3]=0+4=4
  //   pair(5,7): temp=arr[5]=5, arr[5]=arr[7]=11, arr[7]=11+5=16
  //   => [3,0,7,4,4,11,6,16]
  // Step d=0: at odd indices:
  //   pair(0,1): temp=arr[0]=3, arr[0]=arr[1]=0, arr[1]=0+3=3
  //   pair(2,3): temp=arr[2]=7, arr[2]=arr[3]=4, arr[3]=4+7=11
  //   pair(4,5): temp=arr[4]=4, arr[4]=arr[5]=11, arr[5]=11+4=15
  //   pair(6,7): temp=arr[6]=6, arr[6]=arr[7]=16, arr[7]=16+6=22
  //   => [0,3,4,11,11,15,16,22]

  const vizSteps = [
    { label: "Input", data: [3, 1, 7, 0, 4, 1, 6, 3], phase: "input", highlight: [] as number[] },
    { label: "Up d=1", data: [3, 4, 7, 7, 4, 5, 6, 9], phase: "up", highlight: [1, 3, 5, 7] },
    { label: "Up d=2", data: [3, 4, 7, 11, 4, 5, 6, 14], phase: "up", highlight: [3, 7] },
    { label: "Up d=4", data: [3, 4, 7, 11, 4, 5, 6, 25], phase: "up", highlight: [7] },
    { label: "Set 0", data: [3, 4, 7, 11, 4, 5, 6, 0], phase: "zero", highlight: [7] },
    { label: "Down d=4", data: [3, 4, 7, 0, 4, 5, 6, 11], phase: "down", highlight: [3, 7] },
    { label: "Down d=2", data: [3, 0, 7, 4, 4, 11, 6, 16], phase: "down", highlight: [1, 3, 5, 7] },
    { label: "Down d=1", data: [0, 3, 4, 11, 11, 15, 16, 22], phase: "down", highlight: [0, 1, 2, 3, 4, 5, 6, 7] },
  ];

  const N = 8;
  const CELL_W = 44;
  const CELL_H = 28;
  const CELL_GAP = 4;

  const keyPointsDelay = 9 * fps;
  const keyPointsOpacity = interpolate(
    frame - keyPointsDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={4}
      slideNumber={9}
      totalSlides={18}
      leftWidth="48%"
      left={
        <div>
          <SlideTitle
            title="Blelloch Scan Implementation"
            subtitle="Up-sweep (reduce) then down-sweep (distribute)"
          />

          <CodeBlock
            delay={0.5 * fps}
            title="blelloch_scan.cu"
            fontSize={12}
            code={`__global__ void blelloch_scan(
    float *data, int n
) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    temp[2*tid]   = data[2*tid];
    temp[2*tid+1] = data[2*tid+1];

    // Up-sweep (reduce)
    for (int d = 1; d < n; d *= 2) {
        __syncthreads();
        int ai = d * (2*tid+1) - 1;
        int bi = d * (2*tid+2) - 1;
        if (bi < n) temp[bi] += temp[ai];
    }

    // Set last element to 0
    if (tid == 0) temp[n-1] = 0;

    // Down-sweep (distribute)
    for (int d = n/2; d >= 1; d /= 2) {
        __syncthreads();
        int ai = d * (2*tid+1) - 1;
        int bi = d * (2*tid+2) - 1;
        if (bi < n) {
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    data[2*tid]   = temp[2*tid];
    data[2*tid+1] = temp[2*tid+1];
}`}
            highlightLines={[10, 11, 12, 13, 14, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29]}
          />
        </div>
      }
      right={
        <div style={{ paddingTop: 60 }}>
          <FadeInText
            text="Step-by-Step Visualization"
            delay={2 * fps}
            fontSize={18}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 12 }}
          />

          {/* Visualization steps */}
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {vizSteps.map((step, si) => {
              const stepDelay = 2.5 * fps + si * 0.7 * fps;
              const stepSpring = spring({
                frame: frame - stepDelay,
                fps,
                config: { damping: 200 },
              });
              const stepOpacity = interpolate(stepSpring, [0, 1], [0, 1]);

              const phaseColor =
                step.phase === "up" ? THEME.colors.nvidiaGreen
                : step.phase === "down" ? THEME.colors.accentPurple
                : step.phase === "zero" ? THEME.colors.accentOrange
                : THEME.colors.textMuted;

              return (
                <div
                  key={si}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    opacity: stepOpacity,
                  }}
                >
                  {/* Label */}
                  <div
                    style={{
                      width: 72,
                      fontSize: 11,
                      fontWeight: 700,
                      color: phaseColor,
                      fontFamily: fontFamilyCode,
                      textAlign: "right",
                      flexShrink: 0,
                    }}
                  >
                    {step.label}
                  </div>

                  {/* Phase indicator */}
                  <div
                    style={{
                      width: 4,
                      height: CELL_H,
                      borderRadius: 2,
                      backgroundColor: phaseColor,
                      flexShrink: 0,
                      opacity: 0.6,
                    }}
                  />

                  {/* Cells */}
                  <div style={{ display: "flex", gap: CELL_GAP }}>
                    {step.data.map((v, ci) => {
                      const isHighlighted = step.highlight.includes(ci);
                      return (
                        <div
                          key={ci}
                          style={{
                            width: CELL_W,
                            height: CELL_H,
                            borderRadius: 4,
                            backgroundColor: isHighlighted
                              ? `${phaseColor}20`
                              : "rgba(255,255,255,0.03)",
                            border: `1.5px solid ${isHighlighted ? `${phaseColor}80` : "rgba(255,255,255,0.08)"}`,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: 13,
                            fontWeight: isHighlighted ? 700 : 500,
                            color: isHighlighted ? phaseColor : THEME.colors.textSecondary,
                            fontFamily: fontFamilyCode,
                          }}
                        >
                          {v}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Legend */}
          <div
            style={{
              display: "flex",
              gap: 16,
              marginTop: 14,
              opacity: interpolate(
                frame - 8 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            {[
              { label: "Up-sweep", color: THEME.colors.nvidiaGreen },
              { label: "Set 0", color: THEME.colors.accentOrange },
              { label: "Down-sweep", color: THEME.colors.accentPurple },
            ].map((item) => (
              <div key={item.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: item.color }} />
                <span style={{ fontSize: 12, color: item.color, fontFamily: fontFamilyBody, fontWeight: 600 }}>
                  {item.label}
                </span>
              </div>
            ))}
          </div>

          {/* Key points */}
          <div
            style={{
              marginTop: 14,
              padding: "12px 16px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: keyPointsOpacity,
            }}
          >
            <div style={{ fontSize: 13, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.6 }}>
              <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>Key points:</span>
            </div>
            <div style={{ fontSize: 12, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, lineHeight: 1.6, marginTop: 4 }}>
              Add{" "}
              <span style={{ fontFamily: fontFamilyCode, color: THEME.colors.accentCyan }}>CONFLICT_FREE_OFFSET()</span>{" "}
              padding to avoid shared memory bank conflicts.
              For arrays larger than one block, scan per-block then scan the block sums.
            </div>
          </div>
        </div>
      }
    />
  );
};
