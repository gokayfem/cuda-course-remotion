import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, FadeInText, BulletPoint } from "../../../components/AnimatedText";
import { CodeBlock } from "../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../styles/fonts";

export const S12_GridStrideLoop: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Animate the grid-stride visualization
  const animPhase = interpolate(
    frame,
    [4 * fps, 8 * fps],
    [0, 3],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const totalElements = 16;
  const numThreads = 4;

  return (
    <SlideLayout variant="gradient" slideNumber={12} totalSlides={18}>
      <SlideTitle
        title="Grid-Stride Loop Pattern"
        subtitle="The professional way to handle ANY data size with ANY thread count"
      />

      <div style={{ display: "flex", gap: 40, flex: 1 }}>
        {/* Left: Code */}
        <div style={{ flex: 1 }}>
          <CodeBlock
            delay={0.5 * fps}
            title="Grid-stride loop"
            fontSize={19}
            code={`__global__ void grid_stride_add(
    const float *a, const float *b,
    float *c, int n
) {
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread handles MULTIPLE elements
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// 4 threads handle 16 elements!
grid_stride_add<<<1, 4>>>(a, b, c, 16);`}
            highlightLines={[7, 10]}
          />

          <div style={{ marginTop: 20 }}>
            <BulletPoint index={0} delay={4 * fps} text="Handles ANY array size with fixed thread count" highlight />
            <BulletPoint index={1} delay={4 * fps} text="More flexible than 1-element-per-thread" />
            <BulletPoint index={2} delay={4 * fps} text="Preferred pattern in production CUDA code" highlight />
          </div>
        </div>

        {/* Right: Visualization */}
        <div style={{ flex: 0.8 }}>
          <FadeInText
            text="Visualization: 4 threads, 16 elements"
            delay={3 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.nvidiaGreen}
            style={{ marginBottom: 16 }}
          />

          {/* Array elements */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 4 }}>
            {Array.from({ length: totalElements }).map((_, i) => {
              const threadOwner = i % numThreads;
              const pass = Math.floor(i / numThreads);
              const isActive = pass <= animPhase;

              const colors = [
                THEME.colors.accentBlue,
                THEME.colors.nvidiaGreen,
                THEME.colors.accentPurple,
                THEME.colors.accentOrange,
              ];

              const cellDelay = 4 * fps + i * 2;
              const cellSpring = spring({
                frame: frame - cellDelay,
                fps,
                config: { damping: 200 },
              });

              return (
                <div
                  key={i}
                  style={{
                    height: 52,
                    backgroundColor: isActive ? `${colors[threadOwner]}25` : "rgba(255,255,255,0.03)",
                    border: `2px solid ${isActive ? colors[threadOwner] : "rgba(255,255,255,0.1)"}`,
                    borderRadius: 6,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    opacity: interpolate(cellSpring, [0, 1], [0, 1]),
                  }}
                >
                  <span style={{ fontSize: 14, fontWeight: 700, color: isActive ? colors[threadOwner] : THEME.colors.textMuted, fontFamily: fontFamilyCode }}>
                    [{i}]
                  </span>
                  {isActive && (
                    <span style={{ fontSize: 10, color: colors[threadOwner], fontFamily: fontFamilyCode }}>
                      T{threadOwner}
                    </span>
                  )}
                </div>
              );
            })}
          </div>

          {/* Legend */}
          <div style={{ display: "flex", gap: 12, marginTop: 16, flexWrap: "wrap" }}>
            {["Thread 0", "Thread 1", "Thread 2", "Thread 3"].map((label, i) => {
              const colors = [
                THEME.colors.accentBlue,
                THEME.colors.nvidiaGreen,
                THEME.colors.accentPurple,
                THEME.colors.accentOrange,
              ];
              return (
                <div
                  key={label}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    opacity: interpolate(
                      frame - 5 * fps,
                      [0, 0.3 * fps],
                      [0, 1],
                      { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                    ),
                  }}
                >
                  <div style={{ width: 14, height: 14, backgroundColor: colors[i], borderRadius: 3 }} />
                  <span style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyCode }}>{label}</span>
                </div>
              );
            })}
          </div>

          {/* Stride explanation */}
          <div
            style={{
              marginTop: 24,
              padding: "16px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              opacity: interpolate(
                frame - 7 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div style={{ fontFamily: fontFamilyCode, fontSize: 16, color: THEME.colors.textPrimary, lineHeight: 2 }}>
              stride = blockDim.x * gridDim.x = <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>4</span><br />
              T0: i = 0, 4, 8, 12<br />
              T1: i = 1, 5, 9, 13<br />
              T2: i = 2, 6, 10, 14<br />
              T3: i = 3, 7, 11, 15
            </div>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
