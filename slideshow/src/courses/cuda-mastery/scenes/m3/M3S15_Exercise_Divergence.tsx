import React from "react";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

export const M3S15_Exercise_Divergence: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const showSolution = frame > 8 * fps;

  const hintOpacity = interpolate(
    frame - 4 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const insightOpacity = interpolate(
    frame - 11.5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="accent" moduleNumber={3}>
      <SlideTitle
        title="Exercise: Fix the Divergence"
        subtitle="Restructure a divergent kernel to minimize warp divergence"
      />

      <div style={{ display: "flex", gap: 36, flex: 1 }}>
        {/* Left: Problem */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text="The Problem: Heavy Divergence"
            delay={0.5 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentRed}
            style={{ marginBottom: 10 }}
          />

          <CodeBlock
            delay={1 * fps}
            title="divergent_kernel.cu (SLOW)"
            fontSize={15}
            code={`__global__ void process(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float val = data[tid];

    // Heavy divergence within each warp!
    if (val < 0.0f) {
        val = -val;              // path A
        val = sqrtf(val) * 2.0f;
    } else if (val < 1.0f) {
        val = val * val;         // path B
    } else if (val < 10.0f) {
        val = logf(val);         // path C
    } else {
        val = 1.0f / val;       // path D
    }

    data[tid] = val;
}`}
            highlightLines={[8, 11, 13, 15]}
          />

          {/* Divergence visualization */}
          <div
            style={{
              marginTop: 12,
              opacity: interpolate(
                frame - 3 * fps,
                [0, 0.4 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div style={{ fontSize: 14, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginBottom: 6 }}>
              Warp execution (random data):
            </div>
            <div style={{ display: "flex", gap: 2 }}>
              {["A", "B", "C", "D", "B", "A", "D", "C", "B", "A", "C", "B", "D", "A", "B", "C"].map((path, i) => {
                const pathColors: Record<string, string> = {
                  A: THEME.colors.accentRed,
                  B: THEME.colors.accentBlue,
                  C: THEME.colors.accentOrange,
                  D: THEME.colors.accentPurple,
                };
                const cellSpring = spring({
                  frame: frame - (3.2 * fps + i * 1.5),
                  fps,
                  config: { damping: 200 },
                });
                return (
                  <div
                    key={i}
                    style={{
                      width: 32,
                      height: 24,
                      backgroundColor: `${pathColors[path]}25`,
                      border: `1px solid ${pathColors[path]}`,
                      borderRadius: 3,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 10,
                      color: pathColors[path],
                      fontFamily: fontFamilyCode,
                      fontWeight: 700,
                      opacity: interpolate(cellSpring, [0, 1], [0, 1]),
                    }}
                  >
                    {path}
                  </div>
                );
              })}
            </div>
            <div style={{ fontSize: 13, color: THEME.colors.accentRed, fontFamily: fontFamilyBody, marginTop: 4, fontWeight: 600 }}>
              4 passes needed -- threads idle 75% of the time!
            </div>
          </div>

          {/* Hint */}
          <div
            style={{
              marginTop: 12,
              padding: "10px 16px",
              backgroundColor: "rgba(79,195,247,0.08)",
              borderRadius: 8,
              borderLeft: `4px solid ${THEME.colors.accentBlue}`,
              opacity: hintOpacity,
            }}
          >
            <span style={{ fontSize: 15, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.5 }}>
              <span style={{ color: THEME.colors.accentBlue, fontWeight: 700 }}>Hint:</span>{" "}
              What if you sort data by category first, so adjacent threads take the same path?
              Or use separate kernels per category?
            </span>
          </div>
        </div>

        {/* Right: Solution */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text={showSolution ? "Solution: Sort-Based Approach" : "Think about it first..."}
            delay={showSolution ? 8 * fps : 5 * fps}
            fontSize={20}
            fontWeight={700}
            color={showSolution ? THEME.colors.nvidiaGreen : THEME.colors.textMuted}
            style={{ marginBottom: 10 }}
          />

          {showSolution && (
            <>
              <CodeBlock
                delay={8.5 * fps}
                title="sorted_divergence.cu (FAST)"
                fontSize={14}
                code={`// Strategy: pre-sort by category,
// then process with minimal divergence

// Step 1: Classify each element
__global__ void classify(float* data, int* cat, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float v = data[tid];
    if (v < 0.0f)       cat[tid] = 0;
    else if (v < 1.0f)  cat[tid] = 1;
    else if (v < 10.0f) cat[tid] = 2;
    else                 cat[tid] = 3;
}

// Step 2: Sort by category (thrust::sort_by_key)

// Step 3: Process -- adjacent threads same path!
__global__ void process_sorted(
    float* data, int* cat, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float val = data[tid];
    int c = cat[tid];

    // Minimal divergence: entire warps take same path
    if (c == 0)      val = sqrtf(-val) * 2.0f;
    else if (c == 1) val = val * val;
    else if (c == 2) val = logf(val);
    else             val = 1.0f / val;

    data[tid] = val;
}`}
                highlightLines={[9, 10, 11, 12, 27, 28, 29, 30]}
              />

              {/* Sorted visualization */}
              <div
                style={{
                  marginTop: 12,
                  opacity: interpolate(
                    frame - 10.5 * fps,
                    [0, 0.4 * fps],
                    [0, 1],
                    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                  ),
                }}
              >
                <div style={{ fontSize: 14, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginBottom: 6 }}>
                  After sorting -- uniform warps:
                </div>
                <div style={{ display: "flex", gap: 2 }}>
                  {["A", "A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "D", "D", "D"].map((path, i) => {
                    const pathColors: Record<string, string> = {
                      A: THEME.colors.accentRed,
                      B: THEME.colors.accentBlue,
                      C: THEME.colors.accentOrange,
                      D: THEME.colors.accentPurple,
                    };
                    const cellSpring = spring({
                      frame: frame - (10.8 * fps + i * 1.5),
                      fps,
                      config: { damping: 200 },
                    });
                    return (
                      <div
                        key={i}
                        style={{
                          width: 32,
                          height: 24,
                          backgroundColor: `${pathColors[path]}25`,
                          border: `1px solid ${pathColors[path]}`,
                          borderRadius: 3,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 10,
                          color: pathColors[path],
                          fontFamily: fontFamilyCode,
                          fontWeight: 700,
                          opacity: interpolate(cellSpring, [0, 1], [0, 1]),
                        }}
                      >
                        {path}
                      </div>
                    );
                  })}
                </div>
                <div style={{ fontSize: 13, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyBody, marginTop: 4, fontWeight: 600 }}>
                  Most warps take 1 pass -- near 100% utilization!
                </div>
              </div>

              <div
                style={{
                  marginTop: 10,
                  padding: "10px 14px",
                  backgroundColor: "rgba(118,185,0,0.08)",
                  borderRadius: 8,
                  opacity: insightOpacity,
                }}
              >
                <span style={{ fontSize: 14, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.5 }}>
                  <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>Trade-off:</span>{" "}
                  Sort cost vs. divergence savings. Worth it when per-branch work is expensive
                  and data is large. Alternative: launch separate kernels per category.
                </span>
              </div>
            </>
          )}
        </div>
      </div>
    </SlideLayout>
  );
};
