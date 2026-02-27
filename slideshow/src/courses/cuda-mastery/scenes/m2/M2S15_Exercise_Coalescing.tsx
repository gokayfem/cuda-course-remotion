import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M2S15_Exercise_Coalescing: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const showSolution = frame > 8 * fps;

  const hintOpacity = interpolate(
    frame - 4 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const resultOpacity = interpolate(
    frame - 11 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="accent" moduleNumber={2} slideNumber={15} totalSlides={18}>
      <SlideTitle
        title="Exercise: Fix the Coalescing!"
        subtitle="Classic matrix transpose — transform strided reads into coalesced access"
      />

      <div style={{ display: "flex", gap: 36, flex: 1 }}>
        {/* Left: Bad code */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text="The Problem: Strided Access"
            delay={0.5 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentRed}
            style={{ marginBottom: 12 }}
          />

          <CodeBlock
            delay={1 * fps}
            title="naive_transpose.cu (SLOW)"
            fontSize={16}
            code={`// Naive matrix transpose: column-major read
__global__ void transpose_naive(
    float *out, const float *in,
    int W, int H
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        // Read: in[x * H + y]  -> STRIDED! (H apart)
        // Write: out[y * W + x] -> coalesced
        out[y * W + x] = in[x * H + y];
    }
}
// Adjacent threads read addresses H apart
// = uncoalesced = many memory transactions`}
            highlightLines={[10, 12, 15, 16]}
          />

          {/* Hint */}
          <div
            style={{
              marginTop: 14,
              padding: "12px 18px",
              backgroundColor: "rgba(79,195,247,0.08)",
              borderRadius: 8,
              borderLeft: `4px solid ${THEME.colors.accentBlue}`,
              opacity: hintOpacity,
            }}
          >
            <span style={{ fontSize: 16, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.5 }}>
              <span style={{ color: THEME.colors.accentBlue, fontWeight: 700 }}>Hint:</span>{" "}
              Use shared memory as a staging area. Load a tile with coalesced reads, then write it transposed with coalesced writes.
            </span>
          </div>

          {/* Access pattern visualization */}
          <div
            style={{
              marginTop: 14,
              display: "flex",
              gap: 20,
              opacity: interpolate(frame - 3 * fps, [0, 0.5 * fps], [0, 1], {
                extrapolateLeft: "clamp", extrapolateRight: "clamp",
              }),
            }}
          >
            <div>
              <div style={{ fontSize: 13, color: THEME.colors.accentRed, fontFamily: fontFamilyBody, fontWeight: 600, marginBottom: 6 }}>
                Naive: Strided Read
              </div>
              <div style={{ display: "flex", gap: 3 }}>
                {[0, 1, 2, 3].map((t) => {
                  const s = spring({ frame: frame - (3.5 * fps + t * 0.15 * fps), fps, config: { damping: 200 } });
                  return (
                    <div key={t} style={{ display: "flex", flexDirection: "column", gap: 2, opacity: interpolate(s, [0, 1], [0, 1]) }}>
                      <div style={{ width: 28, height: 18, backgroundColor: `${THEME.colors.accentRed}20`, border: `1px solid ${THEME.colors.accentRed}`, borderRadius: 3, fontSize: 9, color: THEME.colors.accentRed, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: fontFamilyCode }}>
                        T{t}
                      </div>
                      <div style={{ fontSize: 9, color: THEME.colors.textMuted, textAlign: "center", fontFamily: fontFamilyCode }}>
                        +{t}H
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 13, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyBody, fontWeight: 600, marginBottom: 6 }}>
                Goal: Coalesced Read
              </div>
              <div style={{ display: "flex", gap: 3 }}>
                {[0, 1, 2, 3].map((t) => {
                  const s = spring({ frame: frame - (3.5 * fps + t * 0.15 * fps), fps, config: { damping: 200 } });
                  return (
                    <div key={t} style={{ display: "flex", flexDirection: "column", gap: 2, opacity: interpolate(s, [0, 1], [0, 1]) }}>
                      <div style={{ width: 28, height: 18, backgroundColor: `${THEME.colors.nvidiaGreen}20`, border: `1px solid ${THEME.colors.nvidiaGreen}`, borderRadius: 3, fontSize: 9, color: THEME.colors.nvidiaGreen, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: fontFamilyCode }}>
                        T{t}
                      </div>
                      <div style={{ fontSize: 9, color: THEME.colors.textMuted, textAlign: "center", fontFamily: fontFamilyCode }}>
                        +{t}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Right: Solution */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text={showSolution ? "Solution: Shared Memory Tile" : "Think about it first..."}
            delay={showSolution ? 8 * fps : 5 * fps}
            fontSize={22}
            fontWeight={700}
            color={showSolution ? THEME.colors.nvidiaGreen : THEME.colors.textMuted}
            style={{ marginBottom: 12 }}
          />

          {showSolution && (
            <CodeBlock
              delay={8.5 * fps}
              title="tiled_transpose.cu (FAST)"
              fontSize={15}
              code={`#define TILE 32
__global__ void transpose_tiled(
    float *out, const float *in,
    int W, int H
) {
    __shared__ float tile[TILE][TILE+1]; // +1 padding!

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Coalesced READ into shared memory
    if (x < W && y < H)
        tile[threadIdx.y][threadIdx.x] = in[y * W + x];

    __syncthreads();

    // Transposed indices for output
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    // Coalesced WRITE from shared memory
    if (x < H && y < W)
        out[y * H + x] = tile[threadIdx.x][threadIdx.y];
}`}
              highlightLines={[6, 13, 15, 23]}
            />
          )}

          {showSolution && (
            <div
              style={{
                marginTop: 12,
                padding: "12px 16px",
                backgroundColor: "rgba(118,185,0,0.08)",
                borderRadius: 8,
                opacity: resultOpacity,
              }}
            >
              <div style={{ fontSize: 15, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.6 }}>
                <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>Key insight:</span>{" "}
                Both the read (line 13) and write (line 23) are now coalesced.
                Shared memory acts as a "scratchpad" to rearrange the access pattern.
                Note the <span style={{ fontFamily: fontFamilyCode, color: THEME.colors.accentCyan }}>TILE+1</span> padding to avoid bank conflicts!
              </div>
            </div>
          )}
        </div>
      </div>
    </SlideLayout>
  );
};
