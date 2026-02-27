import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M2S05_CoalescingCode: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const perfSpring = spring({
    frame: frame - 5 * fps,
    fps,
    config: { damping: 200 },
  });
  const perfOpacity = interpolate(perfSpring, [0, 1], [0, 1]);

  const noteOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const barWidth = interpolate(
    frame - 5.5 * fps,
    [0, 0.8 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={2} slideNumber={5} totalSlides={18}>
      <SlideTitle
        title="Coalescing in Practice — Good vs Bad Patterns"
        subtitle="Side-by-side comparison of memory access patterns"
      />

      <div style={{ display: "flex", gap: 32, flex: 1 }}>
        {/* Left: Coalesced */}
        <div style={{ flex: 1 }}>
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 12,
            opacity: interpolate(frame - 1 * fps, [0, 0.3 * fps], [0, 1], {
              extrapolateLeft: "clamp", extrapolateRight: "clamp",
            }),
          }}>
            <div style={{
              width: 12, height: 12, borderRadius: 6,
              backgroundColor: THEME.colors.nvidiaGreen,
            }} />
            <span style={{
              fontSize: 20, fontWeight: 700,
              color: THEME.colors.nvidiaGreen,
              fontFamily: fontFamilyBody,
            }}>
              COALESCED (Good)
            </span>
          </div>

          <CodeBlock
            delay={1.5 * fps}
            title="coalesced.cu"
            fontSize={15}
            code={`__global__ void copy_coalesced(
    float *out, const float *in, int N
) {
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;
    if (idx < N) {
        // Thread 0 -> addr 0
        // Thread 1 -> addr 1
        // Thread 2 -> addr 2  ...
        out[idx] = in[idx];
    }
}

// All 32 threads in a warp access
// consecutive addresses -> 1 transaction`}
            highlightLines={[10]}
          />
        </div>

        {/* Right: Strided */}
        <div style={{ flex: 1 }}>
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 12,
            opacity: interpolate(frame - 1 * fps, [0, 0.3 * fps], [0, 1], {
              extrapolateLeft: "clamp", extrapolateRight: "clamp",
            }),
          }}>
            <div style={{
              width: 12, height: 12, borderRadius: 6,
              backgroundColor: THEME.colors.accentRed,
            }} />
            <span style={{
              fontSize: 20, fontWeight: 700,
              color: THEME.colors.accentRed,
              fontFamily: fontFamilyBody,
            }}>
              STRIDED (Bad)
            </span>
          </div>

          <CodeBlock
            delay={2.5 * fps}
            title="strided.cu"
            fontSize={15}
            code={`__global__ void copy_strided(
    float *out, const float *in,
    int N, int stride
) {
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;
    if (idx * stride < N) {
        // Thread 0 -> addr 0
        // Thread 1 -> addr stride
        // Thread 2 -> addr 2*stride ...
        out[idx] = in[idx * stride];
    }
}

// Threads hit different cache lines
// -> up to 32 separate transactions!`}
            highlightLines={[11]}
          />
        </div>
      </div>

      {/* Performance comparison */}
      <div style={{
        display: "flex",
        gap: 32,
        marginTop: 16,
        opacity: perfOpacity,
      }}>
        {/* Coalesced bar */}
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
            <span style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, width: 100 }}>
              Coalesced
            </span>
            <div style={{ flex: 1, height: 18, backgroundColor: "rgba(255,255,255,0.05)", borderRadius: 4 }}>
              <div style={{
                width: `${100 * barWidth}%`,
                height: "100%",
                backgroundColor: THEME.colors.nvidiaGreen,
                borderRadius: 4,
              }} />
            </div>
            <span style={{ fontSize: 14, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyCode, fontWeight: 700, width: 80 }}>
              ~900 GB/s
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, width: 100 }}>
              Strided (32)
            </span>
            <div style={{ flex: 1, height: 18, backgroundColor: "rgba(255,255,255,0.05)", borderRadius: 4 }}>
              <div style={{
                width: `${3.3 * barWidth}%`,
                height: "100%",
                backgroundColor: THEME.colors.accentRed,
                borderRadius: 4,
              }} />
            </div>
            <span style={{ fontSize: 14, color: THEME.colors.accentRed, fontFamily: fontFamilyCode, fontWeight: 700, width: 80 }}>
              ~30 GB/s
            </span>
          </div>
        </div>

        {/* SoA vs AoS note */}
        <div style={{
          flex: 1,
          padding: "12px 20px",
          backgroundColor: "rgba(79,195,247,0.08)",
          borderRadius: 10,
          borderLeft: `4px solid ${THEME.colors.accentBlue}`,
          opacity: noteOpacity,
        }}>
          <span style={{
            fontSize: 16,
            color: THEME.colors.textPrimary,
            fontFamily: fontFamilyBody,
            lineHeight: 1.5,
          }}>
            Real-world pattern:{" "}
            <span style={{ color: THEME.colors.accentBlue, fontWeight: 700 }}>
              Struct of Arrays (SoA)
            </span>{" "}
            gives coalesced access.{" "}
            <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>
              Array of Structs (AoS)
            </span>{" "}
            causes strided access. Next slide explores this.
          </span>
        </div>
      </div>
    </SlideLayout>
  );
};
