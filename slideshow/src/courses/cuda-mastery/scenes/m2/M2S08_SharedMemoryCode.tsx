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

export const M2S08_SharedMemoryCode: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const perfOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const haloSpring = spring({
    frame: frame - 5 * fps,
    fps,
    config: { damping: 200 },
  });
  const haloOpacity = interpolate(haloSpring, [0, 1], [0, 1]);

  const barAnim = interpolate(
    frame - 7.5 * fps,
    [0, 0.8 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={2} slideNumber={8} totalSlides={18}>
      <SlideTitle
        title="Shared Memory — 1D Stencil Example"
        subtitle="Classic pattern: load once, read many times from fast shared memory"
      />

      <div style={{ display: "flex", gap: 32, flex: 1 }}>
        {/* Left: Naive version */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 10,
            opacity: interpolate(frame - 0.8 * fps, [0, 0.3 * fps], [0, 1], {
              extrapolateLeft: "clamp", extrapolateRight: "clamp",
            }),
          }}>
            <div style={{
              width: 12, height: 12, borderRadius: 6,
              backgroundColor: THEME.colors.accentRed,
            }} />
            <span style={{
              fontSize: 18, fontWeight: 700,
              color: THEME.colors.accentRed,
              fontFamily: fontFamilyBody,
            }}>
              Naive (Global Only)
            </span>
          </div>

          <CodeBlock
            delay={1 * fps}
            title="stencil_naive.cu"
            fontSize={14}
            code={`__global__ void stencil_naive(
    float *out, const float *in, int N
) {
    int i = blockIdx.x * blockDim.x
          + threadIdx.x;
    if (i >= 3 && i < N - 3) {
        // 7 global memory reads!
        out[i] = c0 * in[i]
               + c1 * (in[i-1] + in[i+1])
               + c2 * (in[i-2] + in[i+2])
               + c3 * (in[i-3] + in[i+3]);
    }
}`}
            highlightLines={[7, 8, 9, 10, 11]}
          />

          <div style={{
            marginTop: 10,
            padding: "8px 14px",
            backgroundColor: "rgba(255,82,82,0.08)",
            borderRadius: 6,
            borderLeft: `3px solid ${THEME.colors.accentRed}`,
            opacity: interpolate(frame - 3 * fps, [0, 0.3 * fps], [0, 1], {
              extrapolateLeft: "clamp", extrapolateRight: "clamp",
            }),
          }}>
            <span style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody }}>
              Each thread reads <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>7 values from global memory</span>.
              Adjacent threads re-read overlapping elements.
            </span>
          </div>
        </div>

        {/* Right: Shared memory version */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 10,
            opacity: interpolate(frame - 3 * fps, [0, 0.3 * fps], [0, 1], {
              extrapolateLeft: "clamp", extrapolateRight: "clamp",
            }),
          }}>
            <div style={{
              width: 12, height: 12, borderRadius: 6,
              backgroundColor: THEME.colors.nvidiaGreen,
            }} />
            <span style={{
              fontSize: 18, fontWeight: 700,
              color: THEME.colors.nvidiaGreen,
              fontFamily: fontFamilyBody,
            }}>
              Shared Memory (Fast)
            </span>
          </div>

          <CodeBlock
            delay={3.5 * fps}
            title="stencil_shared.cu"
            fontSize={14}
            code={`__global__ void stencil_shared(
    float *out, const float *in, int N
) {
    __shared__ float s[BLOCK + 6];
    int i = blockIdx.x * blockDim.x
          + threadIdx.x;
    int si = threadIdx.x + 3; // halo offset

    // 1 global read per thread (coalesced)
    s[si] = in[i];

    // Load halo elements
    if (threadIdx.x < 3) {
        s[si - 3] = in[i - 3];
        s[si + BLOCK] = in[i + BLOCK];
    }
    __syncthreads(); // CRITICAL!

    // 7 reads from shared (~5 cycles each)
    out[i] = c0*s[si]
           + c1*(s[si-1] + s[si+1])
           + c2*(s[si-2] + s[si+2])
           + c3*(s[si-3] + s[si+3]);
}`}
            highlightLines={[4, 10, 17, 19, 20, 21, 22, 23]}
          />
        </div>
      </div>

      {/* Bottom: Halo explanation + Performance */}
      <div style={{ display: "flex", gap: 24, marginTop: 12 }}>
        {/* Halo diagram */}
        <div style={{
          flex: 1,
          padding: "10px 16px",
          backgroundColor: "rgba(79,195,247,0.08)",
          borderRadius: 8,
          borderLeft: `3px solid ${THEME.colors.accentBlue}`,
          opacity: haloOpacity,
        }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: THEME.colors.accentBlue, fontFamily: fontFamilyBody, marginBottom: 6 }}>
            Halo Elements
          </div>
          <div style={{ display: "flex", gap: 2, alignItems: "center" }}>
            {/* Left halo */}
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={`lh-${i}`} style={{
                width: 24, height: 20, borderRadius: 3,
                backgroundColor: `${THEME.colors.accentPurple}30`,
                border: `1px solid ${THEME.colors.accentPurple}60`,
                fontSize: 10, color: THEME.colors.accentPurple,
                fontFamily: fontFamilyCode,
                display: "flex", alignItems: "center", justifyContent: "center",
              }}>
                H
              </div>
            ))}
            {/* Block data */}
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={`d-${i}`} style={{
                width: 24, height: 20, borderRadius: 3,
                backgroundColor: `${THEME.colors.nvidiaGreen}30`,
                border: `1px solid ${THEME.colors.nvidiaGreen}60`,
                fontSize: 10, color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyCode,
                display: "flex", alignItems: "center", justifyContent: "center",
              }}>
                D
              </div>
            ))}
            {/* Right halo */}
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={`rh-${i}`} style={{
                width: 24, height: 20, borderRadius: 3,
                backgroundColor: `${THEME.colors.accentPurple}30`,
                border: `1px solid ${THEME.colors.accentPurple}60`,
                fontSize: 10, color: THEME.colors.accentPurple,
                fontFamily: fontFamilyCode,
                display: "flex", alignItems: "center", justifyContent: "center",
              }}>
                H
              </div>
            ))}
          </div>
          <span style={{ fontSize: 13, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginTop: 4, display: "block" }}>
            Extra elements at edges needed for stencil computation
          </span>
        </div>

        {/* Performance */}
        <div style={{
          flex: 1,
          padding: "10px 16px",
          backgroundColor: "rgba(118,185,0,0.08)",
          borderRadius: 8,
          borderLeft: `3px solid ${THEME.colors.nvidiaGreen}`,
          opacity: perfOpacity,
        }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyBody, marginBottom: 8 }}>
            Performance Comparison
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <span style={{ fontSize: 13, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, width: 70 }}>Naive</span>
            <div style={{ flex: 1, height: 14, backgroundColor: "rgba(255,255,255,0.05)", borderRadius: 3 }}>
              <div style={{ width: `${35 * barAnim}%`, height: "100%", backgroundColor: THEME.colors.accentRed, borderRadius: 3 }} />
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <span style={{ fontSize: 13, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, width: 70 }}>Shared</span>
            <div style={{ flex: 1, height: 14, backgroundColor: "rgba(255,255,255,0.05)", borderRadius: 3 }}>
              <div style={{ width: `${90 * barAnim}%`, height: "100%", backgroundColor: THEME.colors.nvidiaGreen, borderRadius: 3 }} />
            </div>
          </div>
          <span style={{ fontSize: 14, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyBody, fontWeight: 700 }}>
            Shared memory version: 2-3x faster
          </span>
        </div>
      </div>
    </SlideLayout>
  );
};
