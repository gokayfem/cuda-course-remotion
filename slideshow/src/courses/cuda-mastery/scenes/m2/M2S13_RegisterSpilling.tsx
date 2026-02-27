import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText, BulletPoint } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M2S13_RegisterSpilling: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const warningOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const fixOpacity = interpolate(
    frame - 5.5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Pulsing animation for warning
  const warningPulse = interpolate(
    frame % (fps * 2),
    [0, fps, fps * 2],
    [1, 0.7, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={2} slideNumber={13} totalSlides={18}>
      <SlideTitle
        title="Register Spilling & Local Memory"
        subtitle="When registers overflow, performance pays the price"
      />

      <div style={{ display: "flex", gap: 36, flex: 1 }}>
        {/* Left: Explanation and spilling code */}
        <div style={{ flex: 1 }}>
          <BulletPoint
            index={0}
            delay={0.8 * fps}
            text="When a thread needs more than 255 registers..."
            subtext="Variables 'spill' to local memory (stored in DRAM)"
            icon="!"
          />
          <BulletPoint
            index={1}
            delay={0.8 * fps}
            text="Causes of spilling"
            subtext="Too many local variables, large arrays, dynamic indexing"
            icon="?"
          />
          <BulletPoint
            index={2}
            delay={0.8 * fps}
            text="400x slower than register access"
            subtext="Spilled data goes through the full memory hierarchy"
            icon="X"
            highlight
          />

          <div style={{ marginTop: 14 }}>
            <CodeBlock
              delay={2.5 * fps}
              title="spilling_example.cu"
              fontSize={16}
              code={`// This WILL spill to local memory:
__global__ void heavy_kernel(float *data) {
    float arr[64]; // too large for registers
    int idx = threadIdx.x;

    // Dynamic indexing forces spill
    for (int i = 0; i < 64; i++) {
        arr[i] = data[idx + i] * 2.0f;
    }

    // Compiler can't keep arr[] in registers
    // because index 'i' is not compile-time constant
    data[idx] = arr[idx % 64];
}`}
              highlightLines={[3, 7, 13]}
            />
          </div>
        </div>

        {/* Right: Fixes and warning */}
        <div style={{ flex: 0.85, display: "flex", flexDirection: "column", gap: 16 }}>
          <FadeInText
            text="How to Detect & Fix"
            delay={4.5 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 4 }}
          />

          <div style={{ opacity: fixOpacity }}>
            <CodeBlock
              delay={5.5 * fps}
              title="compiler_hints.cu"
              fontSize={16}
              code={`// Compiler flag: limit registers per thread
// nvcc --maxrregcount=64 kernel.cu

// Per-kernel hint:
__global__ void
__launch_bounds__(256, 4) // maxThreads, minBlocks
my_kernel(float *data) {
    // Compiler targets 64 regs (256*64=16384)
    // to fit 4 blocks per SM
}

// Check spilling with:
// nvcc --ptxas-options=-v kernel.cu
// Look for "spill stores" and "spill loads"`}
              highlightLines={[2, 6, 13]}
            />
          </div>

          {/* Warning box */}
          <div
            style={{
              padding: "16px 20px",
              backgroundColor: `rgba(255,82,82,${0.06 * warningPulse + 0.04})`,
              borderRadius: 10,
              border: `2px solid ${THEME.colors.accentRed}60`,
              opacity: warningOpacity,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
              <span style={{ fontSize: 22, color: THEME.colors.accentRed, fontWeight: 900 }}>WARNING</span>
            </div>
            <span style={{ fontSize: 17, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.5 }}>
              <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>"Local memory" is a misnomer</span>{" "}
              — it lives in{" "}
              <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>GLOBAL memory (DRAM)</span>{" "}
              and is just as slow! The name "local" only means it is private to each thread.
            </span>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
