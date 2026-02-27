import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../components/AnimatedText";
import { CodeBlock } from "../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../styles/fonts";

export const S05_CUDAQualifiers: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const qualifiers = [
    {
      keyword: "__global__",
      runs: "GPU",
      calledFrom: "CPU",
      desc: "Kernel entry point — this is what you launch",
      color: THEME.colors.nvidiaGreen,
    },
    {
      keyword: "__device__",
      runs: "GPU",
      calledFrom: "GPU only",
      desc: "Helper function called from other GPU code",
      color: THEME.colors.accentBlue,
    },
    {
      keyword: "__host__",
      runs: "CPU",
      calledFrom: "CPU",
      desc: "Normal CPU function (default, can be omitted)",
      color: THEME.colors.accentOrange,
    },
    {
      keyword: "__host__ __device__",
      runs: "Both",
      calledFrom: "Both",
      desc: "Compiles for CPU AND GPU — utility functions",
      color: THEME.colors.accentPurple,
    },
  ];

  return (
    <SlideLayout variant="gradient" slideNumber={5} totalSlides={18}>
      <SlideTitle
        title="CUDA Function Qualifiers"
        subtitle="Tell the compiler WHERE your code runs and WHO can call it"
      />

      <div style={{ display: "flex", gap: 48, flex: 1 }}>
        {/* Left: Table */}
        <div style={{ flex: 1 }}>
          {/* Header */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "200px 80px 100px 1fr",
              gap: 8,
              padding: "12px 16px",
              backgroundColor: "rgba(255,255,255,0.05)",
              borderRadius: 8,
              marginBottom: 8,
              opacity: interpolate(
                frame - 0.8 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            {["Qualifier", "Runs on", "Called from", "Purpose"].map((h) => (
              <span key={h} style={{ fontSize: 15, fontWeight: 700, color: THEME.colors.textMuted, fontFamily: fontFamilyBody }}>
                {h}
              </span>
            ))}
          </div>

          {/* Rows */}
          {qualifiers.map((q, i) => {
            const rowDelay = 1.2 * fps + i * 0.4 * fps;
            const rowOpacity = interpolate(
              frame - rowDelay,
              [0, 0.3 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );

            return (
              <div
                key={q.keyword}
                style={{
                  display: "grid",
                  gridTemplateColumns: "200px 80px 100px 1fr",
                  gap: 8,
                  padding: "14px 16px",
                  backgroundColor: `${q.color}08`,
                  borderLeft: `3px solid ${q.color}`,
                  borderRadius: 6,
                  marginBottom: 6,
                  opacity: rowOpacity,
                }}
              >
                <span style={{ fontSize: 17, fontWeight: 700, color: q.color, fontFamily: fontFamilyCode }}>
                  {q.keyword}
                </span>
                <span style={{ fontSize: 16, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody }}>
                  {q.runs}
                </span>
                <span style={{ fontSize: 16, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody }}>
                  {q.calledFrom}
                </span>
                <span style={{ fontSize: 16, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody }}>
                  {q.desc}
                </span>
              </div>
            );
          })}
        </div>

        {/* Right: Code example */}
        <div style={{ flex: 0.8 }}>
          <FadeInText
            text="In Practice"
            delay={3.5 * fps}
            fontSize={24}
            fontWeight={700}
            color={THEME.colors.nvidiaGreen}
            style={{ marginBottom: 16 }}
          />

          <CodeBlock
            delay={3.8 * fps}
            title="qualifiers.cu"
            fontSize={17}
            code={`// Kernel — launched from CPU
__global__ void add_kernel(
    float *a, float *b, float *c
) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// GPU helper — called from kernel
__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

// Works on BOTH CPU and GPU
__host__ __device__
float square(float x) {
    return x * x;
}`}
            highlightLines={[1, 2, 9, 10, 14, 15]}
          />
        </div>
      </div>
    </SlideLayout>
  );
};
