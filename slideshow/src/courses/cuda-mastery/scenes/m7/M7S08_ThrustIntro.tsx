import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

interface CodeLine {
  text: string;
  color: string;
  indent: number;
}

const CPU_CODE: CodeLine[] = [
  { text: "// CPU (STL)", color: THEME.colors.syntaxComment, indent: 0 },
  { text: "std::vector<int> v(1000000);", color: THEME.colors.textCode, indent: 0 },
  { text: "std::sort(", color: THEME.colors.syntaxFunction, indent: 0 },
  { text: "    v.begin(), v.end());", color: THEME.colors.textCode, indent: 0 },
  { text: "int sum = std::accumulate(", color: THEME.colors.syntaxFunction, indent: 0 },
  { text: "    v.begin(), v.end(), 0);", color: THEME.colors.textCode, indent: 0 },
];

const GPU_CODE: CodeLine[] = [
  { text: "// GPU (Thrust)", color: THEME.colors.syntaxComment, indent: 0 },
  { text: "thrust::device_vector<int> v(1000000);", color: THEME.colors.textCode, indent: 0 },
  { text: "thrust::sort(", color: THEME.colors.syntaxFunction, indent: 0 },
  { text: "    v.begin(), v.end());", color: THEME.colors.textCode, indent: 0 },
  { text: "int sum = thrust::reduce(", color: THEME.colors.syntaxFunction, indent: 0 },
  { text: "    v.begin(), v.end(), 0);", color: THEME.colors.textCode, indent: 0 },
];

// Lines that match between CPU and GPU (indices)
const MATCHING_PATTERNS = [
  { cpu: 1, gpu: 1, label: "vector" },
  { cpu: 2, gpu: 2, label: "sort" },
  { cpu: 4, gpu: 4, label: "reduce" },
];

export const M7S08_ThrustIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Code block animations
  const cpuOpacity = interpolate(
    frame - 1 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const gpuOpacity = interpolate(
    frame - 3 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Green glow highlight for matching syntax
  const glowPhase = interpolate(
    frame - 5 * fps,
    [0, 1 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Bottom tagline
  const taglineOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const renderCodeBlock = (
    lines: CodeLine[],
    title: string,
    titleColor: string,
    opacity: number,
    isGpu: boolean
  ) => (
    <div
      style={{
        opacity,
        width: 460,
      }}
    >
      <div
        style={{
          fontSize: 13,
          fontFamily: fontFamilyBody,
          fontWeight: 700,
          color: titleColor,
          marginBottom: 10,
          letterSpacing: "1px",
        }}
      >
        {title}
      </div>
      <div
        style={{
          padding: "16px 20px",
          backgroundColor: THEME.colors.bgCode,
          borderRadius: 10,
          border: `1px solid ${isGpu && glowPhase > 0 ? THEME.colors.nvidiaGreen + "60" : "rgba(255,255,255,0.08)"}`,
          boxShadow: isGpu && glowPhase > 0
            ? `0 0 ${20 * glowPhase}px rgba(118,185,0,${0.15 * glowPhase})`
            : "none",
          width: 440,
        }}
      >
        {lines.map((line, i) => {
          const lineDelay = (isGpu ? 3.5 : 1.5) + i * 0.15;
          const lineOpacity = interpolate(
            frame - lineDelay * fps,
            [0, 0.3 * fps],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );

          // Check if this line should be highlighted
          const isMatch = isGpu && MATCHING_PATTERNS.some((p) => p.gpu === i);
          const highlightGlow = isMatch ? glowPhase : 0;

          return (
            <div
              key={`line-${i}`}
              style={{
                fontSize: 14,
                fontFamily: fontFamilyCode,
                color: line.color,
                lineHeight: 1.8,
                opacity: lineOpacity,
                whiteSpace: "pre",
                backgroundColor:
                  highlightGlow > 0
                    ? `rgba(118,185,0,${0.08 * highlightGlow})`
                    : "transparent",
                borderRadius: 4,
                paddingLeft: 4,
                marginLeft: -4,
              }}
            >
              {line.text}
            </div>
          );
        })}
      </div>
    </div>
  );

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={7}
      leftWidth="52%"
      left={
        <div style={{ width: 600 }}>
          <SlideTitle
            title="Thrust -- STL for the GPU"
            subtitle="Familiar C++ interface, GPU performance"
          />

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 20,
              marginTop: 4,
              width: 580,
            }}
          >
            {renderCodeBlock(CPU_CODE, "CPU (STL)", THEME.colors.accentBlue, cpuOpacity, false)}
            {renderCodeBlock(GPU_CODE, "GPU (THRUST)", THEME.colors.nvidiaGreen, gpuOpacity, true)}
          </div>

          {/* Similarity callout */}
          <div
            style={{
              marginTop: 20,
              display: "flex",
              alignItems: "center",
              gap: 12,
              opacity: interpolate(
                frame - 6 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
              width: 500,
            }}
          >
            <div
              style={{
                width: 4,
                height: 32,
                backgroundColor: THEME.colors.nvidiaGreen,
                borderRadius: 2,
              }}
            />
            <span
              style={{
                fontSize: 16,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                color: THEME.colors.nvidiaGreen,
              }}
            >
              Near-identical syntax — just swap the namespace
            </span>
          </div>

          {/* Bottom tagline */}
          <div
            style={{
              marginTop: 16,
              padding: "12px 18px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: taglineOpacity,
              width: 500,
            }}
          >
            <span
              style={{
                fontSize: 17,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
              }}
            >
              10 lines of Thrust ={" "}
              <span style={{ color: THEME.colors.nvidiaGreen }}>
                100 lines of raw CUDA
              </span>
            </span>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 60, width: 420 }}>
          <BulletPoint
            index={0}
            delay={4 * fps}
            text="Automatic GPU memory management"
            icon="1"
          />
          <BulletPoint
            index={1}
            delay={4 * fps}
            text="Familiar STL-like interface"
            icon="2"
            highlight
          />
          <BulletPoint
            index={2}
            delay={4 * fps}
            text="sort, reduce, scan, transform, copy..."
            icon="3"
          />
          <BulletPoint
            index={3}
            delay={4 * fps}
            text="Interop with raw CUDA: thrust::device_ptr"
            icon="4"
            highlight
          />
        </div>
      }
    />
  );
};
