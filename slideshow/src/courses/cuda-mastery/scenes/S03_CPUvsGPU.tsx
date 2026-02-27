import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../styles/fonts";

const CoreBox: React.FC<{
  x: number;
  y: number;
  size: number;
  color: string;
  label: string;
  delay: number;
}> = ({ x, y, size, color, label, delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const s = spring({ frame: frame - delay, fps, config: { damping: 200 } });
  const opacity = interpolate(s, [0, 1], [0, 1]);
  const scale = interpolate(s, [0, 1], [0.5, 1]);

  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        width: size,
        height: size,
        backgroundColor: `${color}30`,
        border: `2px solid ${color}`,
        borderRadius: 6,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        opacity,
        transform: `scale(${scale})`,
        fontSize: size > 30 ? 12 : 9,
        color,
        fontFamily: fontFamilyCode,
        fontWeight: 700,
      }}
    >
      {label}
    </div>
  );
};

export const S03_CPUvsGPU: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const analogyOpacity = interpolate(
    frame,
    [6 * fps, 7 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" slideNumber={3} totalSlides={18}>
      <SlideTitle
        title="CPU vs GPU Architecture"
        subtitle="Few powerful cores vs. thousands of simple cores"
      />

      <div style={{ display: "flex", gap: 60, flex: 1 }}>
        {/* CPU Side */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text="CPU — Latency Optimized"
            delay={0.5 * fps}
            fontSize={26}
            fontWeight={700}
            color={THEME.colors.accentOrange}
            style={{ marginBottom: 20 }}
          />

          {/* CPU diagram */}
          <div
            style={{
              position: "relative",
              height: 340,
              backgroundColor: "rgba(255,171,64,0.05)",
              borderRadius: 12,
              border: `1px solid rgba(255,171,64,0.2)`,
              padding: 20,
            }}
          >
            {/* Big cache */}
            <FadeInText
              text="Large L3 Cache (32MB)"
              delay={1 * fps}
              fontSize={14}
              color={THEME.colors.textMuted}
              style={{ position: "absolute", top: 10, left: 20 }}
            />
            <div
              style={{
                position: "absolute",
                top: 30,
                left: 20,
                right: 20,
                height: 50,
                backgroundColor: "rgba(255,171,64,0.1)",
                border: `1px solid rgba(255,171,64,0.3)`,
                borderRadius: 8,
                opacity: interpolate(
                  frame - 1.2 * fps,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 13,
                color: THEME.colors.accentOrange,
                fontFamily: fontFamilyCode,
              }}
            >
              Branch Predictor | Out-of-Order Execution | Prefetcher
            </div>

            {/* 8 big cores */}
            {Array.from({ length: 8 }).map((_, i) => (
              <CoreBox
                key={i}
                x={20 + (i % 4) * 95}
                y={110 + Math.floor(i / 4) * 100}
                size={80}
                color={THEME.colors.accentOrange}
                label={`Core ${i}`}
                delay={1.5 * fps + i * 3}
              />
            ))}

            <FadeInText
              text="8 cores, each 5+ GHz, deep pipeline"
              delay={3 * fps}
              fontSize={14}
              color={THEME.colors.textSecondary}
              style={{ position: "absolute", bottom: 10, left: 20 }}
            />
          </div>

          <FadeInText
            text="Great at: complex logic, branching, sequential tasks"
            delay={3.5 * fps}
            fontSize={18}
            color={THEME.colors.textSecondary}
            style={{ marginTop: 12 }}
          />
        </div>

        {/* GPU Side */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text="GPU — Throughput Optimized"
            delay={0.5 * fps}
            fontSize={26}
            fontWeight={700}
            color={THEME.colors.nvidiaGreen}
            style={{ marginBottom: 20 }}
          />

          {/* GPU diagram */}
          <div
            style={{
              position: "relative",
              height: 340,
              backgroundColor: "rgba(118,185,0,0.05)",
              borderRadius: 12,
              border: `1px solid rgba(118,185,0,0.2)`,
              padding: 20,
            }}
          >
            <FadeInText
              text="Small Cache, Simple Control"
              delay={1 * fps}
              fontSize={14}
              color={THEME.colors.textMuted}
              style={{ position: "absolute", top: 10, left: 20 }}
            />

            {/* Hundreds of tiny cores */}
            <div
              style={{
                position: "absolute",
                top: 40,
                left: 15,
                display: "grid",
                gridTemplateColumns: "repeat(20, 18px)",
                gap: 2,
              }}
            >
              {Array.from({ length: 200 }).map((_, i) => {
                const coreDelay = 1.5 * fps + i * 0.3;
                const s = spring({
                  frame: frame - coreDelay,
                  fps,
                  config: { damping: 200 },
                });
                return (
                  <div
                    key={i}
                    style={{
                      width: 18,
                      height: 14,
                      backgroundColor: `${THEME.colors.nvidiaGreen}40`,
                      border: `1px solid ${THEME.colors.nvidiaGreen}60`,
                      borderRadius: 2,
                      opacity: interpolate(s, [0, 1], [0, 1]),
                    }}
                  />
                );
              })}
            </div>

            <FadeInText
              text="16,896 CUDA cores (H100), simple ALUs"
              delay={4 * fps}
              fontSize={14}
              color={THEME.colors.textSecondary}
              style={{ position: "absolute", bottom: 10, left: 20 }}
            />
          </div>

          <FadeInText
            text="Great at: same operation on massive data (SIMT)"
            delay={4.5 * fps}
            fontSize={18}
            color={THEME.colors.textSecondary}
            style={{ marginTop: 12 }}
          />
        </div>
      </div>

      {/* Analogy */}
      <div
        style={{
          marginTop: 20,
          padding: "16px 32px",
          backgroundColor: "rgba(118,185,0,0.08)",
          borderRadius: 12,
          border: `1px solid rgba(118,185,0,0.2)`,
          opacity: analogyOpacity,
          textAlign: "center",
        }}
      >
        <span
          style={{
            fontSize: 24,
            color: THEME.colors.textPrimary,
            fontFamily: fontFamilyBody,
          }}
        >
          CPU = <span style={{ color: THEME.colors.accentOrange, fontWeight: 700 }}>A few brilliant professors</span> solving hard problems
          {" | "}
          GPU = <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>An army of students</span> all solving simple problems at once
        </span>
      </div>
    </SlideLayout>
  );
};
