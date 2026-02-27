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
import { fontFamilyBody } from "../../../../styles/fonts";

type Module = {
  number: number;
  title: string;
  isCurrent: boolean;
};

const modules: Module[] = [
  { number: 1, title: "GPU Architecture", isCurrent: false },
  { number: 2, title: "Memory Hierarchy", isCurrent: false },
  { number: 3, title: "Thread Sync", isCurrent: false },
  { number: 4, title: "Parallel Patterns", isCurrent: false },
  { number: 5, title: "Performance Optimization", isCurrent: false },
  { number: 6, title: "Streams & Concurrency", isCurrent: false },
  { number: 7, title: "Libraries", isCurrent: false },
  { number: 8, title: "Matrix Multiplication", isCurrent: false },
  { number: 9, title: "Attention & Transformers", isCurrent: false },
  { number: 10, title: "Advanced Topics", isCurrent: true },
];

export const M10S15_CurriculumReview: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const bottomOpacity = interpolate(
    frame - 10 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="accent" moduleNumber={10}>
      <SlideTitle
        title="Your CUDA Mastery Journey -- All 10 Modules"
        subtitle="From first kernel to advanced GPU programming"
      />

      {/* Timeline */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 0,
          flex: 1,
          width: 1776,
          position: "relative",
        }}
      >
        {/* Horizontal timeline layout: two rows of 5 */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 28,
            flex: 1,
            justifyContent: "center",
          }}
        >
          {[0, 1].map((rowIdx) => (
            <div
              key={rowIdx}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 0,
                width: 1776,
              }}
            >
              {modules.slice(rowIdx * 5, rowIdx * 5 + 5).map((mod, i) => {
                const globalIdx = rowIdx * 5 + i;
                const nodeDelay = 1 * fps + globalIdx * 0.5 * fps;
                const nodeSpring = spring({
                  frame: frame - nodeDelay,
                  fps,
                  config: { damping: 200 },
                });
                const nodeOpacity = interpolate(nodeSpring, [0, 1], [0, 1]);
                const nodeScale = interpolate(nodeSpring, [0, 1], [0.7, 1]);

                // Connecting line animation
                const lineDelay = nodeDelay + 0.2 * fps;
                const lineWidth = interpolate(
                  frame - lineDelay,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                );

                // Glow effect for current module
                const glowIntensity = mod.isCurrent
                  ? interpolate(
                      Math.sin(frame * 0.08),
                      [-1, 1],
                      [0.4, 1]
                    )
                  : 0;

                return (
                  <React.Fragment key={mod.number}>
                    {/* Node */}
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        opacity: nodeOpacity,
                        transform: `scale(${nodeScale})`,
                        flexShrink: 0,
                        width: 160,
                      }}
                    >
                      {/* Circle */}
                      <div
                        style={{
                          width: 52,
                          height: 52,
                          borderRadius: 26,
                          backgroundColor: mod.isCurrent
                            ? THEME.colors.nvidiaGreen
                            : `${THEME.colors.nvidiaGreen}30`,
                          border: `3px solid ${THEME.colors.nvidiaGreen}`,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 20,
                          fontWeight: 800,
                          color: mod.isCurrent
                            ? "#0a0a0a"
                            : THEME.colors.nvidiaGreen,
                          fontFamily: fontFamilyBody,
                          boxShadow: mod.isCurrent
                            ? `0 0 ${20 + glowIntensity * 20}px ${THEME.colors.nvidiaGreen}${Math.round(glowIntensity * 80).toString(16).padStart(2, "0")}`
                            : "none",
                        }}
                      >
                        {mod.number}
                      </div>

                      {/* Label */}
                      <div
                        style={{
                          marginTop: 10,
                          fontSize: 14,
                          fontWeight: mod.isCurrent ? 700 : 500,
                          color: mod.isCurrent
                            ? THEME.colors.nvidiaGreen
                            : THEME.colors.textSecondary,
                          fontFamily: fontFamilyBody,
                          textAlign: "center",
                          lineHeight: 1.3,
                          width: 140,
                        }}
                      >
                        {mod.title}
                      </div>

                      {/* Checkmark for completed */}
                      <div
                        style={{
                          marginTop: 6,
                          fontSize: 16,
                          color: THEME.colors.nvidiaGreen,
                        }}
                      >
                        {"\u2713"}
                      </div>
                    </div>

                    {/* Connecting line */}
                    {i < 4 && (
                      <div
                        style={{
                          flex: 1,
                          height: 3,
                          backgroundColor: `${THEME.colors.nvidiaGreen}40`,
                          borderRadius: 2,
                          position: "relative",
                          overflow: "hidden",
                          marginTop: -30,
                        }}
                      >
                        <div
                          style={{
                            height: "100%",
                            width: `${lineWidth * 100}%`,
                            backgroundColor: THEME.colors.nvidiaGreen,
                            borderRadius: 2,
                          }}
                        />
                      </div>
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Bottom message */}
      <div
        style={{
          marginTop: 8,
          padding: "14px 28px",
          background: "linear-gradient(135deg, rgba(118,185,0,0.12), rgba(79,195,247,0.08))",
          borderRadius: 12,
          border: `1px solid ${THEME.colors.nvidiaGreen}30`,
          textAlign: "center",
          width: 1776,
          opacity: bottomOpacity,
        }}
      >
        <div
          style={{
            fontSize: 20,
            fontWeight: 700,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyBody,
          }}
        >
          From "Hello GPU" to Flash Attention -- you've come a long way!
        </div>
      </div>
    </SlideLayout>
  );
};
