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

const ROW_W = 260;
const ROW_H = 24;
const THREAD_COUNT = 5;

export const M8S04_NaiveAnalysis: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Staggered text reveals
  const line1Opacity = interpolate(
    frame - 1.5 * fps,
    [0, 0.4 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );
  const line2Opacity = interpolate(
    frame - 2.5 * fps,
    [0, 0.4 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );
  const line3Opacity = interpolate(
    frame - 3.5 * fps,
    [0, 0.4 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Arrows diagram animation
  const arrowsOpacity = interpolate(
    frame - 4.5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Arithmetic intensity formula
  const formulaOpacity = interpolate(
    frame - 6 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const solutionOpacity = interpolate(
    frame - 10 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={8}
      leftWidth="52%"
      left={
        <div style={{ width: 600 }}>
          <SlideTitle
            title="Why Naive is Terrible"
            subtitle="Memory access analysis"
          />

          {/* Animated text reveals */}
          <div style={{ marginTop: 12 }}>
            <div
              style={{
                fontSize: 18,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
                marginBottom: 14,
                opacity: line1Opacity,
                lineHeight: 1.5,
              }}
            >
              Each thread reads{" "}
              <span style={{ color: THEME.colors.accentBlue, fontWeight: 700 }}>
                K elements from A row
              </span>
            </div>
            <div
              style={{
                fontSize: 18,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
                marginBottom: 14,
                opacity: line2Opacity,
                lineHeight: 1.5,
              }}
            >
              Each thread reads{" "}
              <span style={{ color: THEME.colors.accentPurple, fontWeight: 700 }}>
                K elements from B column
              </span>
            </div>
            <div
              style={{
                fontSize: 18,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
                marginBottom: 24,
                opacity: line3Opacity,
                lineHeight: 1.5,
              }}
            >
              <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>
                N threads read the SAME A row
              </span>
              {" \u2192 N redundant reads!"}
            </div>
          </div>

          {/* Arrows diagram: multiple threads pointing at same row */}
          <div
            style={{
              opacity: arrowsOpacity,
              marginTop: 8,
              width: 500,
            }}
          >
            <svg width={500} height={180} viewBox="0 0 500 180">
              {/* A row block */}
              <rect
                x={20}
                y={70}
                width={ROW_W}
                height={ROW_H}
                rx={4}
                fill={`${THEME.colors.accentBlue}30`}
                stroke={THEME.colors.accentBlue}
                strokeWidth={1.5}
              />
              <text
                x={150}
                y={87}
                textAnchor="middle"
                fill={THEME.colors.accentBlue}
                fontSize={13}
                fontWeight={700}
                fontFamily={fontFamilyBody}
              >
                A[row 0] (K elements)
              </text>

              {/* Thread arrows */}
              {Array.from({ length: THREAD_COUNT }).map((_, i) => {
                const arrowDelay = (5 + i * 0.3) * fps;
                const arrowOpacity = interpolate(
                  frame - arrowDelay,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                );

                const tx = 330 + (i % 3) * 55;
                const ty = 20 + i * 30;

                return (
                  <g key={i} opacity={arrowOpacity}>
                    <line
                      x1={280}
                      y1={82}
                      x2={tx}
                      y2={ty + 10}
                      stroke={THEME.colors.accentOrange}
                      strokeWidth={1.5}
                      strokeDasharray="4,3"
                    />
                    <circle
                      cx={tx}
                      cy={ty + 10}
                      r={8}
                      fill={`${THEME.colors.accentOrange}30`}
                      stroke={THEME.colors.accentOrange}
                      strokeWidth={1.5}
                    />
                    <text
                      x={tx}
                      y={ty + 14}
                      textAnchor="middle"
                      fill={THEME.colors.accentOrange}
                      fontSize={10}
                      fontWeight={700}
                      fontFamily={fontFamilyCode}
                    >
                      T{i}
                    </text>
                  </g>
                );
              })}

              {/* Label */}
              <text
                x={400}
                y={170}
                textAnchor="middle"
                fill={THEME.colors.accentRed}
                fontSize={13}
                fontWeight={700}
                fontFamily={fontFamilyBody}
              >
                N threads, same data!
              </text>
            </svg>
          </div>

          {/* Arithmetic intensity formula */}
          <div
            style={{
              marginTop: 12,
              padding: "12px 18px",
              backgroundColor: "rgba(255,82,82,0.08)",
              border: `1px solid ${THEME.colors.accentRed}40`,
              borderRadius: 8,
              opacity: formulaOpacity,
              width: 480,
            }}
          >
            <div
              style={{
                fontSize: 16,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyCode,
                fontWeight: 600,
                lineHeight: 1.6,
              }}
            >
              AI = 2K FLOPs / (2K x 4 bytes) ={" "}
              <span style={{ color: THEME.colors.accentRed, fontWeight: 800 }}>
                0.25 FLOPs/byte
              </span>
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 40, width: 460 }}>
          <div
            style={{
              fontSize: 22,
              fontWeight: 800,
              color: THEME.colors.accentRed,
              fontFamily: fontFamilyBody,
              marginBottom: 24,
              opacity: interpolate(
                frame - 3 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            The Problem
          </div>

          <BulletPoint
            index={0}
            delay={4 * fps}
            text="Arithmetic intensity = 0.25"
            subtext="Deeply memory-bound"
            icon="!"
            highlight
          />
          <BulletPoint
            index={1}
            delay={4 * fps}
            text="N^2 threads each read K values"
            subtext="= N^2 K total global reads"
            icon="!"
          />
          <BulletPoint
            index={2}
            delay={4 * fps}
            text="But only N^2 K FLOPs of useful compute"
            icon="!"
          />
          <BulletPoint
            index={3}
            delay={4 * fps}
            text="Data reuse is ZERO between threads"
            icon="!"
            highlight
          />

          {/* Solution hint */}
          <div
            style={{
              marginTop: 32,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: solutionOpacity,
              width: 420,
            }}
          >
            <span
              style={{
                fontSize: 17,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
              }}
            >
              Solution: Share data between threads using shared memory
            </span>
          </div>
        </div>
      }
    />
  );
};
