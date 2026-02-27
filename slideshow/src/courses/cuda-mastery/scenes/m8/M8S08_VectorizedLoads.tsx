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

const BUFFER_W = 180;
const BUFFER_H = 40;

export const M8S08_VectorizedLoads: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const box1Opacity = interpolate(
    frame - 1.5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const box2Opacity = interpolate(
    frame - 5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Double buffering ping-pong animation
  const pingPongStart = 6.5 * fps;
  const pingPongCycle = 2 * fps;
  const pingPongElapsed = Math.max(0, frame - pingPongStart);
  const pingPongPhase = (pingPongElapsed % pingPongCycle) / pingPongCycle;
  const isLoadingA = pingPongPhase < 0.5;

  const pingPongOpacity = interpolate(
    frame - pingPongStart,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const codeOpacity = interpolate(
    frame - 3 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={8}
      leftWidth="50%"
      left={
        <div style={{ width: 560 }}>
          <SlideTitle
            title="Vectorized Loads & Double Buffering"
            subtitle="Maximizing memory throughput"
          />

          {/* Technique box 1: float4 */}
          <div
            style={{
              marginTop: 12,
              padding: "16px 20px",
              backgroundColor: "rgba(26,26,46,0.8)",
              borderRadius: 10,
              border: `2px solid ${THEME.colors.accentBlue}60`,
              opacity: box1Opacity,
              width: 520,
            }}
          >
            <div
              style={{
                fontSize: 18,
                fontWeight: 800,
                color: THEME.colors.accentBlue,
                fontFamily: fontFamilyBody,
                marginBottom: 10,
              }}
            >
              float4 Vectorized Loads
            </div>
            <div
              style={{
                fontSize: 15,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                marginBottom: 8,
                lineHeight: 1.5,
              }}
            >
              Load 128 bits per instruction
            </div>
            <div
              style={{
                fontSize: 15,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                marginBottom: 12,
                lineHeight: 1.5,
              }}
            >
              4x bandwidth efficiency for global{"\u2192"}shared transfers
            </div>

            {/* Code snippet */}
            <div
              style={{
                padding: "8px 14px",
                backgroundColor: "#0d1117",
                borderRadius: 6,
                border: "1px solid rgba(255,255,255,0.08)",
                opacity: codeOpacity,
              }}
            >
              <pre
                style={{
                  margin: 0,
                  fontFamily: fontFamilyCode,
                  fontSize: 13,
                  lineHeight: 1.5,
                  color: THEME.colors.textCode,
                  whiteSpace: "pre-wrap",
                }}
              >
                <span style={{ color: THEME.colors.syntaxType }}>float4</span>
                {" tmp = "}
                <span style={{ color: THEME.colors.syntaxKeyword }}>reinterpret_cast</span>
                {"<"}
                <span style={{ color: THEME.colors.syntaxType }}>float4</span>
                {"*>(&A[...])["}
                <span style={{ color: THEME.colors.syntaxNumber }}>0</span>
                {"];"}
              </pre>
            </div>

            {/* Visual: 4 float cells vs 1 float cell */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                marginTop: 12,
                opacity: codeOpacity,
              }}
            >
              <div style={{ display: "flex", gap: 2 }}>
                {[1, 2, 3, 4].map((v) => (
                  <div
                    key={v}
                    style={{
                      width: 40,
                      height: 24,
                      borderRadius: 3,
                      backgroundColor: `${THEME.colors.accentBlue}40`,
                      border: `1px solid ${THEME.colors.accentBlue}`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 11,
                      fontWeight: 700,
                      color: THEME.colors.accentBlue,
                      fontFamily: fontFamilyCode,
                    }}
                  >
                    f{v}
                  </div>
                ))}
              </div>
              <span
                style={{
                  fontSize: 13,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyBody,
                }}
              >
                = 1 load instruction (128 bits)
              </span>
            </div>
          </div>

          {/* Technique box 2: Double Buffering */}
          <div
            style={{
              marginTop: 20,
              padding: "16px 20px",
              backgroundColor: "rgba(26,26,46,0.8)",
              borderRadius: 10,
              border: `2px solid ${THEME.colors.nvidiaGreen}60`,
              opacity: box2Opacity,
              width: 520,
            }}
          >
            <div
              style={{
                fontSize: 18,
                fontWeight: 800,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                marginBottom: 10,
              }}
            >
              Double Buffering
            </div>
            <div
              style={{
                fontSize: 15,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                marginBottom: 12,
                lineHeight: 1.5,
              }}
            >
              Prefetch next tile while computing current
            </div>

            {/* Ping-pong diagram */}
            <div style={{ opacity: pingPongOpacity }}>
              <svg width={480} height={100} viewBox="0 0 480 100">
                {/* Buffer A */}
                <rect
                  x={10}
                  y={10}
                  width={BUFFER_W}
                  height={BUFFER_H}
                  rx={6}
                  fill={isLoadingA ? `${THEME.colors.accentOrange}30` : `${THEME.colors.nvidiaGreen}30`}
                  stroke={isLoadingA ? THEME.colors.accentOrange : THEME.colors.nvidiaGreen}
                  strokeWidth={2}
                />
                <text
                  x={10 + BUFFER_W / 2}
                  y={35}
                  textAnchor="middle"
                  fill={isLoadingA ? THEME.colors.accentOrange : THEME.colors.nvidiaGreen}
                  fontSize={13}
                  fontWeight={700}
                  fontFamily={fontFamilyBody}
                >
                  {isLoadingA ? "Loading tile N+1" : "Computing tile N"}
                </text>

                {/* Buffer B */}
                <rect
                  x={10}
                  y={56}
                  width={BUFFER_W}
                  height={BUFFER_H}
                  rx={6}
                  fill={!isLoadingA ? `${THEME.colors.accentOrange}30` : `${THEME.colors.nvidiaGreen}30`}
                  stroke={!isLoadingA ? THEME.colors.accentOrange : THEME.colors.nvidiaGreen}
                  strokeWidth={2}
                />
                <text
                  x={10 + BUFFER_W / 2}
                  y={81}
                  textAnchor="middle"
                  fill={!isLoadingA ? THEME.colors.accentOrange : THEME.colors.nvidiaGreen}
                  fontSize={13}
                  fontWeight={700}
                  fontFamily={fontFamilyBody}
                >
                  {!isLoadingA ? "Loading tile N+1" : "Computing tile N"}
                </text>

                {/* Overlap label */}
                <text
                  x={240}
                  y={50}
                  textAnchor="start"
                  fill={THEME.colors.textSecondary}
                  fontSize={13}
                  fontFamily={fontFamilyBody}
                >
                  Overlap: load + compute
                </text>

                {/* Arrow indicating overlap */}
                <line
                  x1={200}
                  y1={30}
                  x2={230}
                  y2={45}
                  stroke={THEME.colors.textMuted}
                  strokeWidth={1.5}
                  strokeDasharray="3,2"
                />
                <line
                  x1={200}
                  y1={76}
                  x2={230}
                  y2={55}
                  stroke={THEME.colors.textMuted}
                  strokeWidth={1.5}
                  strokeDasharray="3,2"
                />
              </svg>
            </div>

            <div
              style={{
                fontSize: 14,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                marginTop: 4,
              }}
            >
              Hides memory latency behind computation
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 60, width: 480 }}>
          <BulletPoint
            index={0}
            delay={8 * fps}
            text="Combines all optimizations: tiling + registers + vectors + prefetch"
            icon="1"
          />
          <BulletPoint
            index={1}
            delay={8 * fps}
            text="Shared memory layout avoids bank conflicts"
            icon="2"
            highlight
          />
          <BulletPoint
            index={2}
            delay={8 * fps}
            text="Achieves 80-90% of cuBLAS performance"
            icon="3"
          />
          <BulletPoint
            index={3}
            delay={8 * fps}
            text="cuBLAS adds: auto-tuning, Tensor Cores, assembly-level tricks"
            icon="4"
            highlight
          />
        </div>
      }
    />
  );
};
