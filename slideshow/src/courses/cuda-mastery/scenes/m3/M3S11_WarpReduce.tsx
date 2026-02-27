import React from "react";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

const LANE_COUNT = 8;
const CELL_SIZE = 56;
const CELL_GAP = 4;

const reductionSteps = [
  { offset: 4, label: "Step 1: offset=16 (shown as 4)" },
  { offset: 2, label: "Step 2: offset=8 (shown as 2)" },
  { offset: 1, label: "Step 3: offset=4 (shown as 1)" },
];

const ButterflyStep: React.FC<{
  stepIndex: number;
  offset: number;
  values: number[];
  delay: number;
  label: string;
}> = ({ stepIndex, offset, values, delay, label }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const stepSpring = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });
  const stepOpacity = interpolate(stepSpring, [0, 1], [0, 1]);

  const arrowProgress = interpolate(
    frame - (delay + 0.3 * fps),
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const resultOpacity = interpolate(
    frame - (delay + 0.8 * fps),
    [0, 0.3 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const colors = [
    THEME.colors.accentRed,
    THEME.colors.accentOrange,
    THEME.colors.accentYellow,
    THEME.colors.nvidiaGreen,
    THEME.colors.accentCyan,
    THEME.colors.accentBlue,
    THEME.colors.accentPurple,
    THEME.colors.accentPink,
  ];

  const newValues = values.map((v, i) =>
    i + offset < LANE_COUNT ? v + values[i + offset] : v
  );

  const activeReceivers = Array.from({ length: LANE_COUNT }).filter(
    (_, i) => i + offset < LANE_COUNT
  ).length;

  return (
    <div style={{ opacity: stepOpacity, marginBottom: 12 }}>
      <div
        style={{
          fontSize: 14,
          color: THEME.colors.accentCyan,
          fontFamily: fontFamilyBody,
          fontWeight: 600,
          marginBottom: 6,
        }}
      >
        {label}
      </div>

      <div style={{ position: "relative", height: 80 }}>
        {/* Source values */}
        <div style={{ display: "flex", gap: CELL_GAP, position: "absolute", top: 0 }}>
          {values.map((v, i) => {
            const isActive = i + offset < LANE_COUNT;
            const isSource = i >= offset && i < LANE_COUNT;
            return (
              <div
                key={i}
                style={{
                  width: CELL_SIZE,
                  height: 30,
                  backgroundColor: isActive
                    ? `${colors[i]}20`
                    : "rgba(255,255,255,0.03)",
                  border: `1.5px solid ${isActive ? colors[i] : THEME.colors.textMuted}50`,
                  borderRadius: 4,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 12,
                  color: isActive ? colors[i] : THEME.colors.textMuted,
                  fontFamily: fontFamilyCode,
                  fontWeight: 700,
                }}
              >
                {v}
              </div>
            );
          })}
        </div>

        {/* Arrows */}
        <svg
          style={{
            position: "absolute",
            top: 30,
            left: 0,
            width: LANE_COUNT * (CELL_SIZE + CELL_GAP),
            height: 20,
            overflow: "visible",
          }}
        >
          {Array.from({ length: LANE_COUNT }).map((_, i) => {
            if (i + offset >= LANE_COUNT) return null;
            const x1 = (i + offset) * (CELL_SIZE + CELL_GAP) + CELL_SIZE / 2;
            const x2 = i * (CELL_SIZE + CELL_GAP) + CELL_SIZE / 2;
            const curX = x1 + (x2 - x1) * arrowProgress;
            return (
              <g key={i}>
                <line
                  x1={x1}
                  y1={2}
                  x2={curX}
                  y2={16 * arrowProgress}
                  stroke={colors[i + offset]}
                  strokeWidth={1.5}
                  opacity={arrowProgress * 0.8}
                />
                {arrowProgress > 0.8 && (
                  <polygon
                    points={`${x2},18 ${x2 - 3},12 ${x2 + 3},12`}
                    fill={colors[i + offset]}
                    opacity={interpolate(arrowProgress, [0.8, 1], [0, 0.8])}
                  />
                )}
                <text
                  x={(x1 + x2) / 2}
                  y={12}
                  fill={THEME.colors.textMuted}
                  fontSize={9}
                  fontFamily={fontFamilyCode}
                  textAnchor="middle"
                  opacity={arrowProgress * 0.6}
                >
                  +
                </text>
              </g>
            );
          })}
        </svg>

        {/* Result values */}
        <div style={{ display: "flex", gap: CELL_GAP, position: "absolute", top: 48 }}>
          {newValues.map((v, i) => {
            const wasActive = i + offset < LANE_COUNT;
            return (
              <div
                key={i}
                style={{
                  width: CELL_SIZE,
                  height: 30,
                  backgroundColor: wasActive
                    ? `${THEME.colors.nvidiaGreen}15`
                    : "rgba(255,255,255,0.02)",
                  border: `1.5px solid ${wasActive ? THEME.colors.nvidiaGreen : THEME.colors.textMuted}40`,
                  borderRadius: 4,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 12,
                  color: wasActive ? THEME.colors.nvidiaGreen : THEME.colors.textMuted,
                  fontFamily: fontFamilyCode,
                  fontWeight: 700,
                  opacity: resultOpacity,
                }}
              >
                {v === values[i] && !wasActive ? v : v}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export const M3S11_WarpReduce: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const initialValues = [1, 2, 3, 4, 5, 6, 7, 8];

  const step0Values = initialValues;
  const step1Values = step0Values.map((v, i) =>
    i + 4 < LANE_COUNT ? v + step0Values[i + 4] : v
  );
  const step2Values = step1Values.map((v, i) =>
    i + 2 < LANE_COUNT ? v + step1Values[i + 2] : v
  );

  const allStepValues = [step0Values, step1Values, step2Values];

  const compareOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout moduleNumber={3}>
      <SlideTitle
        title="Warp-Level Reduction"
        subtitle="Butterfly pattern using __shfl_down_sync -- no shared memory needed"
      />

      <div style={{ display: "flex", gap: 36, flex: 1 }}>
        {/* Left: butterfly animation */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text="Butterfly Reduction Pattern (8 lanes shown)"
            delay={0.5 * fps}
            fontSize={18}
            fontWeight={600}
            color={THEME.colors.accentBlue}
            style={{ marginBottom: 12 }}
          />

          {reductionSteps.map((step, idx) => (
            <ButterflyStep
              key={idx}
              stepIndex={idx}
              offset={step.offset}
              values={allStepValues[idx]}
              delay={(1.5 + idx * 2.5) * fps}
              label={step.label}
            />
          ))}

          {/* Final result highlight */}
          <div
            style={{
              marginTop: 8,
              padding: "8px 14px",
              backgroundColor: "rgba(118,185,0,0.1)",
              borderRadius: 8,
              border: `2px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: interpolate(
                frame - 8.5 * fps,
                [0, 0.4 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <span style={{ fontSize: 16, color: THEME.colors.nvidiaGreen, fontWeight: 700, fontFamily: fontFamilyBody }}>
              Lane 0 holds the sum: 1+2+3+4+5+6+7+8 = 36
            </span>
          </div>
        </div>

        {/* Right: code + comparison */}
        <div style={{ flex: 1 }}>
          <CodeBlock
            delay={1 * fps}
            title="warp_reduce.cu"
            fontSize={15}
            code={`__device__ float warpReduceSum(float val) {
    // Full warp participates
    for (int offset = warpSize / 2;
         offset > 0; offset >>= 1) {
        val += __shfl_down_sync(
            0xFFFFFFFF, val, offset
        );
    }
    return val;  // lane 0 has total
}

// 5 iterations for 32 threads:
// offset=16, 8, 4, 2, 1`}
            highlightLines={[5, 6]}
          />

          {/* Speed comparison */}
          <div style={{ marginTop: 20, opacity: compareOpacity }}>
            <FadeInText
              text="Speed Comparison"
              delay={9 * fps}
              fontSize={18}
              fontWeight={700}
              color={THEME.colors.accentOrange}
              style={{ marginBottom: 12 }}
            />

            {[
              { label: "Shared memory reduction", cycles: "~30 cycles", width: "85%", color: THEME.colors.accentRed },
              { label: "Warp shuffle reduction", cycles: "~5 cycles", width: "17%", color: THEME.colors.nvidiaGreen },
            ].map((bar, i) => {
              const barSpring = spring({
                frame: frame - (9.5 * fps + i * 0.4 * fps),
                fps,
                config: { damping: 200 },
              });
              const barOpacity = interpolate(barSpring, [0, 1], [0, 1]);
              const barWidth = interpolate(barSpring, [0, 1], [0, 1]);

              return (
                <div key={i} style={{ marginBottom: 12, opacity: barOpacity }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 14, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody }}>
                      {bar.label}
                    </span>
                    <span style={{ fontSize: 14, color: bar.color, fontFamily: fontFamilyCode, fontWeight: 700 }}>
                      {bar.cycles}
                    </span>
                  </div>
                  <div
                    style={{
                      height: 14,
                      backgroundColor: "rgba(255,255,255,0.06)",
                      borderRadius: 7,
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: `calc(${bar.width} * ${barWidth})`,
                        backgroundColor: bar.color,
                        borderRadius: 7,
                        opacity: 0.8,
                      }}
                    />
                  </div>
                </div>
              );
            })}

            <FadeInText
              text="6x faster -- no synchronization barriers needed!"
              delay={10.5 * fps}
              fontSize={16}
              fontWeight={600}
              color={THEME.colors.nvidiaGreen}
              style={{ marginTop: 8 }}
            />
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
