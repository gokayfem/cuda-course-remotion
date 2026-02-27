import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint, FadeInText } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

const ATOMIC_OPS = [
  { name: "atomicAdd", desc: "*addr += val" },
  { name: "atomicSub", desc: "*addr -= val" },
  { name: "atomicMin", desc: "*addr = min(*addr, val)" },
  { name: "atomicMax", desc: "*addr = max(*addr, val)" },
  { name: "atomicExch", desc: "*addr = val (swap)" },
  { name: "atomicCAS", desc: "Compare And Swap" },
  { name: "atomicAnd", desc: "*addr &= val" },
  { name: "atomicOr", desc: "*addr |= val" },
];

export const M3S08_AtomicOps: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const raceCode = `// BUG: Race condition!
__global__ void count(int* data, int* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (data[idx] > 0)
    *result += 1;  // Multiple threads read-modify-write
}`;

  const fixCode = `// FIX: Atomic operation
__global__ void count(int* data, int* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (data[idx] > 0)
    atomicAdd(result, 1);  // Serialized, correct
}`;

  // Race condition visualization
  const THREAD_COUNT = 4;
  const raceDelay = 3 * fps;
  const raceProgress = interpolate(
    frame - raceDelay,
    [0, 2.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const threadColors = [
    THEME.colors.nvidiaGreen,
    THEME.colors.accentBlue,
    THEME.colors.accentOrange,
    THEME.colors.accentPurple,
  ];

  const renderRaceVisualization = () => {
    const visSpring = spring({
      frame: frame - raceDelay,
      fps,
      config: { damping: 200 },
    });
    const visOpacity = interpolate(visSpring, [0, 1], [0, 1]);

    // All threads try to write to same address
    const TARGET_X = 220;
    const TARGET_Y = 50;
    const containerHeight = 120;

    return (
      <div
        style={{
          position: "relative",
          height: containerHeight,
          opacity: visOpacity,
          marginTop: 8,
        }}
      >
        {/* Target memory address */}
        <div
          style={{
            position: "absolute",
            left: TARGET_X,
            top: TARGET_Y - 12,
            width: 80,
            height: 40,
            backgroundColor: "rgba(255,82,82,0.15)",
            border: `2px solid ${THEME.colors.accentRed}`,
            borderRadius: 6,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 14,
            color: THEME.colors.accentRed,
            fontFamily: fontFamilyCode,
            fontWeight: 700,
          }}
        >
          result
        </div>

        {/* Thread arrows converging */}
        {Array.from({ length: THREAD_COUNT }).map((_, i) => {
          const startX = 20;
          const startY = i * 28 + 8;
          const arrowProgress = interpolate(
            raceProgress,
            [i * 0.15, i * 0.15 + 0.4],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          const currentX = startX + (TARGET_X - startX - 10) * arrowProgress;
          const currentY =
            startY + (TARGET_Y - startY) * arrowProgress;

          return (
            <React.Fragment key={i}>
              <div
                style={{
                  position: "absolute",
                  left: startX,
                  top: startY,
                  fontSize: 12,
                  color: threadColors[i],
                  fontFamily: fontFamilyCode,
                  fontWeight: 700,
                }}
              >
                T{i}
              </div>
              {/* Arrow line */}
              <div
                style={{
                  position: "absolute",
                  left: startX + 24,
                  top: startY + 6,
                  width: currentX - startX - 24,
                  height: 2,
                  backgroundColor: threadColors[i],
                  borderRadius: 1,
                  transformOrigin: "left center",
                  transform: `rotate(${Math.atan2(currentY - startY - 6, currentX - startX - 24) * (180 / Math.PI)}deg)`,
                }}
              />
            </React.Fragment>
          );
        })}

        {/* Collision indicator */}
        {raceProgress > 0.7 && (
          <div
            style={{
              position: "absolute",
              left: TARGET_X + 90,
              top: TARGET_Y - 8,
              fontSize: 14,
              color: THEME.colors.accentRed,
              fontFamily: fontFamilyBody,
              fontWeight: 700,
              opacity: interpolate(
                raceProgress,
                [0.7, 1],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            RACE!
          </div>
        )}
      </div>
    );
  };

  const renderLeft = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <SlideTitle
        title="Atomic Operations"
        subtitle="Safe concurrent read-modify-write"
      />

      <FadeInText
        text="The Problem: Race Condition"
        fontSize={18}
        fontWeight={700}
        delay={0.3 * fps}
        color={THEME.colors.accentRed}
      />
      <CodeBlock
        code={raceCode}
        title="race_condition.cu"
        fontSize={14}
        delay={0.5 * fps}
        highlightLines={[5]}
      />

      {renderRaceVisualization()}

      <FadeInText
        text="The Fix: atomicAdd"
        fontSize={18}
        fontWeight={700}
        delay={5 * fps}
        color={THEME.colors.nvidiaGreen}
      />
      <CodeBlock
        code={fixCode}
        title="atomic_fix.cu"
        fontSize={14}
        delay={5.2 * fps}
        highlightLines={[5]}
      />
    </div>
  );

  const renderRight = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <FadeInText
        text="Atomic Operations"
        fontSize={20}
        fontWeight={700}
        delay={1 * fps}
        color={THEME.colors.accentBlue}
      />

      {/* Atomic ops table */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 4,
          marginBottom: 8,
        }}
      >
        {ATOMIC_OPS.map((op, i) => {
          const opDelay = 1.5 * fps + i * 0.12 * fps;
          const opSpring = spring({
            frame: frame - opDelay,
            fps,
            config: { damping: 200 },
          });
          const opOpacity = interpolate(opSpring, [0, 1], [0, 1]);

          return (
            <div
              key={op.name}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "4px 8px",
                backgroundColor:
                  i % 2 === 0
                    ? "rgba(255,255,255,0.02)"
                    : "transparent",
                borderRadius: 4,
                opacity: opOpacity,
              }}
            >
              <code
                style={{
                  fontSize: 14,
                  color: THEME.colors.syntaxFunction,
                  fontFamily: fontFamilyCode,
                  fontWeight: 700,
                  width: 110,
                  flexShrink: 0,
                }}
              >
                {op.name}
              </code>
              <span
                style={{
                  fontSize: 14,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyCode,
                }}
              >
                {op.desc}
              </span>
            </div>
          );
        })}
      </div>

      <BulletPoint
        index={0}
        delay={4 * fps}
        text="Guaranteed atomicity"
        subtext="Read-modify-write as a single, indivisible operation."
        highlight
      />
      <BulletPoint
        index={1}
        delay={4 * fps}
        text="Returns old value"
        subtext="atomicAdd(addr, val) returns the previous value at addr."
      />
      <BulletPoint
        index={2}
        delay={4 * fps}
        text="Performance cost"
        subtext="Atomics serialize access. Many threads hitting the same address = bottleneck."
      />

      {/* Warning box */}
      <div
        style={{
          marginTop: 8,
          padding: "10px 14px",
          backgroundColor: "rgba(255,171,64,0.10)",
          border: `1px solid ${THEME.colors.accentOrange}50`,
          borderRadius: 8,
          opacity: interpolate(
            frame - 6.5 * fps,
            [0, 0.5 * fps],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          ),
        }}
      >
        <div
          style={{
            fontSize: 15,
            color: THEME.colors.accentOrange,
            fontFamily: fontFamilyBody,
            fontWeight: 700,
          }}
        >
          Atomics serialize thread access to the same address.
        </div>
        <div
          style={{
            fontSize: 13,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
            marginTop: 4,
          }}
        >
          Use shared memory atomics first, then one global atomic per bin for best performance.
        </div>
      </div>
    </div>
  );

  return (
    <TwoColumnLayout
      moduleNumber={3}
      left={renderLeft()}
      right={renderRight()}
      leftWidth="50%"
    />
  );
};
