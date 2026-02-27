import React from "react";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

const LaneArrow: React.FC<{
  fromLane: number;
  toLane: number;
  color: string;
  delay: number;
  laneWidth: number;
  yOffset: number;
  label?: string;
}> = ({ fromLane, toLane, color, delay, laneWidth, yOffset, label }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = interpolate(frame - delay, [0, 0.4 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const x1 = fromLane * laneWidth + laneWidth / 2;
  const x2 = toLane * laneWidth + laneWidth / 2;

  return (
    <svg
      style={{
        position: "absolute",
        left: 0,
        top: yOffset,
        width: "100%",
        height: 40,
        pointerEvents: "none",
        overflow: "visible",
      }}
    >
      <line
        x1={x1}
        y1={0}
        x2={x1 + (x2 - x1) * progress}
        y2={30 * progress}
        stroke={color}
        strokeWidth={2}
        opacity={progress}
      />
      {progress > 0.8 && (
        <polygon
          points={`${x2},30 ${x2 - 4},22 ${x2 + 4},22`}
          fill={color}
          opacity={interpolate(progress, [0.8, 1], [0, 1])}
        />
      )}
      {label && progress > 0.5 && (
        <text
          x={(x1 + x2) / 2}
          y={18}
          fill={color}
          fontSize={10}
          fontFamily={fontFamilyCode}
          textAnchor="middle"
          opacity={interpolate(progress, [0.5, 1], [0, 0.8])}
        >
          {label}
        </text>
      )}
    </svg>
  );
};

export const M3S10_WarpShuffle: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const laneCount = 8;
  const laneWidth = 72;
  const laneColors = [
    THEME.colors.accentRed,
    THEME.colors.accentOrange,
    THEME.colors.accentYellow,
    THEME.colors.nvidiaGreen,
    THEME.colors.accentCyan,
    THEME.colors.accentBlue,
    THEME.colors.accentPurple,
    THEME.colors.accentPink,
  ];

  const insightOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      moduleNumber={3}
      leftWidth="48%"
      left={
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
          <SlideTitle
            title="Warp Shuffle"
            subtitle="Direct register-to-register data exchange between lanes"
          />

          <FadeInText
            text="__shfl_sync Variants"
            delay={0.5 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.nvidiaGreen}
            style={{ marginBottom: 10 }}
          />

          <CodeBlock
            delay={1 * fps}
            title="shfl_sync variants"
            fontSize={14}
            code={`// Read from any lane (direct)
int val = __shfl_sync(mask, var, srcLane);

// Read from lane (laneId - delta)
int val = __shfl_up_sync(mask, var, delta);

// Read from lane (laneId + delta)
int val = __shfl_down_sync(mask, var, delta);

// Read from lane (laneId ^ laneMask)
int val = __shfl_xor_sync(mask, var, laneMask);`}
            highlightLines={[2, 5, 8, 11]}
          />

          <div style={{ marginTop: 12 }}>
            <FadeInText
              text="mask = 0xFFFFFFFF for full warp participation"
              delay={4 * fps}
              fontSize={16}
              color={THEME.colors.textSecondary}
            />
          </div>

          <div
            style={{
              marginTop: "auto",
              padding: "12px 16px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 8,
              border: `2px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: insightOpacity,
            }}
          >
            <span style={{ fontSize: 16, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.5 }}>
              <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>Key insight:</span>{" "}
              No shared memory needed! Data moves directly between thread registers in a single clock cycle.
            </span>
          </div>
        </div>
      }
      right={
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <FadeInText
            text="Data Movement Between Lanes"
            delay={2 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentBlue}
            style={{ marginBottom: 4 }}
          />

          {/* __shfl_down_sync visualization */}
          <div style={{ position: "relative" }}>
            <FadeInText
              text="__shfl_down_sync(mask, val, 2)"
              delay={3 * fps}
              fontSize={14}
              fontWeight={600}
              color={THEME.colors.accentCyan}
              style={{ marginBottom: 8, fontFamily: fontFamilyCode }}
            />

            {/* Source row */}
            <div style={{ display: "flex", gap: 2, marginBottom: 4 }}>
              {Array.from({ length: laneCount }).map((_, i) => {
                const cellSpring = spring({
                  frame: frame - (3.2 * fps + i * 2),
                  fps,
                  config: { damping: 200 },
                });
                return (
                  <div
                    key={i}
                    style={{
                      width: laneWidth - 4,
                      height: 32,
                      backgroundColor: `${laneColors[i]}25`,
                      border: `1.5px solid ${laneColors[i]}`,
                      borderRadius: 4,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 11,
                      color: laneColors[i],
                      fontFamily: fontFamilyCode,
                      fontWeight: 700,
                      opacity: interpolate(cellSpring, [0, 1], [0, 1]),
                    }}
                  >
                    L{i}:{String.fromCharCode(65 + i)}
                  </div>
                );
              })}
            </div>

            {/* Arrows for shfl_down with offset=2 */}
            {Array.from({ length: laneCount - 2 }).map((_, i) => (
              <LaneArrow
                key={i}
                fromLane={i + 2}
                toLane={i}
                color={laneColors[i + 2]}
                delay={4.5 * fps + i * 0.1 * fps}
                laneWidth={laneWidth - 2}
                yOffset={36}
              />
            ))}

            {/* Destination row */}
            <div style={{ display: "flex", gap: 2, marginTop: 44 }}>
              {Array.from({ length: laneCount }).map((_, i) => {
                const showResult = frame > 5.5 * fps;
                const resultOpacity = interpolate(
                  frame - (5.5 * fps + i * 0.05 * fps),
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                );
                const hasNewValue = i < laneCount - 2;
                return (
                  <div
                    key={i}
                    style={{
                      width: laneWidth - 4,
                      height: 32,
                      backgroundColor: hasNewValue
                        ? `${laneColors[i + 2]}20`
                        : "rgba(255,255,255,0.03)",
                      border: `1.5px solid ${hasNewValue ? laneColors[i + 2] : THEME.colors.textMuted}60`,
                      borderRadius: 4,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 11,
                      color: hasNewValue ? laneColors[i + 2] : THEME.colors.textMuted,
                      fontFamily: fontFamilyCode,
                      fontWeight: 700,
                      opacity: showResult ? resultOpacity : 0,
                    }}
                  >
                    {hasNewValue ? `L${i}:${String.fromCharCode(67 + i)}` : `L${i}:--`}
                  </div>
                );
              })}
            </div>
          </div>

          {/* __shfl_xor_sync visualization */}
          <div style={{ position: "relative", marginTop: 16 }}>
            <FadeInText
              text="__shfl_xor_sync(mask, val, 1)"
              delay={6 * fps}
              fontSize={14}
              fontWeight={600}
              color={THEME.colors.accentPurple}
              style={{ marginBottom: 8, fontFamily: fontFamilyCode }}
            />

            <div style={{ display: "flex", gap: 2, marginBottom: 4 }}>
              {Array.from({ length: laneCount }).map((_, i) => {
                const cellSpring = spring({
                  frame: frame - (6.3 * fps + i * 2),
                  fps,
                  config: { damping: 200 },
                });
                return (
                  <div
                    key={i}
                    style={{
                      width: laneWidth - 4,
                      height: 32,
                      backgroundColor: `${laneColors[i]}25`,
                      border: `1.5px solid ${laneColors[i]}`,
                      borderRadius: 4,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 11,
                      color: laneColors[i],
                      fontFamily: fontFamilyCode,
                      fontWeight: 700,
                      opacity: interpolate(cellSpring, [0, 1], [0, 1]),
                    }}
                  >
                    L{i}:{String.fromCharCode(65 + i)}
                  </div>
                );
              })}
            </div>

            {/* XOR arrows: pairs swap (0<->1, 2<->3, etc.) */}
            {Array.from({ length: laneCount / 2 }).map((_, pair) => {
              const left = pair * 2;
              const right = left + 1;
              return (
                <React.Fragment key={pair}>
                  <LaneArrow
                    fromLane={left}
                    toLane={right}
                    color={laneColors[left]}
                    delay={7 * fps + pair * 0.1 * fps}
                    laneWidth={laneWidth - 2}
                    yOffset={36}
                  />
                  <LaneArrow
                    fromLane={right}
                    toLane={left}
                    color={laneColors[right]}
                    delay={7 * fps + pair * 0.1 * fps}
                    laneWidth={laneWidth - 2}
                    yOffset={36}
                  />
                </React.Fragment>
              );
            })}

            {/* Result row for XOR */}
            <div style={{ display: "flex", gap: 2, marginTop: 44 }}>
              {Array.from({ length: laneCount }).map((_, i) => {
                const partner = i ^ 1;
                const resultOpacity = interpolate(
                  frame - (8 * fps + i * 0.05 * fps),
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                );
                return (
                  <div
                    key={i}
                    style={{
                      width: laneWidth - 4,
                      height: 32,
                      backgroundColor: `${laneColors[partner]}20`,
                      border: `1.5px solid ${laneColors[partner]}60`,
                      borderRadius: 4,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 11,
                      color: laneColors[partner],
                      fontFamily: fontFamilyCode,
                      fontWeight: 700,
                      opacity: frame > 8 * fps ? resultOpacity : 0,
                    }}
                  >
                    L{i}:{String.fromCharCode(65 + partner)}
                  </div>
                );
              })}
            </div>

            <FadeInText
              text="XOR(1) swaps adjacent pairs -- used in butterfly reductions"
              delay={8.5 * fps}
              fontSize={14}
              color={THEME.colors.textSecondary}
              style={{ marginTop: 8 }}
            />
          </div>
        </div>
      }
    />
  );
};
