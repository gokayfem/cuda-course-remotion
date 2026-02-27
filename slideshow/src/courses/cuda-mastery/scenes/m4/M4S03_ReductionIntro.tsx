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
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M4S03_ReductionIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Tree diagram: 8 numbers reducing in 3 steps
  const inputValues = [4, 7, 2, 5, 1, 8, 3, 6];
  const levels = [
    inputValues,
    [11, 7, 9, 9],
    [18, 18],
    [36],
  ];

  const NODE_W = 56;
  const NODE_H = 38;
  const TREE_W = 520;
  const LEVEL_V_GAP = 24;

  const getNodeX = (level: number, index: number): number => {
    const count = levels[level].length;
    if (count === 1) return TREE_W / 2;
    return (NODE_W / 2) + index * ((TREE_W - NODE_W) / (count - 1));
  };

  const renderTree = () => {
    return (
      <div style={{ width: TREE_W, position: "relative" }}>
        {/* SVG connectors */}
        <svg
          width={TREE_W}
          height={levels.length * (NODE_H + LEVEL_V_GAP)}
          style={{ position: "absolute", top: 0, left: 0 }}
        >
          {levels.slice(0, -1).map((parentLevel, li) => {
            const childLevel = levels[li + 1];
            const connDelay = 1.2 * fps + (li + 1) * 0.8 * fps;
            const connOpacity = interpolate(
              frame - connDelay,
              [0, 0.3 * fps],
              [0, 0.4],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );
            const parentY = li * (NODE_H + LEVEL_V_GAP) + NODE_H;
            const childY = (li + 1) * (NODE_H + LEVEL_V_GAP);

            return childLevel.map((_, ci) => {
              const cx = getNodeX(li + 1, ci);
              const lx = getNodeX(li, ci * 2);
              const rx = getNodeX(li, ci * 2 + 1);
              return (
                <g key={`${li}-${ci}`} opacity={connOpacity}>
                  <line
                    x1={lx} y1={parentY} x2={cx} y2={childY}
                    stroke={THEME.colors.nvidiaGreen}
                    strokeWidth={1.5}
                    strokeDasharray="4,3"
                  />
                  <line
                    x1={rx} y1={parentY} x2={cx} y2={childY}
                    stroke={THEME.colors.nvidiaGreen}
                    strokeWidth={1.5}
                    strokeDasharray="4,3"
                  />
                  {/* Plus sign at midpoint */}
                  <text
                    x={(lx + rx) / 2}
                    y={(parentY + childY) / 2 + 4}
                    textAnchor="middle"
                    fill={THEME.colors.accentOrange}
                    fontSize={14}
                    fontFamily={fontFamilyCode}
                    fontWeight={700}
                  >
                    +
                  </text>
                </g>
              );
            });
          })}
        </svg>

        {/* Nodes */}
        {levels.map((level, li) => {
          const levelDelay = 0.8 * fps + li * 0.8 * fps;
          const levelSpring = spring({
            frame: frame - levelDelay,
            fps,
            config: { damping: 200 },
          });
          const levelOpacity = interpolate(levelSpring, [0, 1], [0, 1]);

          const isFirst = li === 0;
          const isLast = li === levels.length - 1;

          return (
            <div
              key={li}
              style={{
                display: "flex",
                justifyContent: "center",
                position: "relative",
                height: NODE_H,
                marginBottom: LEVEL_V_GAP,
                opacity: levelOpacity,
                width: TREE_W,
              }}
            >
              {level.map((v, i) => {
                const x = getNodeX(li, i) - NODE_W / 2;
                return (
                  <div
                    key={i}
                    style={{
                      position: "absolute",
                      left: x,
                      width: NODE_W,
                      height: NODE_H,
                      borderRadius: 6,
                      backgroundColor: isLast
                        ? "rgba(118,185,0,0.25)"
                        : isFirst
                          ? "rgba(24,255,255,0.12)"
                          : "rgba(118,185,0,0.10)",
                      border: `2px solid ${isLast ? THEME.colors.nvidiaGreen : isFirst ? THEME.colors.accentCyan : THEME.colors.nvidiaGreen}70`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: isLast ? 20 : 16,
                      fontWeight: 700,
                      color: isLast ? THEME.colors.nvidiaGreen : isFirst ? THEME.colors.accentCyan : THEME.colors.nvidiaGreenLight,
                      fontFamily: fontFamilyCode,
                    }}
                  >
                    {v}
                  </div>
                );
              })}

              {/* Step label */}
              <div
                style={{
                  position: "absolute",
                  right: -50,
                  top: NODE_H / 2 - 8,
                  fontSize: 12,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyCode,
                  whiteSpace: "nowrap",
                }}
              >
                {li === 0 ? "" : `Step ${li}`}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // Complexity comparison
  const complexityDelay = 4 * fps;
  const complexitySpring = spring({
    frame: frame - complexityDelay,
    fps,
    config: { damping: 200 },
  });
  const complexityOpacity = interpolate(complexitySpring, [0, 1], [0, 1]);

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={4}
      slideNumber={3}
      totalSlides={18}
      leftWidth="55%"
      left={
        <div>
          <SlideTitle
            title="Parallel Reduction"
            subtitle="Sum N numbers: Sequential = O(N), Parallel = O(log N) steps"
          />

          {/* Tree diagram */}
          <div style={{ marginTop: 8 }}>
            {renderTree()}
          </div>

          {/* Complexity comparison */}
          <div
            style={{
              marginTop: 16,
              display: "flex",
              gap: 20,
              opacity: complexityOpacity,
            }}
          >
            <div
              style={{
                padding: "10px 18px",
                backgroundColor: "rgba(255,82,82,0.10)",
                borderRadius: 8,
                border: `1px solid ${THEME.colors.accentRed}40`,
              }}
            >
              <span style={{ fontSize: 14, color: THEME.colors.accentRed, fontFamily: fontFamilyCode, fontWeight: 700 }}>
                Sequential: O(N)
              </span>
              <div style={{ fontSize: 12, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginTop: 4 }}>
                7 additions for 8 elements
              </div>
            </div>
            <div
              style={{
                padding: "10px 18px",
                backgroundColor: "rgba(118,185,0,0.10)",
                borderRadius: 8,
                border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              }}
            >
              <span style={{ fontSize: 14, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyCode, fontWeight: 700 }}>
                Parallel: O(log N) steps
              </span>
              <div style={{ fontSize: 12, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginTop: 4 }}>
                3 steps for 8 elements
              </div>
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 80 }}>
          <FadeInText
            text="Why does this matter on GPU?"
            delay={1.5 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 16 }}
          />

          <BulletPoint
            index={0}
            delay={2 * fps}
            text="Sequential wastes 1000s of cores"
            subtext="Only 1 thread does work while the rest sit idle."
          />
          <BulletPoint
            index={1}
            delay={2 * fps}
            text="Parallel uses all threads"
            subtext="At step k, N/2^k threads each do one addition simultaneously."
          />
          <BulletPoint
            index={2}
            delay={2 * fps}
            text="Log2(1M) = only 20 steps"
            subtext="1 million elements reduced in just 20 parallel steps!"
            highlight
          />
          <BulletPoint
            index={3}
            delay={2 * fps}
            text="Key to ML: loss, gradients, norms"
            subtext="Every backward pass uses reduction for gradient accumulation."
          />
        </div>
      }
    />
  );
};
