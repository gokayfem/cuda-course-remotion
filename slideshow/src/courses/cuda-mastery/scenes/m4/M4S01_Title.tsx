import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
  AbsoluteFill,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideBackground } from "../../../../components/SlideBackground";
import { fontFamilyHeading, fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M4S01_Title: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const barWidth = interpolate(frame, [0, 1 * fps], [0, 400], {
    extrapolateRight: "clamp",
  });

  const titleSpring = spring({
    frame,
    fps,
    config: { damping: 200 },
    delay: 0.3 * fps,
  });
  const titleOpacity = interpolate(titleSpring, [0, 1], [0, 1]);
  const titleY = interpolate(titleSpring, [0, 1], [40, 0]);

  const subtitleOpacity = interpolate(
    frame,
    [1 * fps, 1.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const moduleOpacity = interpolate(
    frame,
    [1.5 * fps, 2 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Tree reduction data: 8 values reducing to 1 in 3 levels
  const values = [
    [4, 7, 2, 5, 1, 8, 3, 6], // level 0: 8 values
    [11, 7, 9, 9],              // level 1: 4 values
    [18, 18],                    // level 2: 2 values
    [36],                        // level 3: 1 value
  ];

  const NODE_W = 52;
  const NODE_H = 36;
  const LEVEL_GAP = 60;
  const TREE_WIDTH = 560;

  const renderLevel = (level: number) => {
    const vals = values[level];
    const count = vals.length;
    const totalWidth = count * NODE_W + (count - 1) * ((TREE_WIDTH - count * NODE_W) / Math.max(count - 1, 1));
    const spacing = count > 1 ? (TREE_WIDTH - count * NODE_W) / (count - 1) : 0;

    const levelDelay = 1.2 * fps + level * 0.6 * fps;
    const levelSpring = spring({
      frame: frame - levelDelay,
      fps,
      config: { damping: 200 },
    });
    const levelOpacity = interpolate(levelSpring, [0, 1], [0, 1]);
    const levelScale = interpolate(levelSpring, [0, 1], [0.8, 1]);

    const isLast = level === values.length - 1;

    return (
      <div
        key={level}
        style={{
          display: "flex",
          justifyContent: "center",
          gap: spacing,
          opacity: levelOpacity,
          transform: `scale(${levelScale})`,
          width: TREE_WIDTH,
        }}
      >
        {vals.map((v, i) => (
          <div
            key={i}
            style={{
              width: NODE_W,
              height: NODE_H,
              borderRadius: 6,
              backgroundColor: isLast
                ? "rgba(118,185,0,0.25)"
                : level === 0
                  ? "rgba(79,195,247,0.15)"
                  : "rgba(118,185,0,0.12)",
              border: `2px solid ${isLast ? THEME.colors.nvidiaGreen : level === 0 ? THEME.colors.accentCyan : THEME.colors.nvidiaGreen}80`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: isLast ? 20 : 16,
              fontWeight: 700,
              color: isLast ? THEME.colors.nvidiaGreen : level === 0 ? THEME.colors.accentCyan : THEME.colors.nvidiaGreenLight,
              fontFamily: fontFamilyCode,
            }}
          >
            {v}
          </div>
        ))}
      </div>
    );
  };

  // Render connecting lines between levels
  const renderConnectors = (level: number) => {
    const parentCount = values[level].length;
    const childCount = values[level + 1].length;
    const connDelay = 1.5 * fps + (level + 1) * 0.6 * fps;
    const connOpacity = interpolate(
      frame - connDelay,
      [0, 0.3 * fps],
      [0, 0.5],
      { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
    );

    return (
      <div
        key={`conn-${level}`}
        style={{
          height: LEVEL_GAP - NODE_H,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          opacity: connOpacity,
          width: TREE_WIDTH,
        }}
      >
        <svg width={TREE_WIDTH} height={LEVEL_GAP - NODE_H} style={{ overflow: "visible" }}>
          {Array.from({ length: childCount }).map((_, ci) => {
            const leftParent = ci * 2;
            const rightParent = ci * 2 + 1;
            const parentSpacing = (idx: number) => {
              const total = parentCount;
              if (total === 1) return TREE_WIDTH / 2;
              return (NODE_W / 2) + idx * ((TREE_WIDTH - NODE_W) / (total - 1));
            };
            const childSpacing = (idx: number) => {
              const total = childCount;
              if (total === 1) return TREE_WIDTH / 2;
              return (NODE_W / 2) + idx * ((TREE_WIDTH - NODE_W) / (total - 1));
            };
            const cx = childSpacing(ci);
            const lx = parentSpacing(leftParent);
            const rx = parentSpacing(rightParent);

            return (
              <React.Fragment key={ci}>
                <line
                  x1={lx} y1={0} x2={cx} y2={LEVEL_GAP - NODE_H}
                  stroke={THEME.colors.nvidiaGreen}
                  strokeWidth={1.5}
                  strokeDasharray="4,3"
                />
                <line
                  x1={rx} y1={0} x2={cx} y2={LEVEL_GAP - NODE_H}
                  stroke={THEME.colors.nvidiaGreen}
                  strokeWidth={1.5}
                  strokeDasharray="4,3"
                />
              </React.Fragment>
            );
          })}
        </svg>
      </div>
    );
  };

  // Step labels
  const stepLabels = ["Input (8 values)", "Step 1: pair-sum", "Step 2: pair-sum", "Result"];

  return (
    <AbsoluteFill>
      <SlideBackground variant="accent" />

      {/* Tree reduction visual on right */}
      <div
        style={{
          position: "absolute",
          right: 60,
          top: 140,
          width: 600,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        {values.map((_, level) => (
          <React.Fragment key={level}>
            <div style={{ display: "flex", alignItems: "center", gap: 14, width: TREE_WIDTH }}>
              {/* Step label */}
              <div
                style={{
                  position: "absolute",
                  left: -10,
                  fontSize: 12,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyBody,
                  width: 0,
                  opacity: interpolate(
                    frame - (1.2 * fps + level * 0.6 * fps),
                    [0, 0.3 * fps],
                    [0, 1],
                    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                  ),
                }}
              >
              </div>
              {renderLevel(level)}
            </div>
            {level < values.length - 1 && renderConnectors(level)}
          </React.Fragment>
        ))}

        {/* Label below tree */}
        <div
          style={{
            marginTop: 20,
            fontSize: 16,
            color: THEME.colors.accentCyan,
            fontFamily: fontFamilyBody,
            fontWeight: 600,
            textAlign: "center",
            opacity: interpolate(
              frame - 3.5 * fps,
              [0, 0.5 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          Parallel Reduction: O(log N) steps
        </div>
      </div>

      {/* Main content on left */}
      <div
        style={{
          position: "absolute",
          left: 100,
          top: "50%",
          transform: "translateY(-50%)",
          maxWidth: 800,
        }}
      >
        <div
          style={{
            width: barWidth,
            height: 6,
            backgroundColor: THEME.colors.nvidiaGreen,
            borderRadius: 3,
            marginBottom: 32,
          }}
        />

        <h1
          style={{
            fontSize: 72,
            fontWeight: 900,
            color: THEME.colors.textPrimary,
            fontFamily: fontFamilyHeading,
            margin: 0,
            opacity: titleOpacity,
            transform: `translateY(${titleY}px)`,
            lineHeight: 1.1,
            letterSpacing: "-2px",
          }}
        >
          Parallel{" "}
          <span style={{ color: THEME.colors.nvidiaGreen }}>Patterns</span>
        </h1>

        <p
          style={{
            fontSize: 28,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
            margin: 0,
            marginTop: 24,
            opacity: subtitleOpacity,
            fontWeight: 400,
          }}
        >
          Reduction, Scan, Histogram & Compaction
        </p>

        <div
          style={{
            marginTop: 48,
            display: "flex",
            gap: 16,
            alignItems: "center",
            opacity: moduleOpacity,
          }}
        >
          <div
            style={{
              padding: "12px 28px",
              backgroundColor: "rgba(118,185,0,0.15)",
              border: `2px solid ${THEME.colors.nvidiaGreen}`,
              borderRadius: 30,
              fontSize: 22,
              color: THEME.colors.nvidiaGreen,
              fontFamily: fontFamilyBody,
              fontWeight: 700,
            }}
          >
            Module 4
          </div>
          <span
            style={{
              fontSize: 22,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
            }}
          >
            The building blocks of GPU computing
          </span>
        </div>
      </div>

      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          height: 6,
          background: `linear-gradient(90deg, ${THEME.colors.nvidiaGreen}, ${THEME.colors.accentCyan}, ${THEME.colors.accentBlue})`,
        }}
      />
    </AbsoluteFill>
  );
};
