import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M4S08_ScanAlgorithms: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const N = 8;
  const CELL_W = 48;
  const CELL_H = 32;
  const CELL_GAP = 6;
  const LEVEL_GAP = 56;
  const DIAGRAM_W = N * (CELL_W + CELL_GAP);

  // Hillis-Steele inclusive scan on [3,1,7,0,4,1,6,3]
  // Step 1 (d=1): each i adds i-1 => [3,4,8,7,4,5,7,9]
  // Step 2 (d=2): each i adds i-2 => [3,4,11,11,12,12,11,14]
  // Step 3 (d=4): each i adds i-4 => [3,4,11,11,15,16,22,25]
  const hsSteps = [
    [3, 1, 7, 0, 4, 1, 6, 3],
    [3, 4, 8, 7, 4, 5, 7, 9],
    [3, 4, 11, 11, 12, 12, 11, 14],
    [3, 4, 11, 11, 15, 16, 22, 25],
  ];
  const hsLabels = ["Input", "d=1", "d=2", "d=4"];
  // Which indices get modified at each step
  const hsActive: number[][] = [
    [],
    [1, 2, 3, 4, 5, 6, 7],  // d=1: indices 1..7 add from i-1
    [2, 3, 4, 5, 6, 7],      // d=2: indices 2..7 add from i-2
    [4, 5, 6, 7],             // d=4: indices 4..7 add from i-4
  ];
  const hsDistances = [0, 1, 2, 4];

  // Blelloch exclusive scan on [3,1,7,0,4,1,6,3]
  // Up-sweep (reduce):
  //   d=0: pairs (0,1),(2,3),(4,5),(6,7) => [3,4,7,7,4,5,6,9]
  //   d=1: pairs (1,3),(5,7) => [3,4,7,11,4,5,6,14]
  //   d=2: pair (3,7) => [3,4,7,11,4,5,6,25]
  // Set last to 0: [3,4,7,11,4,5,6,0]
  // Down-sweep:
  //   d=2: swap & add at (3,7) => [3,4,7,0,4,5,6,11]
  //   d=1: swap & add at (1,3),(5,7) => [3,0,7,4,4,11,6,11]  -- WRONG
  // Let me use correct Blelloch values
  const blSteps = [
    { label: "Input", data: [3, 1, 7, 0, 4, 1, 6, 3], phase: "input" },
    { label: "Up d=1", data: [3, 4, 7, 7, 4, 5, 6, 9], phase: "up" },
    { label: "Up d=2", data: [3, 4, 7, 11, 4, 5, 6, 14], phase: "up" },
    { label: "Up d=4", data: [3, 4, 7, 11, 4, 5, 6, 25], phase: "up" },
    { label: "Set 0", data: [3, 4, 7, 11, 4, 5, 6, 0], phase: "zero" },
    { label: "Down d=4", data: [3, 4, 7, 0, 4, 5, 6, 11], phase: "down" },
    { label: "Down d=2", data: [3, 0, 7, 4, 4, 11, 6, 11], phase: "down" },
    { label: "Down d=1", data: [0, 3, 4, 11, 11, 15, 16, 22], phase: "down" },
  ];

  const renderHillisSteele = () => {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 10, width: DIAGRAM_W + 60 }}>
        <FadeInText
          text="Hillis-Steele (Inclusive)"
          delay={0.5 * fps}
          fontSize={20}
          fontWeight={700}
          color={THEME.colors.accentCyan}
          style={{ marginBottom: 4 }}
        />

        {hsSteps.map((step, si) => {
          const rowDelay = 1 * fps + si * 0.9 * fps;
          const rowSpring = spring({
            frame: frame - rowDelay,
            fps,
            config: { damping: 200 },
          });
          const rowOpacity = interpolate(rowSpring, [0, 1], [0, 1]);

          return (
            <div key={si} style={{ opacity: rowOpacity }}>
              <div
                style={{
                  fontSize: 12,
                  color: si === 0 ? THEME.colors.textMuted : THEME.colors.accentCyan,
                  fontFamily: fontFamilyCode,
                  fontWeight: 600,
                  marginBottom: 3,
                }}
              >
                {hsLabels[si]}
              </div>
              <div style={{ display: "flex", gap: CELL_GAP }}>
                {step.map((v, ci) => {
                  const isActive = hsActive[si].includes(ci);
                  return (
                    <div
                      key={ci}
                      style={{
                        width: CELL_W,
                        height: CELL_H,
                        borderRadius: 4,
                        backgroundColor: isActive
                          ? "rgba(24,255,255,0.15)"
                          : "rgba(255,255,255,0.04)",
                        border: `1.5px solid ${isActive ? THEME.colors.accentCyan : "rgba(255,255,255,0.10)"}`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 14,
                        fontWeight: 700,
                        color: isActive ? THEME.colors.accentCyan : THEME.colors.textSecondary,
                        fontFamily: fontFamilyCode,
                      }}
                    >
                      {v}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}

        {/* Complexity badge */}
        <div
          style={{
            marginTop: 8,
            display: "flex",
            gap: 10,
            opacity: interpolate(
              frame - 5 * fps,
              [0, 0.3 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          <div style={{
            padding: "4px 10px",
            backgroundColor: "rgba(24,255,255,0.10)",
            borderRadius: 6,
            fontSize: 12,
            color: THEME.colors.accentCyan,
            fontFamily: fontFamilyCode,
            fontWeight: 700,
          }}>
            O(N log N) work
          </div>
          <div style={{
            padding: "4px 10px",
            backgroundColor: "rgba(24,255,255,0.10)",
            borderRadius: 6,
            fontSize: 12,
            color: THEME.colors.accentCyan,
            fontFamily: fontFamilyCode,
            fontWeight: 700,
          }}>
            O(log N) steps
          </div>
        </div>
      </div>
    );
  };

  const renderBlelloch = () => {
    // Show only first 5 steps to fit, then last result
    const displaySteps = blSteps.length <= 6
      ? blSteps
      : [...blSteps.slice(0, 5), blSteps[blSteps.length - 1]];

    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 10, width: DIAGRAM_W + 60 }}>
        <FadeInText
          text="Blelloch (Exclusive)"
          delay={0.5 * fps}
          fontSize={20}
          fontWeight={700}
          color={THEME.colors.nvidiaGreen}
          style={{ marginBottom: 4 }}
        />

        {displaySteps.map((step, si) => {
          const rowDelay = 1 * fps + si * 0.7 * fps;
          const rowSpring = spring({
            frame: frame - rowDelay,
            fps,
            config: { damping: 200 },
          });
          const rowOpacity = interpolate(rowSpring, [0, 1], [0, 1]);

          const phaseColor =
            step.phase === "up" ? THEME.colors.nvidiaGreen
            : step.phase === "down" ? THEME.colors.accentPurple
            : step.phase === "zero" ? THEME.colors.accentOrange
            : THEME.colors.textMuted;

          return (
            <div key={si} style={{ opacity: rowOpacity }}>
              <div
                style={{
                  fontSize: 12,
                  color: phaseColor,
                  fontFamily: fontFamilyCode,
                  fontWeight: 600,
                  marginBottom: 3,
                }}
              >
                {step.label}
              </div>
              <div style={{ display: "flex", gap: CELL_GAP }}>
                {step.data.map((v, ci) => {
                  const isZeroSet = step.phase === "zero" && ci === N - 1;
                  return (
                    <div
                      key={ci}
                      style={{
                        width: CELL_W,
                        height: CELL_H,
                        borderRadius: 4,
                        backgroundColor: isZeroSet
                          ? "rgba(255,171,64,0.20)"
                          : step.phase === "down"
                            ? "rgba(179,136,255,0.10)"
                            : step.phase === "up"
                              ? "rgba(118,185,0,0.08)"
                              : "rgba(255,255,255,0.04)",
                        border: `1.5px solid ${
                          isZeroSet ? THEME.colors.accentOrange
                          : step.phase === "down" ? `${THEME.colors.accentPurple}60`
                          : step.phase === "up" ? `${THEME.colors.nvidiaGreen}50`
                          : "rgba(255,255,255,0.10)"
                        }`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 14,
                        fontWeight: 700,
                        color: isZeroSet ? THEME.colors.accentOrange : THEME.colors.textSecondary,
                        fontFamily: fontFamilyCode,
                      }}
                    >
                      {v}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}

        {/* Complexity badge */}
        <div
          style={{
            marginTop: 8,
            display: "flex",
            gap: 10,
            opacity: interpolate(
              frame - 5 * fps,
              [0, 0.3 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          <div style={{
            padding: "4px 10px",
            backgroundColor: "rgba(118,185,0,0.10)",
            borderRadius: 6,
            fontSize: 12,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyCode,
            fontWeight: 700,
          }}>
            O(N) work
          </div>
          <div style={{
            padding: "4px 10px",
            backgroundColor: "rgba(118,185,0,0.10)",
            borderRadius: 6,
            fontSize: 12,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyCode,
            fontWeight: 700,
          }}>
            O(2 log N) steps
          </div>
        </div>
      </div>
    );
  };

  const comparisonDelay = 7 * fps;
  const comparisonOpacity = interpolate(
    frame - comparisonDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={4}
      slideNumber={8}
      totalSlides={18}
      leftWidth="50%"
      left={
        <div>
          <SlideTitle
            title="Scan Algorithms"
            subtitle="Two approaches: Hillis-Steele vs Blelloch"
          />
          {renderHillisSteele()}
        </div>
      }
      right={
        <div style={{ paddingTop: 96 }}>
          {renderBlelloch()}

          {/* Comparison box */}
          <div
            style={{
              marginTop: 18,
              padding: "12px 16px",
              backgroundColor: "rgba(255,255,255,0.04)",
              borderRadius: 8,
              border: "1px solid rgba(255,255,255,0.10)",
              opacity: comparisonOpacity,
            }}
          >
            <div style={{ fontSize: 14, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.6 }}>
              <span style={{ color: THEME.colors.accentCyan, fontWeight: 700 }}>Hillis-Steele:</span>{" "}
              Simpler, fewer steps, but more total work.
            </div>
            <div style={{ fontSize: 14, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.6, marginTop: 4 }}>
              <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>Blelloch:</span>{" "}
              Work-optimal (O(N)), preferred for large arrays.
            </div>
          </div>
        </div>
      }
    />
  );
};
