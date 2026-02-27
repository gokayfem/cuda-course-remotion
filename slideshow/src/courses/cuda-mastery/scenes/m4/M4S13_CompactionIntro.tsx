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
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

const INPUT = [
  { value: "A", keep: true },
  { value: "B", keep: false },
  { value: "C", keep: true },
  { value: "D", keep: true },
  { value: "E", keep: false },
  { value: "F", keep: false },
  { value: "G", keep: true },
  { value: "H", keep: false },
];

const PREDICATES = INPUT.map((item) => (item.keep ? 1 : 0));

const computeExclusiveScan = (arr: readonly number[]): number[] => {
  const result: number[] = [0];
  for (let i = 1; i < arr.length; i++) {
    result.push(result[i - 1] + arr[i - 1]);
  }
  return result;
};

const SCAN = computeExclusiveScan(PREDICATES);
const OUTPUT = INPUT.filter((item) => item.keep);

const CELL_W = 80;
const CELL_H = 48;
const GAP = 6;

export const M4S13_CompactionIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Step timings
  const step1Delay = 1 * fps;   // Input array
  const step2Delay = 2.5 * fps; // Predicates
  const step3Delay = 4.5 * fps; // Exclusive scan
  const step4Delay = 6.5 * fps; // Output

  const renderRow = (
    label: string,
    values: readonly (string | number)[],
    colors: readonly string[],
    delay: number,
    sublabel?: string
  ) => {
    const rowSpring = spring({
      frame: frame - delay,
      fps,
      config: { damping: 200 },
    });
    const rowOpacity = interpolate(rowSpring, [0, 1], [0, 1]);
    const rowY = interpolate(rowSpring, [0, 1], [15, 0]);

    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 20,
          opacity: rowOpacity,
          transform: `translateY(${rowY}px)`,
          marginBottom: 14,
        }}
      >
        <div style={{ width: 160, flexShrink: 0 }}>
          <div
            style={{
              fontSize: 16,
              fontWeight: 700,
              color: THEME.colors.textPrimary,
              fontFamily: fontFamilyBody,
            }}
          >
            {label}
          </div>
          {sublabel && (
            <div
              style={{
                fontSize: 12,
                color: THEME.colors.textMuted,
                fontFamily: fontFamilyBody,
                marginTop: 2,
              }}
            >
              {sublabel}
            </div>
          )}
        </div>

        <div style={{ display: "flex", gap: GAP }}>
          {values.map((v, i) => (
            <div
              key={i}
              style={{
                width: CELL_W,
                height: CELL_H,
                borderRadius: 8,
                backgroundColor: `${colors[i]}18`,
                border: `2px solid ${colors[i]}60`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 18,
                fontWeight: 700,
                color: colors[i],
                fontFamily: fontFamilyCode,
              }}
            >
              {v}
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Colors for input cells
  const inputColors = INPUT.map((item) =>
    item.keep ? THEME.colors.nvidiaGreen : THEME.colors.textMuted
  );

  const predColors = PREDICATES.map((p) =>
    p === 1 ? THEME.colors.nvidiaGreen : THEME.colors.accentRed
  );

  const scanColors = SCAN.map(() => THEME.colors.accentCyan);

  // Output row uses the scatter animation
  const outputDelay = step4Delay;
  const outputSpring = spring({
    frame: frame - outputDelay,
    fps,
    config: { damping: 200 },
  });
  const outputOpacity = interpolate(outputSpring, [0, 1], [0, 1]);
  const outputY = interpolate(outputSpring, [0, 1], [15, 0]);

  const insightOpacity = interpolate(
    frame - 8.5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={4} slideNumber={13} totalSlides={18}>
      <SlideTitle
        title="Stream Compaction: The Problem"
        subtitle="Filter elements based on a predicate -- keep only the selected items"
      />

      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 4 }}>
        {/* Step 1: Input */}
        {renderRow(
          "1. Input",
          INPUT.map((item) => item.value),
          inputColors,
          step1Delay,
          "highlighted = keep"
        )}

        {/* Step 2: Predicates */}
        {renderRow(
          "2. Predicate",
          PREDICATES,
          predColors,
          step2Delay,
          "1 = keep, 0 = discard"
        )}

        {/* Step 3: Exclusive scan */}
        {renderRow(
          "3. Exclusive Scan",
          SCAN,
          scanColors,
          step3Delay,
          "prefix sum of predicates"
        )}

        {/* Step 4: Output */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 20,
            opacity: outputOpacity,
            transform: `translateY(${outputY}px)`,
            marginBottom: 14,
          }}
        >
          <div style={{ width: 160, flexShrink: 0 }}>
            <div
              style={{
                fontSize: 16,
                fontWeight: 700,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
              }}
            >
              4. Output
            </div>
            <div
              style={{
                fontSize: 12,
                color: THEME.colors.textMuted,
                fontFamily: fontFamilyBody,
                marginTop: 2,
              }}
            >
              compacted result
            </div>
          </div>

          <div style={{ display: "flex", gap: GAP }}>
            {OUTPUT.map((item, i) => (
              <div
                key={i}
                style={{
                  width: CELL_W,
                  height: CELL_H,
                  borderRadius: 8,
                  backgroundColor: `${THEME.colors.nvidiaGreen}25`,
                  border: `2px solid ${THEME.colors.nvidiaGreen}`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 18,
                  fontWeight: 700,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyCode,
                }}
              >
                {item.value}
              </div>
            ))}
          </div>
        </div>

        {/* Arrow annotations showing how scan values become write addresses */}
        <div
          style={{
            marginTop: 8,
            padding: "14px 20px",
            backgroundColor: "rgba(79,195,247,0.08)",
            borderRadius: 10,
            border: `1px solid ${THEME.colors.accentCyan}30`,
            opacity: insightOpacity,
          }}
        >
          <div
            style={{
              fontSize: 16,
              color: THEME.colors.textPrimary,
              fontFamily: fontFamilyBody,
              lineHeight: 1.6,
            }}
          >
            <span style={{ color: THEME.colors.accentCyan, fontWeight: 700 }}>
              Key insight:
            </span>{" "}
            The exclusive scan gives each selected element a{" "}
            <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>
              unique output index
            </span>
            . Element "A" (scan=0) writes to output[0], "C" (scan=1) writes to output[1],
            "D" (scan=2) writes to output[2], "G" (scan=3) writes to output[3].
          </div>
        </div>

        {/* Pipeline summary */}
        <div
          style={{
            marginTop: 12,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 14,
            opacity: insightOpacity,
          }}
        >
          {["Predicate", "Scan", "Scatter"].map((step, i) => (
            <React.Fragment key={i}>
              <div
                style={{
                  padding: "8px 20px",
                  backgroundColor: "rgba(118,185,0,0.10)",
                  borderRadius: 8,
                  border: `1px solid ${THEME.colors.nvidiaGreen}40`,
                  fontSize: 16,
                  fontWeight: 700,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                }}
              >
                {step}
              </div>
              {i < 2 && (
                <span
                  style={{
                    fontSize: 20,
                    color: THEME.colors.textMuted,
                  }}
                >
                  {"\u2192"}
                </span>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
    </SlideLayout>
  );
};
