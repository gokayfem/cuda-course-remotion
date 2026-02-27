import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

const INPUT_VALUES = [3, 1, 7, 0, 5, 3, 2, 7, 1, 4, 6, 3, 0, 5, 2, 6];
const NUM_BINS = 8;

const BIN_COLORS = [
  THEME.colors.accentBlue,
  THEME.colors.accentCyan,
  THEME.colors.nvidiaGreen,
  THEME.colors.accentYellow,
  THEME.colors.accentOrange,
  THEME.colors.accentPink,
  THEME.colors.accentPurple,
  THEME.colors.accentRed,
];

const computeHistogram = (values: readonly number[]): number[] => {
  const bins = Array.from({ length: NUM_BINS }, () => 0);
  values.forEach((v) => {
    if (v >= 0 && v < NUM_BINS) {
      bins[v] = bins[v] + 1;
    }
  });
  return bins;
};

export const M4S10_HistogramIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const histogram = computeHistogram(INPUT_VALUES);
  const maxCount = Math.max(...histogram);

  // How many input values have been "counted" so far
  const countProgress = interpolate(
    frame,
    [2 * fps, 6 * fps],
    [0, INPUT_VALUES.length],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const countedSoFar = Math.floor(countProgress);

  // Partial histogram based on counted values
  const partialHistogram = computeHistogram(INPUT_VALUES.slice(0, countedSoFar));

  const raceOpacity = interpolate(
    frame - 7.5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const BAR_MAX_HEIGHT = 180;
  const BAR_WIDTH = 64;
  const CELL_SIZE = 44;

  return (
    <SlideLayout variant="gradient" moduleNumber={4} slideNumber={10} totalSlides={18}>
      <SlideTitle
        title="Histogram: The Problem"
        subtitle="Count occurrences of each value in an array"
      />

      <div style={{ display: "flex", gap: 48, flex: 1 }}>
        {/* Left: Input array + histogram bars */}
        <div style={{ flex: 1.3 }}>
          {/* Input array label */}
          <div
            style={{
              fontSize: 16,
              color: THEME.colors.textSecondary,
              fontFamily: fontFamilyBody,
              fontWeight: 600,
              marginBottom: 8,
            }}
          >
            Input Array
          </div>

          {/* Input array cells */}
          <div style={{ display: "flex", gap: 3, flexWrap: "wrap", width: 8 * (CELL_SIZE + 3), marginBottom: 28 }}>
            {INPUT_VALUES.map((val, i) => {
              const isCounted = i < countedSoFar;
              const cellSpring = spring({
                frame: frame - 0.8 * fps - i * 0.06 * fps,
                fps,
                config: { damping: 200 },
              });
              const cellOpacity = interpolate(cellSpring, [0, 1], [0, 1]);

              return (
                <div
                  key={i}
                  style={{
                    width: CELL_SIZE,
                    height: CELL_SIZE,
                    borderRadius: 6,
                    backgroundColor: isCounted
                      ? `${BIN_COLORS[val]}30`
                      : "rgba(255,255,255,0.06)",
                    border: `2px solid ${isCounted ? BIN_COLORS[val] : "rgba(255,255,255,0.1)"}`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 18,
                    fontWeight: 700,
                    color: isCounted ? BIN_COLORS[val] : THEME.colors.textMuted,
                    fontFamily: fontFamilyCode,
                    opacity: cellOpacity,
                    transition: "background-color 0.2s, border-color 0.2s",
                  }}
                >
                  {val}
                </div>
              );
            })}
          </div>

          {/* Histogram bars */}
          <div
            style={{
              fontSize: 16,
              color: THEME.colors.textSecondary,
              fontFamily: fontFamilyBody,
              fontWeight: 600,
              marginBottom: 12,
            }}
          >
            Histogram (bin counts)
          </div>

          <div style={{ display: "flex", alignItems: "flex-end", gap: 12, height: BAR_MAX_HEIGHT + 40 }}>
            {partialHistogram.map((count, bin) => {
              const barHeight = maxCount > 0
                ? (count / maxCount) * BAR_MAX_HEIGHT
                : 0;

              const barSpring = spring({
                frame: frame - 2 * fps,
                fps,
                config: { damping: 100 },
              });
              const barScale = interpolate(barSpring, [0, 1], [0, 1]);

              return (
                <div
                  key={bin}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    gap: 4,
                  }}
                >
                  {/* Count label */}
                  <div
                    style={{
                      fontSize: 14,
                      fontWeight: 700,
                      color: BIN_COLORS[bin],
                      fontFamily: fontFamilyCode,
                      minHeight: 20,
                    }}
                  >
                    {count > 0 ? count : ""}
                  </div>

                  {/* Bar */}
                  <div
                    style={{
                      width: BAR_WIDTH,
                      height: barHeight * barScale,
                      backgroundColor: `${BIN_COLORS[bin]}40`,
                      border: `2px solid ${BIN_COLORS[bin]}`,
                      borderRadius: 4,
                      minHeight: count > 0 ? 8 : 0,
                    }}
                  />

                  {/* Bin label */}
                  <div
                    style={{
                      fontSize: 14,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyCode,
                      fontWeight: 600,
                    }}
                  >
                    {bin}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Explanation */}
        <div style={{ flex: 0.7 }}>
          <BulletPoint
            index={0}
            delay={1 * fps}
            text="What is a histogram?"
            subtext="Count how many times each value appears. Given N values and B bins, output an array of B counts."
          />
          <BulletPoint
            index={1}
            delay={1 * fps}
            text="Embarrassingly parallel?"
            subtext="Each thread reads one element and increments a bin. But multiple threads may target the same bin simultaneously."
            highlight
          />
          <BulletPoint
            index={2}
            delay={1 * fps}
            text="The race condition"
            subtext="Read-modify-write on the same bin: two threads read count=3, both write count=4. One increment is lost."
          />
          <BulletPoint
            index={3}
            delay={1 * fps}
            text="Solution: atomics"
            subtext="atomicAdd ensures each increment is applied correctly, but comes with a performance cost."
          />
        </div>
      </div>

      {/* Race condition warning box */}
      <div
        style={{
          marginTop: 12,
          padding: "14px 24px",
          backgroundColor: "rgba(255,82,82,0.10)",
          borderRadius: 10,
          border: `2px solid ${THEME.colors.accentRed}60`,
          opacity: raceOpacity,
          display: "flex",
          alignItems: "center",
          gap: 16,
        }}
      >
        <span style={{ fontSize: 24, flexShrink: 0 }}>{"⚠"}</span>
        <span
          style={{
            fontSize: 18,
            color: THEME.colors.textPrimary,
            fontFamily: fontFamilyBody,
            fontWeight: 600,
          }}
        >
          Race Condition:{" "}
          <span style={{ color: THEME.colors.accentRed }}>
            Without atomics, concurrent writes to the same bin lose updates
          </span>
        </span>
      </div>
    </SlideLayout>
  );
};
