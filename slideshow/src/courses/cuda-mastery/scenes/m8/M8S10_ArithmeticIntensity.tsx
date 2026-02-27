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

type TableRow = {
  approach: string;
  ai: string;
  bound: string;
  highlight: boolean;
};

const tableRows: TableRow[] = [
  { approach: "Naive", ai: "0.25", bound: "Memory", highlight: false },
  { approach: "Tiled (32)", ai: "8", bound: "Memory", highlight: false },
  { approach: "Register (8x8)", ai: "64", bound: "Compute!", highlight: true },
  { approach: "+ Vectorized", ai: "64", bound: "Compute", highlight: false },
];

type FormulaLine = {
  label: string;
  formula: string;
  color: string;
};

const formulas: FormulaLine[] = [
  {
    label: "Naive",
    formula: "2 FLOPs / 8 bytes = 0.25",
    color: THEME.colors.accentRed,
  },
  {
    label: "Tiled",
    formula: "2 x TILE FLOPs / 8 bytes = 8",
    color: THEME.colors.accentYellow,
  },
  {
    label: "Register",
    formula: "2 x TM x TN FLOPs / (TM+TN) x 4 bytes = 64",
    color: THEME.colors.nvidiaGreen,
  },
  {
    label: "Ridge point (A100)",
    formula: "~ 9.3 FLOPs/byte",
    color: THEME.colors.accentCyan,
  },
];

export const M8S10_ArithmeticIntensity: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const bottomOpacity = interpolate(
    frame - 10 * fps,
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
        <div style={{ width: 780 }}>
          <SlideTitle title="Arithmetic Intensity Analysis" />

          {/* Table */}
          <div
            style={{
              backgroundColor: "rgba(13,17,23,0.7)",
              borderRadius: 10,
              padding: "16px 20px",
              border: "1px solid rgba(255,255,255,0.08)",
              width: 700,
            }}
          >
            {/* Header */}
            <div
              style={{
                display: "flex",
                paddingBottom: 10,
                borderBottom: "1px solid rgba(255,255,255,0.1)",
                marginBottom: 8,
              }}
            >
              {["Approach", "AI (FLOPs/byte)", "Bound"].map((h, i) => (
                <div
                  key={h}
                  style={{
                    width: i === 0 ? 240 : i === 1 ? 240 : 220,
                    fontSize: 14,
                    fontWeight: 700,
                    color: THEME.colors.textMuted,
                    fontFamily: fontFamilyBody,
                    textAlign: i === 0 ? "left" : "center",
                  }}
                >
                  {h}
                </div>
              ))}
            </div>

            {/* Rows */}
            {tableRows.map((row, i) => {
              const rowDelay = 1.5 * fps + i * 1.2 * fps;
              const rowSpring = spring({
                frame: frame - rowDelay,
                fps,
                config: { damping: 200 },
              });
              const rowOpacity = interpolate(rowSpring, [0, 1], [0, 1]);
              const rowX = interpolate(rowSpring, [0, 1], [-15, 0]);

              return (
                <div
                  key={row.approach}
                  style={{
                    display: "flex",
                    padding: "10px 0",
                    borderBottom:
                      i < tableRows.length - 1
                        ? "1px solid rgba(255,255,255,0.05)"
                        : "none",
                    opacity: rowOpacity,
                    transform: `translateX(${rowX}px)`,
                    backgroundColor: row.highlight
                      ? "rgba(118,185,0,0.1)"
                      : "transparent",
                    borderRadius: row.highlight ? 6 : 0,
                    paddingLeft: row.highlight ? 8 : 0,
                    paddingRight: row.highlight ? 8 : 0,
                  }}
                >
                  <div
                    style={{
                      width: 240,
                      fontSize: 17,
                      color: row.highlight
                        ? THEME.colors.nvidiaGreen
                        : THEME.colors.textPrimary,
                      fontFamily: fontFamilyCode,
                      fontWeight: row.highlight ? 700 : 400,
                    }}
                  >
                    {row.approach}
                  </div>
                  <div
                    style={{
                      width: 240,
                      fontSize: 17,
                      color: row.highlight
                        ? THEME.colors.nvidiaGreen
                        : THEME.colors.accentYellow,
                      fontFamily: fontFamilyCode,
                      fontWeight: 700,
                      textAlign: "center",
                    }}
                  >
                    {row.ai}
                  </div>
                  <div
                    style={{
                      width: 220,
                      fontSize: 17,
                      color: row.highlight
                        ? THEME.colors.nvidiaGreen
                        : THEME.colors.textSecondary,
                      fontFamily: fontFamilyCode,
                      fontWeight: row.highlight ? 700 : 400,
                      textAlign: "center",
                    }}
                  >
                    {row.bound}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      }
      right={
        <div style={{ width: 560, marginTop: 20 }}>
          <FadeInText
            text="Formula Breakdown"
            delay={2 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 20 }}
          />

          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {formulas.map((f, i) => {
              const fDelay = 3 * fps + i * 1.5 * fps;
              const fSpring = spring({
                frame: frame - fDelay,
                fps,
                config: { damping: 200 },
              });
              const fOpacity = interpolate(fSpring, [0, 1], [0, 1]);

              return (
                <div
                  key={f.label}
                  style={{
                    padding: "10px 16px",
                    backgroundColor: `${f.color}10`,
                    borderLeft: `3px solid ${f.color}`,
                    borderRadius: 6,
                    opacity: fOpacity,
                  }}
                >
                  <div
                    style={{
                      fontSize: 14,
                      color: f.color,
                      fontFamily: fontFamilyBody,
                      fontWeight: 700,
                      marginBottom: 4,
                    }}
                  >
                    {f.label}:
                  </div>
                  <div
                    style={{
                      fontSize: 16,
                      color: THEME.colors.textPrimary,
                      fontFamily: fontFamilyCode,
                    }}
                  >
                    {f.formula}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Bottom insight */}
          <div
            style={{
              marginTop: 24,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.nvidiaGreen}30`,
              opacity: bottomOpacity,
            }}
          >
            <div
              style={{
                fontSize: 16,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
                lineHeight: 1.5,
              }}
            >
              Goal: get arithmetic intensity above the ridge point
            </div>
            <div
              style={{
                fontSize: 14,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                marginTop: 4,
                lineHeight: 1.4,
              }}
            >
              Above 9.3 FLOPs/byte = compute-bound territory
            </div>
          </div>
        </div>
      }
    />
  );
};
