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

interface TypeRow {
  readonly name: string;
  readonly bits: string;
  readonly range: string;
  readonly useCase: string;
  readonly highlight: boolean;
}

const DATA_TYPES: readonly TypeRow[] = [
  { name: "FP32", bits: "32", range: "\u00b13.4e38", useCase: "Master weights", highlight: false },
  { name: "FP16", bits: "16", range: "\u00b165504", useCase: "Forward/backward", highlight: true },
  { name: "BF16", bits: "16", range: "\u00b13.4e38", useCase: "Training (better range)", highlight: true },
  { name: "TF32", bits: "19", range: "\u00b13.4e38", useCase: "Auto on Ampere", highlight: false },
  { name: "FP8", bits: "8", range: "\u00b1448", useCase: "Hopper inference", highlight: false },
];

const COLUMNS = ["Type", "Bits", "Range", "Use Case"] as const;
const COL_WIDTHS = [80, 60, 120, 200] as const;

export const M10S04_MixedPrecisionIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const headerDelay = 0.8 * fps;
  const headerOpacity = interpolate(
    frame - headerDelay,
    [0, 0.4 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={10}
      leftWidth="52%"
      left={
        <div style={{ width: 800 }}>
          <SlideTitle
            title="Mixed Precision Training"
            subtitle="Using different numeric formats for speed and accuracy"
          />

          {/* Data type comparison table */}
          <div
            style={{
              backgroundColor: "rgba(255,255,255,0.03)",
              borderRadius: 10,
              border: "1px solid rgba(255,255,255,0.08)",
              overflow: "hidden",
              opacity: headerOpacity,
            }}
          >
            {/* Header */}
            <div
              style={{
                display: "flex",
                padding: "10px 16px",
                backgroundColor: "rgba(255,255,255,0.05)",
                borderBottom: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              {COLUMNS.map((col, ci) => (
                <div
                  key={col}
                  style={{
                    width: COL_WIDTHS[ci],
                    fontSize: 14,
                    fontWeight: 700,
                    color: THEME.colors.textSecondary,
                    fontFamily: fontFamilyBody,
                    flexShrink: 0,
                  }}
                >
                  {col}
                </div>
              ))}
            </div>

            {/* Rows */}
            {DATA_TYPES.map((row, ri) => {
              const rowDelay = 1.2 * fps + ri * 0.4 * fps;
              const rowSpring = spring({
                frame: frame - rowDelay,
                fps,
                config: { damping: 200 },
              });
              const rowOpacity = interpolate(rowSpring, [0, 1], [0, 1]);
              const rowX = interpolate(rowSpring, [0, 1], [-20, 0]);

              const bgColor = row.highlight
                ? "rgba(118,185,0,0.08)"
                : "transparent";
              const borderColor = row.highlight
                ? `${THEME.colors.nvidiaGreen}30`
                : "rgba(255,255,255,0.04)";

              return (
                <div
                  key={row.name}
                  style={{
                    display: "flex",
                    padding: "10px 16px",
                    backgroundColor: bgColor,
                    borderBottom: `1px solid ${borderColor}`,
                    opacity: rowOpacity,
                    transform: `translateX(${rowX}px)`,
                  }}
                >
                  <div
                    style={{
                      width: COL_WIDTHS[0],
                      fontSize: 15,
                      fontWeight: 700,
                      color: row.highlight
                        ? THEME.colors.nvidiaGreen
                        : THEME.colors.textPrimary,
                      fontFamily: fontFamilyCode,
                      flexShrink: 0,
                    }}
                  >
                    {row.name}
                  </div>
                  <div
                    style={{
                      width: COL_WIDTHS[1],
                      fontSize: 15,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyCode,
                      flexShrink: 0,
                    }}
                  >
                    {row.bits}
                  </div>
                  <div
                    style={{
                      width: COL_WIDTHS[2],
                      fontSize: 14,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyCode,
                      flexShrink: 0,
                    }}
                  >
                    {row.range}
                  </div>
                  <div
                    style={{
                      width: COL_WIDTHS[3],
                      fontSize: 14,
                      color: row.highlight
                        ? THEME.colors.nvidiaGreen
                        : THEME.colors.textMuted,
                      fontFamily: fontFamilyBody,
                      fontWeight: row.highlight ? 600 : 400,
                      flexShrink: 0,
                    }}
                  >
                    {row.useCase}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      }
      right={
        <div style={{ width: 540, marginTop: 80 }}>
          <div
            style={{
              fontSize: 24,
              fontWeight: 700,
              color: THEME.colors.accentBlue,
              fontFamily: fontFamilyBody,
              marginBottom: 20,
              opacity: interpolate(
                frame - 3.5 * fps,
                [0, 0.4 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            Why Mixed Precision?
          </div>
          <BulletPoint
            text="2x less memory -- larger batch sizes"
            index={0}
            delay={4 * fps}
            highlight
          />
          <BulletPoint
            text="2-8x faster compute on Tensor Cores"
            index={1}
            delay={4 * fps}
            highlight
          />
          <BulletPoint
            text="Same model quality with proper techniques"
            index={2}
            delay={4 * fps}
            subtext="Loss scaling and master weights preserve accuracy"
          />
          <BulletPoint
            text="Industry standard since 2018"
            index={3}
            delay={4 * fps}
            subtext="NVIDIA mixed precision paper + PyTorch AMP"
          />
        </div>
      }
    />
  );
};
