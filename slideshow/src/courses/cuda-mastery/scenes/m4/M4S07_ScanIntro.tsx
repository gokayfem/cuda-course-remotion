import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M4S07_ScanIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const input =     [3, 1, 7, 0, 4, 1, 6, 3];
  const inclusive =  [3, 4, 11, 11, 15, 16, 22, 25];
  const exclusive =  [0, 3, 4, 11, 11, 15, 16, 22];

  const CELL_W = 72;
  const CELL_H = 48;
  const CELL_GAP = 6;
  const GRID_W = input.length * (CELL_W + CELL_GAP) - CELL_GAP;

  const rows = [
    { label: "Input", data: input, color: THEME.colors.accentCyan, delay: 0.8 * fps },
    { label: "Inclusive Scan", data: inclusive, color: THEME.colors.nvidiaGreen, delay: 2.5 * fps },
    { label: "Exclusive Scan", data: exclusive, color: THEME.colors.accentPurple, delay: 4.5 * fps },
  ];

  const identityDelay = 6 * fps;
  const identityOpacity = interpolate(
    frame - identityDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={4} slideNumber={7} totalSlides={18}>
      <SlideTitle
        title="Prefix Sum (Scan)"
        subtitle="Compute all running totals in parallel"
      />

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 20,
          flex: 1,
        }}
      >
        {/* Index header */}
        <div
          style={{
            display: "flex",
            gap: CELL_GAP,
            marginLeft: 160,
            opacity: interpolate(
              frame - 0.5 * fps,
              [0, 0.3 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          {input.map((_, i) => (
            <div
              key={i}
              style={{
                width: CELL_W,
                fontSize: 13,
                color: THEME.colors.textMuted,
                fontFamily: fontFamilyCode,
                textAlign: "center",
              }}
            >
              [{i}]
            </div>
          ))}
        </div>

        {/* Data rows */}
        {rows.map((row, ri) => {
          const rowSpring = spring({
            frame: frame - row.delay,
            fps,
            config: { damping: 200 },
          });
          const rowOpacity = interpolate(rowSpring, [0, 1], [0, 1]);
          const rowY = interpolate(rowSpring, [0, 1], [15, 0]);

          return (
            <div
              key={ri}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 16,
                opacity: rowOpacity,
                transform: `translateY(${rowY}px)`,
              }}
            >
              {/* Row label */}
              <div
                style={{
                  width: 140,
                  fontSize: 16,
                  fontWeight: 700,
                  color: row.color,
                  fontFamily: fontFamilyBody,
                  textAlign: "right",
                }}
              >
                {row.label}
              </div>

              {/* Cells */}
              <div style={{ display: "flex", gap: CELL_GAP }}>
                {row.data.map((v, ci) => {
                  const cellDelay = row.delay + ci * 0.08 * fps;
                  const cellSpring = spring({
                    frame: frame - cellDelay,
                    fps,
                    config: { damping: 200 },
                  });
                  const cellScale = interpolate(cellSpring, [0, 1], [0.7, 1]);

                  // Highlight the identity element for exclusive scan
                  const isIdentity = ri === 2 && ci === 0;

                  return (
                    <div
                      key={ci}
                      style={{
                        width: CELL_W,
                        height: CELL_H,
                        borderRadius: 6,
                        backgroundColor: isIdentity
                          ? "rgba(255,171,64,0.20)"
                          : `${row.color}15`,
                        border: `2px solid ${isIdentity ? THEME.colors.accentOrange : row.color}60`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 20,
                        fontWeight: 700,
                        color: isIdentity ? THEME.colors.accentOrange : row.color,
                        fontFamily: fontFamilyCode,
                        transform: `scale(${cellScale})`,
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

        {/* Explanation boxes */}
        <div
          style={{
            display: "flex",
            gap: 24,
            marginTop: 24,
            width: GRID_W + 160,
            justifyContent: "center",
          }}
        >
          {/* Inclusive explanation */}
          <div
            style={{
              flex: 1,
              padding: "14px 18px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: interpolate(
                frame - 3.5 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div style={{ fontSize: 15, fontWeight: 700, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyBody, marginBottom: 6 }}>
              Inclusive Scan
            </div>
            <div style={{ fontSize: 13, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, lineHeight: 1.5 }}>
              <span style={{ fontFamily: fontFamilyCode, color: THEME.colors.textCode }}>out[i] = in[0] + in[1] + ... + in[i]</span>
              <br />
              Includes the current element in the sum.
            </div>
          </div>

          {/* Exclusive explanation */}
          <div
            style={{
              flex: 1,
              padding: "14px 18px",
              backgroundColor: "rgba(179,136,255,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.accentPurple}40`,
              opacity: interpolate(
                frame - 5.5 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div style={{ fontSize: 15, fontWeight: 700, color: THEME.colors.accentPurple, fontFamily: fontFamilyBody, marginBottom: 6 }}>
              Exclusive Scan
            </div>
            <div style={{ fontSize: 13, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, lineHeight: 1.5 }}>
              <span style={{ fontFamily: fontFamilyCode, color: THEME.colors.textCode }}>out[i] = in[0] + in[1] + ... + in[i-1]</span>
              <br />
              Excludes the current element. Starts with{" "}
              <span style={{ color: THEME.colors.accentOrange, fontWeight: 700 }}>identity (0)</span>.
            </div>
          </div>
        </div>

        {/* Identity element callout */}
        <div
          style={{
            opacity: identityOpacity,
            padding: "10px 20px",
            backgroundColor: "rgba(255,171,64,0.08)",
            borderRadius: 8,
            border: `1px solid ${THEME.colors.accentOrange}40`,
            textAlign: "center",
          }}
        >
          <span
            style={{
              fontSize: 15,
              color: THEME.colors.textPrimary,
              fontFamily: fontFamilyBody,
              fontWeight: 600,
            }}
          >
            The{" "}
            <span style={{ color: THEME.colors.accentOrange, fontWeight: 700 }}>identity element</span>{" "}
            (0 for sum, 1 for product) seeds the exclusive scan.
            Exclusive scan is more common in GPU algorithms.
          </span>
        </div>
      </div>
    </SlideLayout>
  );
};
