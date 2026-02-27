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

const BLOCK_TILE = 8; // 64x64 block shown as 8x8 for display
const THREAD_TILE = 2; // 8x8 thread tile shown as 2x2 for display
const CELL = 20;
const GAP = 2;

export const M8S07_RegisterTiling: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const diagramOpacity = interpolate(
    frame - 1.5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Animate: highlight one thread's sub-matrix
  const threadHighlightOpacity = interpolate(
    frame - 3 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Outer product animation
  const outerStart = 5 * fps;
  const outerProgress = interpolate(
    frame - outerStart,
    [0, 1.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const regLoadOpacity = interpolate(
    frame - 4 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const flopsLabelOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const perfOpacity = interpolate(
    frame - 10 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const blockGridW = BLOCK_TILE * (CELL + GAP) - GAP;

  // Thread sub-tile position (highlight thread at row 1, col 2 in the grid of threads)
  const threadRow = 1;
  const threadCol = 2;
  const subStartR = threadRow * THREAD_TILE;
  const subEndR = subStartR + THREAD_TILE;
  const subStartC = threadCol * THREAD_TILE;
  const subEndC = subStartC + THREAD_TILE;

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={8}
      leftWidth="52%"
      left={
        <div style={{ width: 600 }}>
          <SlideTitle
            title="Register Tiling (Thread Coarsening)"
            subtitle="Each thread computes a TM x TN sub-matrix of C"
          />

          {/* Block tile in shared memory */}
          <div style={{ marginTop: 12, opacity: diagramOpacity }}>
            <div
              style={{
                fontSize: 15,
                fontWeight: 700,
                color: THEME.colors.accentCyan,
                fontFamily: fontFamilyBody,
                marginBottom: 8,
              }}
            >
              Block Tile in Shared Memory (64x64)
            </div>

            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                width: blockGridW,
                gap: GAP,
              }}
            >
              {Array.from({ length: BLOCK_TILE * BLOCK_TILE }).map((_, idx) => {
                const r = Math.floor(idx / BLOCK_TILE);
                const c = idx % BLOCK_TILE;
                const isThread =
                  r >= subStartR && r < subEndR && c >= subStartC && c < subEndC;

                let bg = "rgba(255,255,255,0.05)";
                let border = "rgba(255,255,255,0.1)";

                if (isThread) {
                  bg = `${THEME.colors.nvidiaGreen}50`;
                  border = THEME.colors.nvidiaGreen;
                }

                return (
                  <div
                    key={idx}
                    style={{
                      width: CELL,
                      height: CELL,
                      borderRadius: 2,
                      backgroundColor: bg,
                      border: `1px solid ${border}`,
                      opacity:
                        isThread
                          ? threadHighlightOpacity
                          : 1,
                    }}
                  />
                );
              })}
            </div>

            <div
              style={{
                fontSize: 12,
                color: THEME.colors.textMuted,
                fontFamily: fontFamilyCode,
                marginTop: 6,
              }}
            >
              Green = one thread's 8x8 output sub-matrix
            </div>
          </div>

          {/* Register load + outer product diagram */}
          <div style={{ marginTop: 24, opacity: regLoadOpacity }}>
            <div
              style={{
                fontSize: 15,
                fontWeight: 700,
                color: THEME.colors.accentYellow,
                fontFamily: fontFamilyBody,
                marginBottom: 10,
              }}
            >
              Outer Product (per k-step)
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 20, width: 550 }}>
              {/* A column fragment (TM values) */}
              <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <div
                  style={{
                    fontSize: 11,
                    color: THEME.colors.accentBlue,
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                    textAlign: "center",
                    marginBottom: 2,
                  }}
                >
                  A regs (TM=8)
                </div>
                {Array.from({ length: 8 }).map((_, i) => {
                  const fillOpacity = interpolate(
                    outerProgress,
                    [i / 8, (i + 1) / 8],
                    [0, 1],
                    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                  );
                  return (
                    <div
                      key={i}
                      style={{
                        width: 28,
                        height: 14,
                        borderRadius: 2,
                        backgroundColor: `rgba(79,195,247,${0.2 + fillOpacity * 0.5})`,
                        border: `1px solid ${THEME.colors.accentBlue}`,
                      }}
                    />
                  );
                })}
              </div>

              <div
                style={{
                  fontSize: 24,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyBody,
                }}
              >
                x
              </div>

              {/* B row fragment (TN values) */}
              <div>
                <div
                  style={{
                    fontSize: 11,
                    color: THEME.colors.accentPurple,
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                    textAlign: "center",
                    marginBottom: 2,
                  }}
                >
                  B regs (TN=8)
                </div>
                <div style={{ display: "flex", gap: 3 }}>
                  {Array.from({ length: 8 }).map((_, i) => {
                    const fillOpacity = interpolate(
                      outerProgress,
                      [i / 8, (i + 1) / 8],
                      [0, 1],
                      { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                    );
                    return (
                      <div
                        key={i}
                        style={{
                          width: 14,
                          height: 28,
                          borderRadius: 2,
                          backgroundColor: `rgba(179,136,255,${0.2 + fillOpacity * 0.5})`,
                          border: `1px solid ${THEME.colors.accentPurple}`,
                        }}
                      />
                    );
                  })}
                </div>
              </div>

              <div
                style={{
                  fontSize: 24,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyBody,
                }}
              >
                =
              </div>

              {/* 8x8 outer product result */}
              <div>
                <div
                  style={{
                    fontSize: 11,
                    color: THEME.colors.nvidiaGreen,
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                    textAlign: "center",
                    marginBottom: 2,
                  }}
                >
                  8x8 = 64 FLOPs
                </div>
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    width: 8 * 17,
                    gap: 2,
                  }}
                >
                  {Array.from({ length: 64 }).map((_, idx) => {
                    const cellDelay = idx / 64;
                    const cellOpacity = interpolate(
                      outerProgress,
                      [cellDelay, cellDelay + 0.3],
                      [0, 1],
                      { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                    );
                    return (
                      <div
                        key={idx}
                        style={{
                          width: 14,
                          height: 14,
                          borderRadius: 1,
                          backgroundColor: `rgba(118,185,0,${cellOpacity * 0.6})`,
                          border: `1px solid ${THEME.colors.nvidiaGreen}60`,
                        }}
                      />
                    );
                  })}
                </div>
              </div>
            </div>

            {/* FLOPs highlight */}
            <div
              style={{
                marginTop: 12,
                fontSize: 15,
                color: THEME.colors.accentYellow,
                fontFamily: fontFamilyCode,
                fontWeight: 700,
                opacity: flopsLabelOpacity,
              }}
            >
              64 FLOPs from only 16 shared memory reads
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 60, width: 480 }}>
          <BulletPoint
            index={0}
            delay={3 * fps}
            text="Each thread computes 8x8 = 64 output elements"
            icon="1"
          />
          <BulletPoint
            index={1}
            delay={3 * fps}
            text="Load TM values of A + TN values of B into registers"
            icon="2"
            highlight
          />
          <BulletPoint
            index={2}
            delay={3 * fps}
            text="Outer product: TM x TN FLOPs per shared mem access"
            icon="3"
          />
          <BulletPoint
            index={3}
            delay={3 * fps}
            text="Arithmetic intensity jumps: 64 / 16 = 4 FLOPs/read"
            icon="4"
            highlight
          />

          {/* Performance badge */}
          <div
            style={{
              marginTop: 40,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: perfOpacity,
              width: 440,
            }}
          >
            <div
              style={{
                fontSize: 17,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                lineHeight: 1.5,
              }}
            >
              Performance:{" "}
              <span style={{ color: THEME.colors.accentYellow, fontWeight: 800 }}>
                ~5,000 GFLOPS
              </span>{" "}
              (~30% of peak)
            </div>
            <div
              style={{
                fontSize: 15,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
                marginTop: 4,
              }}
            >
              10x over basic tiling!
            </div>
          </div>
        </div>
      }
    />
  );
};
