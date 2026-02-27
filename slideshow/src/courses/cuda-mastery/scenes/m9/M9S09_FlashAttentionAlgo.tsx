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

interface AlgoStep {
  readonly num: number;
  readonly title: string;
  readonly detail: string;
  readonly color: string;
}

const ALGO_STEPS: readonly AlgoStep[] = [
  {
    num: 1,
    title: "Divide Q into blocks",
    detail: "Q split into B_r-row tiles, loaded to SRAM",
    color: THEME.colors.accentBlue,
  },
  {
    num: 2,
    title: "Iterate K, V blocks",
    detail: "For each K_j, V_j tile (B_c cols), load to SRAM",
    color: THEME.colors.accentOrange,
  },
  {
    num: 3,
    title: "Compute local S = Q_i x K_j\u1D40",
    detail: "Small B_r x B_c tile fits entirely in SRAM",
    color: THEME.colors.accentPurple,
  },
  {
    num: 4,
    title: "Online softmax on tile",
    detail: "Track running max m_i, denominator l_i per row",
    color: THEME.colors.accentCyan,
  },
  {
    num: 5,
    title: "Accumulate output O_i",
    detail: "O_i = diag(l_old/l_new) * O_old + P_ij * V_j",
    color: THEME.colors.nvidiaGreen,
  },
  {
    num: 6,
    title: "Write O once to HBM",
    detail: "Final output written only after all K,V tiles processed",
    color: THEME.colors.accentYellow,
  },
];

const TILE_ROWS = 4;
const TILE_COLS = 4;
const TILE_SIZE = 28;
const TILE_GAP = 3;

export const M9S09_FlashAttentionAlgo: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const N = 8;
  const tilesPerRow = Math.ceil(N / TILE_ROWS);

  const sweepProgress = interpolate(
    frame - 3 * fps,
    [0, 8 * fps],
    [0, tilesPerRow * tilesPerRow - 0.01],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );
  const activeTileIdx = Math.floor(sweepProgress);
  const activeQBlock = Math.floor(activeTileIdx / tilesPerRow);
  const activeKBlock = activeTileIdx % tilesPerRow;

  return (
    <SlideLayout variant="gradient" moduleNumber={9}>
      <SlideTitle
        title="Flash Attention Algorithm"
        subtitle="Tiled computation keeps data in SRAM, avoiding HBM round-trips"
      />

      <div
        style={{
          flex: 1,
          display: "flex",
          gap: 48,
          width: 1776,
        }}
      >
        {/* Left: algorithm steps */}
        <div style={{ width: 720, flexShrink: 0 }}>
          {ALGO_STEPS.map((step, i) => {
            const stepDelay = 0.8 * fps + i * 0.8 * fps;
            const stepSpring = spring({
              frame: frame - stepDelay,
              fps,
              config: { damping: 200 },
            });
            const opacity = interpolate(stepSpring, [0, 1], [0, 1]);
            const translateX = interpolate(stepSpring, [0, 1], [20, 0]);

            const correspondingTile = Math.floor(
              (frame - 3 * fps) / (8 * fps / (tilesPerRow * tilesPerRow))
            );
            const isStepActive =
              frame >= 3 * fps &&
              i === Math.min(
                Math.floor(correspondingTile / Math.max(1, tilesPerRow * tilesPerRow / 6)),
                5
              );

            return (
              <div
                key={step.num}
                style={{
                  opacity,
                  transform: `translateX(${translateX}px)`,
                  marginBottom: 10,
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                }}
              >
                <div
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: 16,
                    backgroundColor: isStepActive
                      ? `${step.color}35`
                      : `${step.color}15`,
                    border: `2px solid ${step.color}${isStepActive ? "" : "60"}`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 14,
                    fontWeight: 700,
                    color: step.color,
                    fontFamily: fontFamilyBody,
                    flexShrink: 0,
                    boxShadow: isStepActive
                      ? `0 0 10px ${step.color}40`
                      : "none",
                  }}
                >
                  {step.num}
                </div>
                <div style={{ flex: 1 }}>
                  <span
                    style={{
                      fontSize: 18,
                      fontWeight: 700,
                      color: step.color,
                      fontFamily: fontFamilyBody,
                    }}
                  >
                    {step.title}
                  </span>
                  <div
                    style={{
                      fontSize: 15,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyBody,
                      marginTop: 2,
                    }}
                  >
                    {step.detail}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Right: animated tiled matrix visualization */}
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            opacity: interpolate(
              frame - 2.5 * fps,
              [0, 0.5 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          {/* Matrix labels */}
          <div
            style={{
              display: "flex",
              gap: 120,
              marginBottom: 16,
            }}
          >
            <span
              style={{
                fontSize: 16,
                color: THEME.colors.accentBlue,
                fontFamily: fontFamilyCode,
                fontWeight: 700,
              }}
            >
              Q blocks (rows)
            </span>
            <span
              style={{
                fontSize: 16,
                color: THEME.colors.accentOrange,
                fontFamily: fontFamilyCode,
                fontWeight: 700,
              }}
            >
              K/V blocks (cols)
            </span>
          </div>

          {/* Tiled grid showing attention computation */}
          <div
            style={{
              position: "relative",
              display: "flex",
              flexDirection: "column",
              gap: 6,
            }}
          >
            {Array.from({ length: tilesPerRow }).map((_, qIdx) => (
              <div key={qIdx} style={{ display: "flex", gap: 6 }}>
                {/* Q block label */}
                <div
                  style={{
                    width: 40,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 13,
                    color: THEME.colors.accentBlue,
                    fontFamily: fontFamilyCode,
                    fontWeight: 600,
                  }}
                >
                  Q{qIdx}
                </div>

                {Array.from({ length: tilesPerRow }).map((_, kIdx) => {
                  const tileIdx = qIdx * tilesPerRow + kIdx;
                  const isActive =
                    qIdx === activeQBlock && kIdx === activeKBlock;
                  const isProcessed = tileIdx < activeTileIdx;
                  const tileDelay = 2.5 * fps + tileIdx * 2;

                  const tileOpacity = interpolate(
                    frame - tileDelay,
                    [0, 0.3 * fps],
                    [0, 1],
                    {
                      extrapolateLeft: "clamp",
                      extrapolateRight: "clamp",
                    }
                  );

                  const bgColor = isActive
                    ? `${THEME.colors.nvidiaGreen}40`
                    : isProcessed
                      ? `${THEME.colors.nvidiaGreen}18`
                      : "rgba(255,255,255,0.04)";

                  const borderColor = isActive
                    ? THEME.colors.nvidiaGreen
                    : isProcessed
                      ? `${THEME.colors.nvidiaGreen}50`
                      : "rgba(255,255,255,0.1)";

                  return (
                    <div
                      key={kIdx}
                      style={{
                        width: TILE_SIZE * TILE_COLS + TILE_GAP * (TILE_COLS - 1) + 12,
                        height: TILE_SIZE * TILE_ROWS + TILE_GAP * (TILE_ROWS - 1) + 12,
                        backgroundColor: bgColor,
                        border: `2px solid ${borderColor}`,
                        borderRadius: 8,
                        display: "flex",
                        flexWrap: "wrap",
                        alignContent: "center",
                        justifyContent: "center",
                        gap: TILE_GAP,
                        padding: 4,
                        opacity: tileOpacity,
                        boxShadow: isActive
                          ? `0 0 16px ${THEME.colors.nvidiaGreen}50`
                          : "none",
                      }}
                    >
                      {Array.from({
                        length: TILE_ROWS * TILE_COLS,
                      }).map((_, cellIdx) => {
                        const cellColor = isActive
                          ? THEME.colors.nvidiaGreen
                          : isProcessed
                            ? `${THEME.colors.nvidiaGreen}80`
                            : "rgba(255,255,255,0.08)";

                        return (
                          <div
                            key={cellIdx}
                            style={{
                              width: TILE_SIZE,
                              height: TILE_SIZE,
                              backgroundColor: isActive
                                ? `${THEME.colors.nvidiaGreen}30`
                                : isProcessed
                                  ? `${THEME.colors.nvidiaGreen}10`
                                  : "rgba(255,255,255,0.02)",
                              borderRadius: 3,
                              border: `1px solid ${cellColor}40`,
                            }}
                          />
                        );
                      })}
                    </div>
                  );
                })}
              </div>
            ))}

            {/* K/V column labels */}
            <div style={{ display: "flex", gap: 6, marginLeft: 46 }}>
              {Array.from({ length: tilesPerRow }).map((_, kIdx) => (
                <div
                  key={kIdx}
                  style={{
                    width:
                      TILE_SIZE * TILE_COLS +
                      TILE_GAP * (TILE_COLS - 1) +
                      12,
                    textAlign: "center",
                    fontSize: 13,
                    color: THEME.colors.accentOrange,
                    fontFamily: fontFamilyCode,
                    fontWeight: 600,
                  }}
                >
                  K{kIdx}/V{kIdx}
                </div>
              ))}
            </div>
          </div>

          {/* SRAM / HBM labels */}
          <div
            style={{
              marginTop: 20,
              display: "flex",
              gap: 28,
              opacity: interpolate(
                frame - 4 * fps,
                [0, 0.4 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div
              style={{
                padding: "8px 18px",
                backgroundColor: "rgba(118,185,0,0.1)",
                border: `1px solid ${THEME.colors.nvidiaGreen}50`,
                borderRadius: 8,
                fontSize: 14,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyCode,
                fontWeight: 600,
              }}
            >
              Active tile = SRAM
            </div>
            <div
              style={{
                padding: "8px 18px",
                backgroundColor: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 8,
                fontSize: 14,
                color: THEME.colors.textMuted,
                fontFamily: fontFamilyCode,
                fontWeight: 600,
              }}
            >
              Waiting tiles = HBM only
            </div>
            <div
              style={{
                padding: "8px 18px",
                backgroundColor: `${THEME.colors.nvidiaGreen}10`,
                border: `1px solid ${THEME.colors.nvidiaGreen}30`,
                borderRadius: 8,
                fontSize: 14,
                color: `${THEME.colors.nvidiaGreen}90`,
                fontFamily: fontFamilyCode,
                fontWeight: 600,
              }}
            >
              Done = O accumulated
            </div>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
