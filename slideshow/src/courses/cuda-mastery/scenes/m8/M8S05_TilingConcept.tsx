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

const MAT_SIZE = 8;
const TILE_SIZE = 2;
const CELL = 28;
const GAP = 2;
const GRID_W = MAT_SIZE * (CELL + GAP) - GAP;
const TOTAL_TILES = MAT_SIZE / TILE_SIZE; // 4 tiles along K

export const M8S05_TilingConcept: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Tile animation: slide through K dimension
  const animStart = 2.5 * fps;
  const tileDuration = 2.5 * fps;
  const elapsed = Math.max(0, frame - animStart);
  const currentTile = Math.min(
    Math.floor(elapsed / tileDuration),
    TOTAL_TILES - 1
  );
  const tileProgress = Math.min((elapsed % tileDuration) / tileDuration, 1);

  const isAnimating = frame >= animStart;

  // Phase within each tile step
  const loadPhase = interpolate(tileProgress, [0, 0.3], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const computePhase = interpolate(tileProgress, [0.3, 0.7], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const slidePhase = interpolate(tileProgress, [0.7, 1.0], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const diagramOpacity = interpolate(
    frame - 1.5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Step labels
  const step1Opacity = interpolate(loadPhase, [0, 0.5], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const step2Opacity = interpolate(computePhase, [0, 0.5], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const step3Opacity = interpolate(slidePhase, [0, 0.5], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const bottomOpacity = interpolate(
    frame - 11 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Tile highlight range in K dimension
  const tileStart = currentTile * TILE_SIZE;
  const tileEnd = tileStart + TILE_SIZE;

  // Output block highlight (always row 0-1, col 0-1 for demo)
  const outRowStart = 0;
  const outRowEnd = TILE_SIZE;
  const outColStart = 0;
  const outColEnd = TILE_SIZE;

  return (
    <SlideLayout variant="gradient" moduleNumber={8}>
      <SlideTitle
        title="The Tiling Idea"
        subtitle="Load tiles into shared memory, reuse across all threads in a block"
      />

      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: 40,
          alignItems: "flex-start",
          marginTop: 16,
          opacity: diagramOpacity,
          width: 1776,
        }}
      >
        {/* Matrix A */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div
            style={{
              fontSize: 16,
              fontWeight: 700,
              color: THEME.colors.accentBlue,
              fontFamily: fontFamilyBody,
              marginBottom: 8,
            }}
          >
            Matrix A (M x K)
          </div>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              width: GRID_W,
              gap: GAP,
            }}
          >
            {Array.from({ length: MAT_SIZE * MAT_SIZE }).map((_, idx) => {
              const r = Math.floor(idx / MAT_SIZE);
              const c = idx % MAT_SIZE;
              // Highlight: rows in output block, cols in current tile
              const isRowActive = r >= outRowStart && r < outRowEnd;
              const isColActive = c >= tileStart && c < tileEnd;
              const isTileCell = isAnimating && isRowActive && isColActive;
              const isRowOnly = isAnimating && isRowActive && !isColActive;

              let bg = "rgba(255,255,255,0.03)";
              let border = "rgba(255,255,255,0.06)";
              if (isTileCell) {
                bg =
                  loadPhase > 0.3
                    ? `${THEME.colors.accentBlue}50`
                    : `${THEME.colors.accentBlue}20`;
                border = THEME.colors.accentBlue;
              } else if (isRowOnly) {
                bg = `${THEME.colors.accentBlue}10`;
                border = `${THEME.colors.accentBlue}30`;
              }

              return (
                <div
                  key={idx}
                  style={{
                    width: CELL,
                    height: CELL,
                    borderRadius: 3,
                    backgroundColor: bg,
                    border: `1px solid ${border}`,
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
            Tile cols: [{tileStart}..{tileEnd - 1}]
          </div>
        </div>

        {/* Shared memory tiles */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
          <div
            style={{
              fontSize: 16,
              fontWeight: 700,
              color: THEME.colors.accentCyan,
              fontFamily: fontFamilyBody,
              marginBottom: 0,
            }}
          >
            Shared Memory
          </div>

          {/* As tile */}
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              width: TILE_SIZE * (CELL + GAP) - GAP,
              gap: GAP,
              opacity: isAnimating ? loadPhase : 0,
            }}
          >
            {Array.from({ length: TILE_SIZE * TILE_SIZE }).map((_, idx) => (
              <div
                key={`as-${idx}`}
                style={{
                  width: CELL,
                  height: CELL,
                  borderRadius: 3,
                  backgroundColor:
                    computePhase > 0.2
                      ? `${THEME.colors.accentBlue}70`
                      : `${THEME.colors.accentBlue}40`,
                  border: `1.5px solid ${THEME.colors.accentBlue}`,
                }}
              />
            ))}
          </div>
          <div
            style={{
              fontSize: 11,
              color: THEME.colors.accentBlue,
              fontFamily: fontFamilyCode,
              opacity: isAnimating ? loadPhase : 0,
            }}
          >
            As[TILE][TILE]
          </div>

          {/* Bs tile */}
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              width: TILE_SIZE * (CELL + GAP) - GAP,
              gap: GAP,
              opacity: isAnimating ? loadPhase : 0,
            }}
          >
            {Array.from({ length: TILE_SIZE * TILE_SIZE }).map((_, idx) => (
              <div
                key={`bs-${idx}`}
                style={{
                  width: CELL,
                  height: CELL,
                  borderRadius: 3,
                  backgroundColor:
                    computePhase > 0.2
                      ? `${THEME.colors.accentPurple}70`
                      : `${THEME.colors.accentPurple}40`,
                  border: `1.5px solid ${THEME.colors.accentPurple}`,
                }}
              />
            ))}
          </div>
          <div
            style={{
              fontSize: 11,
              color: THEME.colors.accentPurple,
              fontFamily: fontFamilyCode,
              opacity: isAnimating ? loadPhase : 0,
            }}
          >
            Bs[TILE][TILE]
          </div>

          {/* Tile step indicator */}
          <div
            style={{
              marginTop: 8,
              padding: "4px 12px",
              backgroundColor: `${THEME.colors.accentCyan}15`,
              border: `1px solid ${THEME.colors.accentCyan}40`,
              borderRadius: 12,
              fontSize: 13,
              color: THEME.colors.accentCyan,
              fontFamily: fontFamilyCode,
              fontWeight: 700,
              opacity: isAnimating ? 1 : 0,
            }}
          >
            Tile {currentTile + 1} / {TOTAL_TILES}
          </div>
        </div>

        {/* Matrix B */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div
            style={{
              fontSize: 16,
              fontWeight: 700,
              color: THEME.colors.accentPurple,
              fontFamily: fontFamilyBody,
              marginBottom: 8,
            }}
          >
            Matrix B (K x N)
          </div>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              width: GRID_W,
              gap: GAP,
            }}
          >
            {Array.from({ length: MAT_SIZE * MAT_SIZE }).map((_, idx) => {
              const r = Math.floor(idx / MAT_SIZE);
              const c = idx % MAT_SIZE;
              const isRowActive = r >= tileStart && r < tileEnd;
              const isColActive = c >= outColStart && c < outColEnd;
              const isTileCell = isAnimating && isRowActive && isColActive;
              const isColOnly = isAnimating && isColActive && !isRowActive;

              let bg = "rgba(255,255,255,0.03)";
              let border = "rgba(255,255,255,0.06)";
              if (isTileCell) {
                bg =
                  loadPhase > 0.3
                    ? `${THEME.colors.accentPurple}50`
                    : `${THEME.colors.accentPurple}20`;
                border = THEME.colors.accentPurple;
              } else if (isColOnly) {
                bg = `${THEME.colors.accentPurple}10`;
                border = `${THEME.colors.accentPurple}30`;
              }

              return (
                <div
                  key={idx}
                  style={{
                    width: CELL,
                    height: CELL,
                    borderRadius: 3,
                    backgroundColor: bg,
                    border: `1px solid ${border}`,
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
            Tile rows: [{tileStart}..{tileEnd - 1}]
          </div>
        </div>

        {/* Output C block */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div
            style={{
              fontSize: 16,
              fontWeight: 700,
              color: THEME.colors.nvidiaGreen,
              fontFamily: fontFamilyBody,
              marginBottom: 8,
            }}
          >
            C block
          </div>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              width: TILE_SIZE * (CELL + GAP) - GAP,
              gap: GAP,
            }}
          >
            {Array.from({ length: TILE_SIZE * TILE_SIZE }).map((_, idx) => {
              // Fill intensity increases with each completed tile
              const fillAlpha = isAnimating
                ? Math.min((currentTile + computePhase) / TOTAL_TILES, 1) * 0.7 + 0.1
                : 0.1;

              return (
                <div
                  key={idx}
                  style={{
                    width: CELL,
                    height: CELL,
                    borderRadius: 3,
                    backgroundColor: `rgba(118,185,0,${fillAlpha})`,
                    border: `1.5px solid ${THEME.colors.nvidiaGreen}`,
                  }}
                />
              );
            })}
          </div>
          <div
            style={{
              fontSize: 11,
              color: THEME.colors.nvidiaGreen,
              fontFamily: fontFamilyCode,
              marginTop: 6,
            }}
          >
            Accumulating
          </div>
        </div>
      </div>

      {/* Step labels */}
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: 40,
          marginTop: 24,
          width: 1776,
        }}
      >
        <div
          style={{
            padding: "8px 18px",
            backgroundColor: `${THEME.colors.accentBlue}15`,
            border: `1px solid ${THEME.colors.accentBlue}40`,
            borderRadius: 8,
            fontSize: 14,
            color: THEME.colors.accentBlue,
            fontFamily: fontFamilyBody,
            fontWeight: 600,
            opacity: isAnimating ? step1Opacity : 0,
          }}
        >
          Step 1: Load tile into shared memory
        </div>
        <div
          style={{
            padding: "8px 18px",
            backgroundColor: `${THEME.colors.nvidiaGreen}15`,
            border: `1px solid ${THEME.colors.nvidiaGreen}40`,
            borderRadius: 8,
            fontSize: 14,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyBody,
            fontWeight: 600,
            opacity: isAnimating ? step2Opacity : 0,
          }}
        >
          Step 2: All threads compute using shared tile
        </div>
        <div
          style={{
            padding: "8px 18px",
            backgroundColor: `${THEME.colors.accentOrange}15`,
            border: `1px solid ${THEME.colors.accentOrange}40`,
            borderRadius: 8,
            fontSize: 14,
            color: THEME.colors.accentOrange,
            fontFamily: fontFamilyBody,
            fontWeight: 600,
            opacity: isAnimating ? step3Opacity : 0,
          }}
        >
          Step 3: Slide to next tile, accumulate
        </div>
      </div>

      {/* Bottom insight */}
      <div
        style={{
          marginTop: 16,
          padding: "12px 24px",
          backgroundColor: "rgba(118,185,0,0.08)",
          borderRadius: 10,
          border: `1px solid ${THEME.colors.nvidiaGreen}40`,
          opacity: bottomOpacity,
          alignSelf: "center",
        }}
      >
        <span
          style={{
            fontSize: 18,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyBody,
            fontWeight: 700,
          }}
        >
          Data reuse factor = TILE_SIZE {"\u2192"} TILE_SIZE x fewer global reads
        </span>
      </div>
    </SlideLayout>
  );
};
