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
import { fontFamilyHeading, fontFamilyBody } from "../../../../styles/fonts";

const CELL = 48;
const GAP = 4;
const GRID = CELL + GAP;

const matA = [
  [2, 0, 1],
  [3, 1, 0],
  [1, 2, 1],
];
const matB = [
  [1, 0, 2],
  [0, 1, 1],
  [3, 2, 0],
];

const computeC = (a: number[][], b: number[][]): number[][] => {
  const c: number[][] = [];
  for (let i = 0; i < 3; i++) {
    c.push([]);
    for (let j = 0; j < 3; j++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) {
        sum += a[i][k] * b[k][j];
      }
      c[i].push(sum);
    }
  }
  return c;
};

const matC = computeC(matA, matB);

const MatrixGrid: React.FC<{
  matrix: number[][];
  label: string;
  color: string;
  highlightRow?: number;
  highlightCol?: number;
  highlightCell?: { row: number; col: number };
  opacity: number;
  fillProgress?: number;
  filledCells?: Array<{ row: number; col: number }>;
}> = ({
  matrix,
  label,
  color,
  highlightRow,
  highlightCol,
  highlightCell,
  opacity,
  fillProgress = 0,
  filledCells = [],
}) => {
  const gridW = 3 * GRID - GAP;

  return (
    <div style={{ opacity, display: "flex", flexDirection: "column", alignItems: "center" }}>
      <div
        style={{
          fontSize: 16,
          fontFamily: fontFamilyBody,
          fontWeight: 700,
          color,
          marginBottom: 10,
          letterSpacing: "1px",
        }}
      >
        {label}
      </div>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          width: gridW,
          gap: GAP,
        }}
      >
        {matrix.flat().map((val, idx) => {
          const row = Math.floor(idx / 3);
          const col = idx % 3;
          const isHighlightRow = highlightRow === row;
          const isHighlightCol = highlightCol === col;
          const isHighlightCell =
            highlightCell !== undefined &&
            highlightCell.row === row &&
            highlightCell.col === col;
          const isFilled = filledCells.some(
            (c) => c.row === row && c.col === col
          );

          let bg = "rgba(255,255,255,0.06)";
          let border = "rgba(255,255,255,0.1)";
          let textColor: string = THEME.colors.textSecondary;

          if (isHighlightRow || isHighlightCol) {
            bg = `${color}30`;
            border = `${color}80`;
            textColor = color;
          }
          if (isHighlightCell) {
            bg = `${color}50`;
            border = color;
            textColor = THEME.colors.textPrimary;
          }
          if (isFilled) {
            bg = `${color}40`;
            border = `${color}90`;
            textColor = THEME.colors.textPrimary;
          }

          return (
            <div
              key={idx}
              style={{
                width: CELL,
                height: CELL,
                backgroundColor: bg,
                border: `2px solid ${border}`,
                borderRadius: 6,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 18,
                fontWeight: 700,
                color: textColor,
                fontFamily: fontFamilyBody,
              }}
            >
              {isFilled || isHighlightCell ? val : val}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const M8S01_Title: React.FC = () => {
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

  const greenSpring = spring({
    frame,
    fps,
    config: { damping: 200 },
    delay: 0.7 * fps,
  });
  const greenOpacity = interpolate(greenSpring, [0, 1], [0, 1]);
  const greenY = interpolate(greenSpring, [0, 1], [30, 0]);

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

  const matrixOpacity = interpolate(
    frame,
    [1.2 * fps, 1.8 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Dot product highlight cycles through all 9 C elements
  const animStart = 2.5 * fps;
  const cycleDuration = 1.5 * fps;
  const elapsed = Math.max(0, frame - animStart);
  const cellIndex = Math.min(Math.floor(elapsed / cycleDuration), 8);
  const cellProgress = Math.min((elapsed % cycleDuration) / (0.8 * fps), 1);

  const highlightRowA = Math.floor(cellIndex / 3);
  const highlightColB = cellIndex % 3;

  const filledCells: Array<{ row: number; col: number }> = [];
  for (let i = 0; i < cellIndex; i++) {
    filledCells.push({ row: Math.floor(i / 3), col: i % 3 });
  }

  const showHighlight = frame >= animStart;
  const currentCellFilled = cellProgress >= 0.7;
  if (currentCellFilled) {
    filledCells.push({ row: highlightRowA, col: highlightColB });
  }

  return (
    <AbsoluteFill>
      <SlideBackground variant="accent" />

      {/* Matrix multiplication diagram on right */}
      <div
        style={{
          position: "absolute",
          right: 80,
          top: 160,
          width: 620,
          display: "flex",
          alignItems: "center",
          gap: 20,
          opacity: matrixOpacity,
        }}
      >
        <MatrixGrid
          matrix={matA}
          label="A"
          color={THEME.colors.accentBlue}
          highlightRow={showHighlight && !currentCellFilled ? highlightRowA : undefined}
          opacity={1}
        />
        <div
          style={{
            fontSize: 32,
            fontWeight: 700,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
          }}
        >
          x
        </div>
        <MatrixGrid
          matrix={matB}
          label="B"
          color={THEME.colors.accentPurple}
          highlightCol={showHighlight && !currentCellFilled ? highlightColB : undefined}
          opacity={1}
        />
        <div
          style={{
            fontSize: 32,
            fontWeight: 700,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
          }}
        >
          =
        </div>
        <MatrixGrid
          matrix={matC}
          label="C"
          color={THEME.colors.nvidiaGreen}
          highlightCell={
            showHighlight && !currentCellFilled
              ? { row: highlightRowA, col: highlightColB }
              : undefined
          }
          filledCells={filledCells}
          opacity={1}
        />
      </div>

      {/* Dot product label */}
      {showHighlight && !currentCellFilled && (
        <div
          style={{
            position: "absolute",
            right: 180,
            top: 430,
            fontSize: 14,
            color: THEME.colors.accentCyan,
            fontFamily: fontFamilyBody,
            fontWeight: 600,
            opacity: interpolate(cellProgress, [0, 0.3], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          row {highlightRowA} of A . col {highlightColB} of B = C[{highlightRowA}][{highlightColB}]
        </div>
      )}

      {/* Main content on left */}
      <div
        style={{
          position: "absolute",
          left: 100,
          top: "50%",
          transform: "translateY(-50%)",
          maxWidth: 800,
          width: 800,
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
          Matrix Multiplication
        </h1>
        <h1
          style={{
            fontSize: 72,
            fontWeight: 900,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyHeading,
            margin: 0,
            marginTop: 8,
            opacity: greenOpacity,
            transform: `translateY(${greenY}px)`,
            lineHeight: 1.1,
            letterSpacing: "-2px",
          }}
        >
          Deep Dive
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
            width: 600,
          }}
        >
          From Naive to Near-Optimal: The GEMM Journey
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
            Module 8
          </div>
          <span
            style={{
              fontSize: 22,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
            }}
          >
            The kernel that matters most
          </span>
        </div>
      </div>

      {/* Bottom gradient bar */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          height: 6,
          background: `linear-gradient(90deg, ${THEME.colors.nvidiaGreen}, ${THEME.colors.accentBlue}, ${THEME.colors.accentPurple})`,
        }}
      />
    </AbsoluteFill>
  );
};
