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

const CELL_SIZE = 54;
const CELL_GAP = 4;
const GRID_SIZE = 3;

// Row-major order: [0,1,2,3,4,5,6,7,8] maps to (row, col)
const ROW_MAJOR_ORDER = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
];

// Column-major order: same visual matrix, different memory layout
const COL_MAJOR_ORDER = [
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
];

const MATRIX_VALUES = [
  ["a", "b", "c"],
  ["d", "e", "f"],
  ["g", "h", "i"],
];

const MemoryGrid: React.FC<{
  title: string;
  titleColor: string;
  order: number[][];
  arrowDirection: "row" | "col";
  opacity: number;
  highlightPhase: number;
}> = ({ title, titleColor, order, arrowDirection, opacity, highlightPhase }) => {
  const gridWidth = GRID_SIZE * (CELL_SIZE + CELL_GAP) - CELL_GAP;

  return (
    <div
      style={{
        opacity,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        width: 320,
      }}
    >
      <div
        style={{
          fontSize: 18,
          fontFamily: fontFamilyBody,
          fontWeight: 700,
          color: titleColor,
          marginBottom: 12,
        }}
      >
        {title}
      </div>

      {/* Grid */}
      <div
        style={{
          position: "relative",
          width: gridWidth,
          height: gridWidth,
        }}
      >
        {Array.from({ length: GRID_SIZE }).map((_, row) =>
          Array.from({ length: GRID_SIZE }).map((_, col) => {
            const memAddr = order[row][col];
            const isHighlighted =
              highlightPhase > 0 &&
              memAddr <= Math.floor(highlightPhase * 8);

            return (
              <div
                key={`cell-${row}-${col}`}
                style={{
                  position: "absolute",
                  left: col * (CELL_SIZE + CELL_GAP),
                  top: row * (CELL_SIZE + CELL_GAP),
                  width: CELL_SIZE,
                  height: CELL_SIZE,
                  backgroundColor: isHighlighted
                    ? `${titleColor}30`
                    : "rgba(255,255,255,0.06)",
                  border: `2px solid ${isHighlighted ? titleColor : "rgba(255,255,255,0.15)"}`,
                  borderRadius: 6,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <span
                  style={{
                    fontSize: 18,
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                    color: THEME.colors.textPrimary,
                  }}
                >
                  {MATRIX_VALUES[row][col]}
                </span>
                <span
                  style={{
                    fontSize: 11,
                    fontFamily: fontFamilyCode,
                    color: titleColor,
                    marginTop: 2,
                  }}
                >
                  [{memAddr}]
                </span>
              </div>
            );
          })
        )}
      </div>

      {/* Arrow label */}
      <div
        style={{
          marginTop: 12,
          fontSize: 13,
          fontFamily: fontFamilyBody,
          color: THEME.colors.textMuted,
        }}
      >
        Memory: {arrowDirection === "row" ? "[a,b,c,d,e,f,g,h,i]" : "[a,d,g,b,e,h,c,f,i]"}
      </div>
      <div
        style={{
          marginTop: 4,
          fontSize: 12,
          fontFamily: fontFamilyBody,
          color: THEME.colors.textMuted,
        }}
      >
        {arrowDirection === "row" ? "Scans left-to-right, top-to-bottom" : "Scans top-to-bottom, left-to-right"}
      </div>
    </div>
  );
};

const FIX_CODE_LINES = [
  { text: "// Row-major C = A * B", color: THEME.colors.syntaxComment },
  { text: "// Compute C^T = B^T * A^T", color: THEME.colors.syntaxComment },
  { text: "cublasSgemm(handle,", color: THEME.colors.syntaxFunction },
  { text: "    CUBLAS_OP_T, CUBLAS_OP_T,", color: THEME.colors.accentOrange },
  { text: "    N, M, K, ...);", color: THEME.colors.textCode },
];

export const M7S04_ColumnMajor: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Grid animations
  const leftGridSpring = spring({
    frame: frame - 1 * fps,
    fps,
    config: { damping: 200 },
  });
  const leftGridOpacity = interpolate(leftGridSpring, [0, 1], [0, 1]);

  const rightGridSpring = spring({
    frame: frame - 2.5 * fps,
    fps,
    config: { damping: 200 },
  });
  const rightGridOpacity = interpolate(rightGridSpring, [0, 1], [0, 1]);

  // Highlight phase for cells (0 to 1 over time)
  const highlightPhase = interpolate(
    frame,
    [4 * fps, 7 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Warning flash
  const warningDelay = 5 * fps;
  const warningSpring = spring({
    frame: frame - warningDelay,
    fps,
    config: { damping: 150, stiffness: 120 },
  });
  const warningOpacity = interpolate(warningSpring, [0, 1], [0, 1]);
  const warningScale = interpolate(warningSpring, [0, 1], [0.9, 1]);

  // Flash effect on the warning
  const flashIntensity = interpolate(
    frame - warningDelay,
    [0, 0.2 * fps, 0.5 * fps, 0.8 * fps],
    [0, 1, 0.3, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Code fix
  const codeOpacity = interpolate(
    frame - 8 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Bottom tip
  const tipOpacity = interpolate(
    frame - 10 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="dark" moduleNumber={7}>
      <div style={{ width: 1776 }}>
        <SlideTitle
          title="The Column-Major Gotcha"
          subtitle="cuBLAS uses Fortran-style column-major storage"
        />

        {/* Two grids side by side */}
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            justifyContent: "center",
            gap: 80,
            marginTop: 8,
            width: 1776,
          }}
        >
          <MemoryGrid
            title="Row-Major (C/C++)"
            titleColor={THEME.colors.accentBlue}
            order={ROW_MAJOR_ORDER}
            arrowDirection="row"
            opacity={leftGridOpacity}
            highlightPhase={highlightPhase}
          />

          {/* Warning between grids */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              minHeight: 200,
              width: 280,
              opacity: warningOpacity,
              transform: `scale(${warningScale})`,
            }}
          >
            <div
              style={{
                padding: "14px 24px",
                backgroundColor: `rgba(255,82,82,${0.12 + flashIntensity * 0.2})`,
                border: `2px solid ${THEME.colors.accentRed}`,
                borderRadius: 10,
                textAlign: "center",
                boxShadow: `0 0 ${20 + flashIntensity * 30}px rgba(255,82,82,${0.2 + flashIntensity * 0.3})`,
                width: 260,
              }}
            >
              <div
                style={{
                  fontSize: 28,
                  marginBottom: 6,
                }}
              >
                !!
              </div>
              <div
                style={{
                  fontSize: 16,
                  fontFamily: fontFamilyBody,
                  fontWeight: 700,
                  color: THEME.colors.accentRed,
                  lineHeight: 1.4,
                }}
              >
                If you mix them up, results are transposed!
              </div>
            </div>
          </div>

          <MemoryGrid
            title="Column-Major (cuBLAS/Fortran)"
            titleColor={THEME.colors.accentPurple}
            order={COL_MAJOR_ORDER}
            arrowDirection="col"
            opacity={rightGridOpacity}
            highlightPhase={highlightPhase}
          />
        </div>

        {/* Fix: code + tip */}
        <div
          style={{
            display: "flex",
            gap: 40,
            marginTop: 28,
            alignItems: "flex-start",
            width: 1776,
            justifyContent: "center",
          }}
        >
          {/* Code fix */}
          <div
            style={{
              padding: "14px 20px",
              backgroundColor: THEME.colors.bgCode,
              borderRadius: 10,
              border: `1px solid rgba(255,255,255,0.08)`,
              opacity: codeOpacity,
              width: 420,
            }}
          >
            <div
              style={{
                fontSize: 13,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
                color: THEME.colors.nvidiaGreen,
                marginBottom: 8,
                letterSpacing: "1px",
              }}
            >
              THE FIX: TRANSPOSE TRICK
            </div>
            {FIX_CODE_LINES.map((line, i) => (
              <div
                key={`fix-${i}`}
                style={{
                  fontSize: 14,
                  fontFamily: fontFamilyCode,
                  color: line.color,
                  lineHeight: 1.7,
                  whiteSpace: "pre",
                }}
              >
                {line.text}
              </div>
            ))}
          </div>

          {/* Tip */}
          <div style={{ opacity: tipOpacity, width: 400 }}>
            <BulletPoint
              index={0}
              delay={0}
              text='Or just store your matrices column-major from the start'
              icon="*"
              highlight
            />
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
