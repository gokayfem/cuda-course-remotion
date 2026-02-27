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
import { fontFamilyBody, fontFamilyCode, fontFamilyHeading } from "../../../../styles/fonts";

const MATRIX_CELL = 32;
const MATRIX_GAP = 4;

const MatrixGrid: React.FC<{
  rows: number;
  cols: number;
  color: string;
  label: string;
  dimLabel: string;
  opacity: number;
  scale: number;
}> = ({ rows, cols, color, label, dimLabel, opacity, scale }) => {
  const gridWidth = cols * (MATRIX_CELL + MATRIX_GAP) - MATRIX_GAP;
  const gridHeight = rows * (MATRIX_CELL + MATRIX_GAP) - MATRIX_GAP;

  return (
    <div
      style={{
        opacity,
        transform: `scale(${scale})`,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        width: gridWidth + 20,
      }}
    >
      <div
        style={{
          fontSize: 14,
          fontFamily: fontFamilyBody,
          fontWeight: 700,
          color,
          marginBottom: 6,
        }}
      >
        {label}
      </div>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          width: gridWidth,
          gap: MATRIX_GAP,
        }}
      >
        {Array.from({ length: rows * cols }).map((_, idx) => (
          <div
            key={idx}
            style={{
              width: MATRIX_CELL,
              height: MATRIX_CELL,
              backgroundColor: `${color}40`,
              border: `1px solid ${color}80`,
              borderRadius: 3,
            }}
          />
        ))}
      </div>
      <div
        style={{
          fontSize: 12,
          fontFamily: fontFamilyCode,
          color: THEME.colors.textMuted,
          marginTop: 4,
        }}
      >
        {dimLabel}
      </div>
    </div>
  );
};

const CODE_LINES = [
  { text: "cublasHandle_t handle;", color: THEME.colors.textCode },
  { text: "cublasCreate(&handle);", color: THEME.colors.syntaxFunction },
  { text: "", color: "" },
  { text: "cublasSgemm(handle,", color: THEME.colors.syntaxFunction },
  { text: "    CUBLAS_OP_N, CUBLAS_OP_N,", color: THEME.colors.syntaxKeyword },
  { text: "    M, N, K,", color: THEME.colors.syntaxNumber },
  { text: "    &alpha,", color: THEME.colors.textCode },
  { text: "    d_A, lda,", color: THEME.colors.accentBlue },
  { text: "    d_B, ldb,", color: THEME.colors.accentPurple },
  { text: "    &beta,", color: THEME.colors.textCode },
  { text: "    d_C, ldc);", color: THEME.colors.nvidiaGreen },
  { text: "", color: "" },
  { text: "cublasDestroy(handle);", color: THEME.colors.syntaxFunction },
];

export const M7S03_CublasIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Formula animation
  const formulaSpring = spring({
    frame: frame - 1 * fps,
    fps,
    config: { damping: 200 },
  });
  const formulaOpacity = interpolate(formulaSpring, [0, 1], [0, 1]);
  const formulaScale = interpolate(formulaSpring, [0, 1], [0.8, 1]);

  // Matrix diagram animation
  const matrixSpring = spring({
    frame: frame - 3 * fps,
    fps,
    config: { damping: 200 },
  });
  const matrixOpacity = interpolate(matrixSpring, [0, 1], [0, 1]);
  const matrixScale = interpolate(matrixSpring, [0, 1], [0.85, 1]);

  // Bottom text
  const bottomOpacity = interpolate(
    frame - 10 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={7}
      leftWidth="48%"
      left={
        <div style={{ width: 560 }}>
          <SlideTitle
            title="cuBLAS -- Linear Algebra on GPU"
            subtitle="The gold standard for GEMM"
          />

          {/* GEMM Formula */}
          <div
            style={{
              marginTop: 8,
              padding: "18px 24px",
              backgroundColor: "rgba(79,195,247,0.06)",
              borderRadius: 12,
              border: `1px solid ${THEME.colors.accentBlue}30`,
              opacity: formulaOpacity,
              transform: `scale(${formulaScale})`,
              width: 520,
            }}
          >
            <div
              style={{
                fontSize: 30,
                fontFamily: fontFamilyHeading,
                fontWeight: 800,
                color: THEME.colors.textPrimary,
                textAlign: "center",
                letterSpacing: "1px",
              }}
            >
              <span style={{ color: THEME.colors.nvidiaGreen }}>C</span>
              {" = "}
              <span style={{ color: THEME.colors.accentOrange }}>{"alpha"}</span>
              {" * "}
              <span style={{ color: THEME.colors.accentBlue }}>A</span>
              {" x "}
              <span style={{ color: THEME.colors.accentPurple }}>B</span>
              {" + "}
              <span style={{ color: THEME.colors.accentOrange }}>{"beta"}</span>
              {" * "}
              <span style={{ color: THEME.colors.nvidiaGreen }}>C</span>
            </div>
            <div
              style={{
                fontSize: 14,
                fontFamily: fontFamilyCode,
                color: THEME.colors.textMuted,
                textAlign: "center",
                marginTop: 8,
              }}
            >
              A(M x K) * B(K x N) = C(M x N)
            </div>
          </div>

          {/* Matrix diagram */}
          <div
            style={{
              marginTop: 24,
              display: "flex",
              alignItems: "center",
              gap: 16,
              opacity: matrixOpacity,
              transform: `scale(${matrixScale})`,
              width: 520,
              justifyContent: "center",
            }}
          >
            <MatrixGrid
              rows={3}
              cols={2}
              color={THEME.colors.accentBlue}
              label="A"
              dimLabel="M x K"
              opacity={1}
              scale={1}
            />
            <span
              style={{
                fontSize: 24,
                color: THEME.colors.textMuted,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
              }}
            >
              x
            </span>
            <MatrixGrid
              rows={2}
              cols={4}
              color={THEME.colors.accentPurple}
              label="B"
              dimLabel="K x N"
              opacity={1}
              scale={1}
            />
            <span
              style={{
                fontSize: 24,
                color: THEME.colors.textMuted,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
              }}
            >
              =
            </span>
            <MatrixGrid
              rows={3}
              cols={4}
              color={THEME.colors.nvidiaGreen}
              label="C"
              dimLabel="M x N"
              opacity={1}
              scale={1}
            />
          </div>

          {/* Bottom level info */}
          <div
            style={{
              marginTop: 24,
              padding: "10px 16px",
              backgroundColor: "rgba(255,255,255,0.04)",
              borderRadius: 8,
              opacity: bottomOpacity,
              width: 520,
            }}
          >
            <span
              style={{
                fontSize: 15,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
              }}
            >
              <span style={{ color: THEME.colors.accentBlue, fontWeight: 700 }}>
                Level 1
              </span>{" "}
              (vector){" "}
              <span style={{ color: THEME.colors.accentPurple, fontWeight: 700 }}>
                Level 2
              </span>{" "}
              (matrix-vector){" "}
              <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>
                Level 3
              </span>{" "}
              (matrix-matrix)
            </span>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 50, width: 460 }}>
          <div
            style={{
              fontSize: 14,
              fontFamily: fontFamilyBody,
              fontWeight: 700,
              color: THEME.colors.accentBlue,
              marginBottom: 12,
              letterSpacing: "1px",
            }}
          >
            API OVERVIEW
          </div>

          {/* Code block */}
          <div
            style={{
              padding: "16px 20px",
              backgroundColor: THEME.colors.bgCode,
              borderRadius: 10,
              border: `1px solid rgba(255,255,255,0.08)`,
              width: 420,
            }}
          >
            {CODE_LINES.map((line, i) => {
              const lineDelay = (2 + i * 0.2) * fps;
              const lineOpacity = interpolate(
                frame - lineDelay,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              if (line.text === "") {
                return (
                  <div key={`line-${i}`} style={{ height: 10 }} />
                );
              }

              return (
                <div
                  key={`line-${i}`}
                  style={{
                    fontSize: 14,
                    fontFamily: fontFamilyCode,
                    color: line.color,
                    lineHeight: 1.7,
                    opacity: lineOpacity,
                    whiteSpace: "pre",
                  }}
                >
                  {line.text}
                </div>
              );
            })}
          </div>
        </div>
      }
    />
  );
};
