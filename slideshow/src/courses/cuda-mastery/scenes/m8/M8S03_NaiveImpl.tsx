import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

const GRID_SIZE = 6;
const CELL_SIZE = 36;
const CELL_GAP = 3;

export const M8S03_NaiveImpl: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Animate which row/col is being highlighted
  const animStart = 3 * fps;
  const elapsed = Math.max(0, frame - animStart);
  const cycleLen = 2 * fps;
  const cycleIdx = Math.min(Math.floor(elapsed / cycleLen), 5);
  const cycleProgress = (elapsed % cycleLen) / cycleLen;

  // Highlight row 2 of A, cycle through columns of B
  const highlightRow = 2;
  const highlightCol = cycleIdx;

  const dotProgress = interpolate(
    cycleProgress,
    [0, 0.3, 0.7, 1],
    [0, 0, 1, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const showDot = frame >= animStart;

  const perfOpacity = interpolate(
    frame - 10 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const memLabelOpacity = interpolate(
    frame - 6 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const diagramOpacity = interpolate(
    frame - 2 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const gridW = GRID_SIZE * (CELL_SIZE + CELL_GAP) - CELL_GAP;

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={8}
      leftWidth="50%"
      left={
        <div style={{ width: 560 }}>
          <SlideTitle
            title="The Naive Implementation"
            subtitle="One thread per output element"
          />
          <CodeBlock
            delay={0.5 * fps}
            title="matmul_naive.cu"
            fontSize={14}
            code={`__global__ void matmul_naive(
    float* A, float* B, float* C,
    int M, int N, int K) {
  int row = blockIdx.y * blockDim.y
          + threadIdx.y;
  int col = blockIdx.x * blockDim.x
          + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
      sum += A[row*K + k] * B[k*N + col];
    C[row * N + col] = sum;
  }
}`}
            highlightLines={[9, 10, 11]}
          />
        </div>
      }
      right={
        <div style={{ paddingTop: 40, width: 520 }}>
          {/* Diagram title */}
          <div
            style={{
              fontSize: 18,
              fontWeight: 700,
              color: THEME.colors.textPrimary,
              fontFamily: fontFamilyBody,
              marginBottom: 16,
              opacity: diagramOpacity,
            }}
          >
            One Thread Computing C[{highlightRow}][{showDot ? highlightCol : 0}]
          </div>

          {/* Matrix grids side by side */}
          <div
            style={{
              display: "flex",
              gap: 24,
              alignItems: "flex-start",
              opacity: diagramOpacity,
            }}
          >
            {/* Matrix A */}
            <div>
              <div
                style={{
                  fontSize: 14,
                  color: THEME.colors.accentBlue,
                  fontFamily: fontFamilyBody,
                  fontWeight: 700,
                  marginBottom: 6,
                  textAlign: "center",
                }}
              >
                A (row)
              </div>
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  width: gridW,
                  gap: CELL_GAP,
                }}
              >
                {Array.from({ length: GRID_SIZE * GRID_SIZE }).map((_, idx) => {
                  const r = Math.floor(idx / GRID_SIZE);
                  const isHighlight = showDot && r === highlightRow;
                  return (
                    <div
                      key={idx}
                      style={{
                        width: CELL_SIZE,
                        height: CELL_SIZE,
                        borderRadius: 4,
                        backgroundColor: isHighlight
                          ? `${THEME.colors.accentBlue}40`
                          : "rgba(255,255,255,0.04)",
                        border: `1.5px solid ${
                          isHighlight
                            ? THEME.colors.accentBlue
                            : "rgba(255,255,255,0.08)"
                        }`,
                      }}
                    />
                  );
                })}
              </div>
            </div>

            {/* Matrix B */}
            <div>
              <div
                style={{
                  fontSize: 14,
                  color: THEME.colors.accentPurple,
                  fontFamily: fontFamilyBody,
                  fontWeight: 700,
                  marginBottom: 6,
                  textAlign: "center",
                }}
              >
                B (col)
              </div>
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  width: gridW,
                  gap: CELL_GAP,
                }}
              >
                {Array.from({ length: GRID_SIZE * GRID_SIZE }).map((_, idx) => {
                  const c = idx % GRID_SIZE;
                  const isHighlight = showDot && c === highlightCol;
                  return (
                    <div
                      key={idx}
                      style={{
                        width: CELL_SIZE,
                        height: CELL_SIZE,
                        borderRadius: 4,
                        backgroundColor: isHighlight
                          ? `${THEME.colors.accentPurple}40`
                          : "rgba(255,255,255,0.04)",
                        border: `1.5px solid ${
                          isHighlight
                            ? THEME.colors.accentPurple
                            : "rgba(255,255,255,0.08)"
                        }`,
                      }}
                    />
                  );
                })}
              </div>
            </div>
          </div>

          {/* Arrow and C cell */}
          {showDot && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                marginTop: 16,
                opacity: dotProgress,
              }}
            >
              <div
                style={{
                  fontSize: 20,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                }}
              >
                dot product
              </div>
              <div
                style={{
                  fontSize: 20,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyBody,
                }}
              >
                {"\u2192"}
              </div>
              <div
                style={{
                  width: 48,
                  height: 48,
                  borderRadius: 6,
                  backgroundColor: `${THEME.colors.nvidiaGreen}50`,
                  border: `2px solid ${THEME.colors.nvidiaGreen}`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 14,
                  fontWeight: 700,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyCode,
                }}
              >
                C[{highlightRow}][{highlightCol}]
              </div>
            </div>
          )}

          {/* Memory reads label */}
          <div
            style={{
              marginTop: 20,
              padding: "10px 16px",
              backgroundColor: "rgba(255,82,82,0.10)",
              border: `1px solid ${THEME.colors.accentRed}40`,
              borderRadius: 8,
              opacity: memLabelOpacity,
              width: 420,
            }}
          >
            <span
              style={{
                fontSize: 16,
                color: THEME.colors.accentRed,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              2K global memory reads per output element
            </span>
          </div>

          {/* Performance label */}
          <div
            style={{
              marginTop: 16,
              padding: "12px 18px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: perfOpacity,
              width: 420,
            }}
          >
            <span
              style={{
                fontSize: 16,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              Performance:{" "}
              <span style={{ color: THEME.colors.accentRed }}>
                ~50 GFLOPS
              </span>
              {" "}(~0.3% of A100 peak)
            </span>
          </div>
        </div>
      }
    />
  );
};
