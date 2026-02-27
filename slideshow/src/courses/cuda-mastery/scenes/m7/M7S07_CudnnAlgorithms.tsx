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

interface Algorithm {
  name: string;
  description: string;
  speed: number;
  memory: number;
  note: string;
  delay: number;
}

const ALGORITHMS: Algorithm[] = [
  {
    name: "IMPLICIT_GEMM",
    description: "Default, always works",
    speed: 0.5,
    memory: 0.3,
    note: "Safe fallback",
    delay: 0,
  },
  {
    name: "IMPLICIT_PRECOMP_GEMM",
    description: "Precomputed indices",
    speed: 0.65,
    memory: 0.4,
    note: "Better cache use",
    delay: 0.5,
  },
  {
    name: "GEMM",
    description: "im2col + matrix multiply",
    speed: 0.75,
    memory: 0.85,
    note: "Classic approach",
    delay: 1.0,
  },
  {
    name: "FFT",
    description: "Frequency domain",
    speed: 0.8,
    memory: 0.7,
    note: "Best for large filters",
    delay: 1.5,
  },
  {
    name: "WINOGRAD",
    description: "Minimal filtering",
    speed: 0.95,
    memory: 0.5,
    note: "Fastest for 3x3",
    delay: 2.0,
  },
];

const SPEED_BAR_MAX = 260;
const MEM_BAR_MAX = 120;
const ROW_HEIGHT = 72;

export const M7S07_CudnnAlgorithms: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // im2col diagram
  const diagramOpacity = interpolate(
    frame - 8 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Bottom tip
  const tipOpacity = interpolate(
    frame - 11 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="dark" moduleNumber={7}>
      <div style={{ width: 1776 }}>
        <SlideTitle
          title="cuDNN Convolution Algorithms"
          subtitle="5 strategies, auto-selected by cuDNN"
        />

        <div
          style={{
            display: "flex",
            gap: 48,
            marginTop: 4,
            width: 1776,
          }}
        >
          {/* Algorithm table */}
          <div style={{ width: 1000 }}>
            {/* Header */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 16,
                paddingBottom: 8,
                borderBottom: `1px solid rgba(255,255,255,0.1)`,
                marginBottom: 8,
                width: 960,
              }}
            >
              <div
                style={{
                  width: 220,
                  fontSize: 12,
                  fontFamily: fontFamilyBody,
                  fontWeight: 700,
                  color: THEME.colors.textMuted,
                  letterSpacing: "1px",
                }}
              >
                ALGORITHM
              </div>
              <div
                style={{
                  width: 180,
                  fontSize: 12,
                  fontFamily: fontFamilyBody,
                  fontWeight: 700,
                  color: THEME.colors.textMuted,
                  letterSpacing: "1px",
                }}
              >
                DESCRIPTION
              </div>
              <div
                style={{
                  width: 280,
                  fontSize: 12,
                  fontFamily: fontFamilyBody,
                  fontWeight: 700,
                  color: THEME.colors.nvidiaGreen,
                  letterSpacing: "1px",
                }}
              >
                SPEED
              </div>
              <div
                style={{
                  width: 140,
                  fontSize: 12,
                  fontFamily: fontFamilyBody,
                  fontWeight: 700,
                  color: THEME.colors.accentOrange,
                  letterSpacing: "1px",
                }}
              >
                MEMORY
              </div>
            </div>

            {/* Rows */}
            {ALGORITHMS.map((algo, i) => {
              const rowDelay = (1.5 + algo.delay) * fps;
              const rowSpring = spring({
                frame: frame - rowDelay,
                fps,
                config: { damping: 200, stiffness: 100 },
              });
              const rowOpacity = interpolate(rowSpring, [0, 1], [0, 1]);
              const rowX = interpolate(rowSpring, [0, 1], [-20, 0]);

              const speedWidth = interpolate(
                rowSpring,
                [0, 1],
                [0, algo.speed * SPEED_BAR_MAX]
              );
              const memWidth = interpolate(
                rowSpring,
                [0, 1],
                [0, algo.memory * MEM_BAR_MAX]
              );

              return (
                <div
                  key={`algo-${i}`}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 16,
                    height: ROW_HEIGHT,
                    opacity: rowOpacity,
                    transform: `translateX(${rowX}px)`,
                    padding: "0 8px",
                    backgroundColor:
                      i % 2 === 0 ? "rgba(255,255,255,0.02)" : "transparent",
                    borderRadius: 6,
                    width: 960,
                  }}
                >
                  {/* Name */}
                  <div style={{ width: 220 }}>
                    <div
                      style={{
                        fontSize: 14,
                        fontFamily: fontFamilyCode,
                        fontWeight: 700,
                        color: THEME.colors.accentCyan,
                      }}
                    >
                      {algo.name}
                    </div>
                    <div
                      style={{
                        fontSize: 11,
                        fontFamily: fontFamilyBody,
                        color: THEME.colors.textMuted,
                        marginTop: 2,
                      }}
                    >
                      {algo.note}
                    </div>
                  </div>

                  {/* Description */}
                  <div
                    style={{
                      width: 180,
                      fontSize: 13,
                      fontFamily: fontFamilyBody,
                      color: THEME.colors.textSecondary,
                    }}
                  >
                    {algo.description}
                  </div>

                  {/* Speed bar */}
                  <div style={{ width: 280 }}>
                    <div
                      style={{
                        width: speedWidth,
                        height: 16,
                        backgroundColor: THEME.colors.nvidiaGreen,
                        borderRadius: 4,
                        minWidth: 4,
                      }}
                    />
                  </div>

                  {/* Memory bar */}
                  <div style={{ width: 140 }}>
                    <div
                      style={{
                        width: memWidth,
                        height: 16,
                        backgroundColor: THEME.colors.accentOrange,
                        borderRadius: 4,
                        minWidth: 4,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {/* im2col diagram */}
          <div
            style={{
              width: 580,
              opacity: diagramOpacity,
              paddingTop: 8,
            }}
          >
            <div
              style={{
                fontSize: 14,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
                color: THEME.colors.accentPurple,
                marginBottom: 12,
                letterSpacing: "1px",
              }}
            >
              im2col TRANSFORMATION
            </div>

            {/* Input image */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 20,
                width: 560,
              }}
            >
              {/* Input */}
              <div style={{ textAlign: "center", width: 140 }}>
                <div
                  style={{
                    fontSize: 12,
                    fontFamily: fontFamilyBody,
                    color: THEME.colors.textMuted,
                    marginBottom: 6,
                  }}
                >
                  Input (4x4)
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(4, 28px)",
                    gap: 2,
                    justifyContent: "center",
                  }}
                >
                  {Array.from({ length: 16 }).map((_, idx) => (
                    <div
                      key={`in-${idx}`}
                      style={{
                        width: 28,
                        height: 28,
                        backgroundColor:
                          idx < 9 && (idx % 4 < 3) && Math.floor(idx / 4) < 3
                            ? `${THEME.colors.accentBlue}40`
                            : "rgba(255,255,255,0.05)",
                        border: `1px solid ${
                          idx < 9 && (idx % 4 < 3) && Math.floor(idx / 4) < 3
                            ? THEME.colors.accentBlue
                            : "rgba(255,255,255,0.1)"
                        }`,
                        borderRadius: 3,
                        fontSize: 10,
                        fontFamily: fontFamilyCode,
                        color: THEME.colors.textMuted,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      {idx}
                    </div>
                  ))}
                </div>
              </div>

              {/* Arrow */}
              <div
                style={{
                  fontSize: 24,
                  color: THEME.colors.textMuted,
                  width: 40,
                  textAlign: "center",
                }}
              >
                {">"}
              </div>

              {/* Column matrix */}
              <div style={{ textAlign: "center", width: 140 }}>
                <div
                  style={{
                    fontSize: 12,
                    fontFamily: fontFamilyBody,
                    color: THEME.colors.textMuted,
                    marginBottom: 6,
                  }}
                >
                  Patches (Kh*Kw x N)
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(4, 28px)",
                    gap: 2,
                    justifyContent: "center",
                  }}
                >
                  {Array.from({ length: 36 }).map((_, idx) => (
                    <div
                      key={`col-${idx}`}
                      style={{
                        width: 28,
                        height: 14,
                        backgroundColor: `${THEME.colors.accentPurple}30`,
                        border: `1px solid ${THEME.colors.accentPurple}60`,
                        borderRadius: 2,
                      }}
                    />
                  ))}
                </div>
              </div>

              {/* Arrow */}
              <div
                style={{
                  fontSize: 24,
                  color: THEME.colors.textMuted,
                  width: 40,
                  textAlign: "center",
                }}
              >
                {">"}
              </div>

              {/* GEMM */}
              <div style={{ textAlign: "center", width: 100 }}>
                <div
                  style={{
                    fontSize: 12,
                    fontFamily: fontFamilyBody,
                    color: THEME.colors.textMuted,
                    marginBottom: 6,
                  }}
                >
                  GEMM
                </div>
                <div
                  style={{
                    padding: "16px 12px",
                    backgroundColor: `${THEME.colors.nvidiaGreen}20`,
                    border: `2px solid ${THEME.colors.nvidiaGreen}`,
                    borderRadius: 8,
                    fontSize: 14,
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                    color: THEME.colors.nvidiaGreen,
                  }}
                >
                  W x col
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom tip */}
        <div
          style={{
            marginTop: 16,
            padding: "12px 18px",
            backgroundColor: "rgba(118,185,0,0.08)",
            borderRadius: 10,
            border: `1px solid ${THEME.colors.nvidiaGreen}40`,
            opacity: tipOpacity,
            width: 700,
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
            Let cuDNN{" "}
            <span style={{ color: THEME.colors.nvidiaGreen }}>auto-select</span>:
            it benchmarks all algorithms and picks the fastest
          </span>
        </div>
      </div>
    </SlideLayout>
  );
};
