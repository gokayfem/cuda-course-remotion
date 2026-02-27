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

interface BlockDef {
  readonly label: string;
  readonly sublabel: string;
  readonly type: "custom" | "cublas";
}

const BLOCKS: readonly BlockDef[] = [
  { label: "Input", sublabel: "embeddings", type: "cublas" },
  { label: "LayerNorm", sublabel: "custom kernel", type: "custom" },
  { label: "Q, K, V\nProjection", sublabel: "cuBLAS GEMM", type: "cublas" },
  { label: "Attention", sublabel: "custom kernel", type: "custom" },
  { label: "Output\nProjection", sublabel: "cuBLAS GEMM", type: "cublas" },
  { label: "+ Residual", sublabel: "fused add", type: "custom" },
  { label: "LayerNorm", sublabel: "custom kernel", type: "custom" },
  { label: "FFN", sublabel: "cuBLAS GEMM", type: "cublas" },
  { label: "+ Residual", sublabel: "fused add", type: "custom" },
  { label: "Output", sublabel: "to next layer", type: "cublas" },
];

const BLOCK_WIDTH = 130;
const BLOCK_HEIGHT = 72;
const ARROW_GAP = 28;

export const M9S02_TransformerAnatomy: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const totalWidth = BLOCKS.length * BLOCK_WIDTH + (BLOCKS.length - 1) * ARROW_GAP;
  const startX = (1920 - 2 * 72 - totalWidth) / 2;

  return (
    <SlideLayout variant="gradient" moduleNumber={9}>
      <SlideTitle
        title="Inside a Transformer Block"
        subtitle="The full data flow from input to output, with kernel types labeled"
      />

      <div style={{ flex: 1, position: "relative", width: 1776 }}>
        {/* Flow diagram */}
        <div
          style={{
            position: "absolute",
            top: 40,
            left: startX,
            display: "flex",
            alignItems: "center",
            gap: 0,
          }}
        >
          {BLOCKS.map((block, i) => {
            const blockDelay = 0.8 * fps + i * 0.25 * fps;
            const blockSpring = spring({
              frame: frame - blockDelay,
              fps,
              config: { damping: 200 },
            });
            const opacity = interpolate(blockSpring, [0, 1], [0, 1]);
            const scale = interpolate(blockSpring, [0, 1], [0.85, 1]);

            const isCustom = block.type === "custom";
            const color = isCustom
              ? THEME.colors.nvidiaGreen
              : THEME.colors.textMuted;
            const bgColor = isCustom
              ? "rgba(118,185,0,0.12)"
              : "rgba(255,255,255,0.05)";
            const borderColor = isCustom
              ? `${THEME.colors.nvidiaGreen}80`
              : "rgba(255,255,255,0.15)";

            const highlightTime = 3.5 * fps + i * 0.4 * fps;
            const highlightPulse = interpolate(
              frame - highlightTime,
              [0, 0.3 * fps, 0.6 * fps],
              [0, 0.3, 0],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );

            return (
              <React.Fragment key={i}>
                <div
                  style={{
                    width: BLOCK_WIDTH,
                    height: BLOCK_HEIGHT,
                    backgroundColor: bgColor,
                    border: `2px solid ${borderColor}`,
                    borderRadius: 10,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    opacity,
                    transform: `scale(${scale})`,
                    flexShrink: 0,
                    boxShadow: isCustom
                      ? `0 0 ${12 + highlightPulse * 20}px rgba(118,185,0,${0.15 + highlightPulse})`
                      : "none",
                  }}
                >
                  <span
                    style={{
                      color,
                      fontSize: 14,
                      fontWeight: 700,
                      fontFamily: fontFamilyBody,
                      textAlign: "center",
                      lineHeight: 1.2,
                      whiteSpace: "pre-line",
                    }}
                  >
                    {block.label}
                  </span>
                  <span
                    style={{
                      color: THEME.colors.textMuted,
                      fontSize: 11,
                      fontFamily: fontFamilyCode,
                      marginTop: 4,
                      textAlign: "center",
                    }}
                  >
                    {block.sublabel}
                  </span>
                </div>

                {/* Arrow between blocks */}
                {i < BLOCKS.length - 1 && (
                  <div
                    style={{
                      width: ARROW_GAP,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      flexShrink: 0,
                      opacity: interpolate(
                        frame - (blockDelay + 0.15 * fps),
                        [0, 0.2 * fps],
                        [0, 0.6],
                        {
                          extrapolateLeft: "clamp",
                          extrapolateRight: "clamp",
                        }
                      ),
                    }}
                  >
                    <span
                      style={{
                        color: THEME.colors.textMuted,
                        fontSize: 18,
                        fontFamily: fontFamilyBody,
                      }}
                    >
                      {"\u2192"}
                    </span>
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Sequential highlight indicator */}
        {(() => {
          const sweepProgress = interpolate(
            frame - 3.5 * fps,
            [0, BLOCKS.length * 0.4 * fps],
            [0, BLOCKS.length - 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          const activeIdx = Math.floor(sweepProgress);
          const activeBlock = BLOCKS[activeIdx];
          const descDelay = 3.5 * fps;
          const descOpacity = interpolate(
            frame - descDelay,
            [0, 0.3 * fps],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );

          return activeBlock ? (
            <div
              style={{
                position: "absolute",
                top: 140,
                left: startX,
                width: totalWidth,
                textAlign: "center",
                opacity: descOpacity,
              }}
            >
              <span
                style={{
                  fontSize: 22,
                  fontFamily: fontFamilyBody,
                  color:
                    activeBlock.type === "custom"
                      ? THEME.colors.nvidiaGreen
                      : THEME.colors.textSecondary,
                  fontWeight: 600,
                }}
              >
                {activeBlock.label.replace("\n", " ")}
              </span>
              <span
                style={{
                  fontSize: 18,
                  fontFamily: fontFamilyBody,
                  color: THEME.colors.textMuted,
                  marginLeft: 16,
                }}
              >
                {activeBlock.type === "custom"
                  ? "Custom CUDA kernel"
                  : "cuBLAS GEMM call"}
              </span>
            </div>
          ) : null;
        })()}

        {/* Legend */}
        <div
          style={{
            position: "absolute",
            top: 200,
            left: startX,
            width: totalWidth,
            display: "flex",
            justifyContent: "center",
            gap: 48,
            opacity: interpolate(
              frame - 6 * fps,
              [0, 0.5 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div
              style={{
                width: 20,
                height: 20,
                borderRadius: 4,
                backgroundColor: "rgba(118,185,0,0.2)",
                border: `2px solid ${THEME.colors.nvidiaGreen}80`,
              }}
            />
            <span
              style={{
                fontSize: 18,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              Custom CUDA kernels
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div
              style={{
                width: 20,
                height: 20,
                borderRadius: 4,
                backgroundColor: "rgba(255,255,255,0.05)",
                border: "2px solid rgba(255,255,255,0.15)",
              }}
            />
            <span
              style={{
                fontSize: 18,
                color: THEME.colors.textMuted,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              cuBLAS GEMM calls
            </span>
          </div>
        </div>

        {/* Bottom insight */}
        <div
          style={{
            position: "absolute",
            bottom: 20,
            left: 0,
            right: 0,
            textAlign: "center",
            opacity: interpolate(
              frame - 7 * fps,
              [0, 0.5 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          <div
            style={{
              display: "inline-block",
              padding: "14px 32px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              borderLeft: `4px solid ${THEME.colors.nvidiaGreen}`,
            }}
          >
            <span
              style={{
                fontSize: 20,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
              }}
            >
              <span
                style={{
                  color: THEME.colors.nvidiaGreen,
                  fontWeight: 700,
                }}
              >
                Green = custom CUDA kernels
              </span>
              {" that we will build. "}
              <span style={{ color: THEME.colors.textMuted, fontWeight: 600 }}>
                Gray = cuBLAS GEMM
              </span>
              {" calls handled by the library."}
            </span>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
