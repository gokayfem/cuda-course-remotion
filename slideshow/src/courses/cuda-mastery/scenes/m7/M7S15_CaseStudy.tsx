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
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

type Layer = {
  label: string;
  size: string;
  color: string;
};

const layers: Layer[] = [
  { label: "Input", size: "784", color: THEME.colors.textSecondary },
  { label: "Linear", size: "512", color: THEME.colors.accentBlue },
  { label: "ReLU", size: "", color: THEME.colors.accentRed },
  { label: "Linear", size: "10", color: THEME.colors.accentBlue },
  { label: "Softmax", size: "", color: THEME.colors.nvidiaGreen },
];

type ImplStep = {
  label: string;
  code: string;
  color: string;
  library: string;
};

const implSteps: ImplStep[] = [
  {
    label: "Layer 1",
    code: "cublasSgemm(handle, ..., X, W1, ..., Z1)",
    color: THEME.colors.accentBlue,
    library: "cuBLAS",
  },
  {
    label: "ReLU",
    code: "custom kernel: z1[i] = max(0, z1[i])",
    color: THEME.colors.accentRed,
    library: "Custom",
  },
  {
    label: "Layer 2",
    code: "cublasSgemm(handle, ..., Z1, W2, ..., Z2)",
    color: THEME.colors.accentBlue,
    library: "cuBLAS",
  },
  {
    label: "Softmax",
    code: "cudnnSoftmaxForward(handle, ..., Z2, Y)",
    color: THEME.colors.nvidiaGreen,
    library: "cuDNN",
  },
];

const BAR_MAX_WIDTH = 600;
const NAIVE_MS = 45;
const CUBLAS_MS = 0.8;
const MAX_MS = 50;

export const M7S15_CaseStudy: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const perfOpacity = interpolate(
    frame - 10 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const perfProgress = interpolate(
    frame - 10.5 * fps,
    [0, 1 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={7}
      leftWidth="45%"
      left={
        <div style={{ width: 680 }}>
          <SlideTitle title="Case Study: 2-Layer MLP" />

          {/* Network diagram */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 0,
              marginBottom: 28,
              justifyContent: "center",
            }}
          >
            {layers.map((layer, i) => {
              const layerDelay = 0.8 * fps + i * 0.6 * fps;
              const layerSpring = spring({
                frame: frame - layerDelay,
                fps,
                config: { damping: 200 },
              });
              const layerOpacity = interpolate(layerSpring, [0, 1], [0, 1]);

              const activeDelay = 5 * fps + i * 1.2 * fps;
              const isActive = frame > activeDelay && frame < activeDelay + 1 * fps;
              const glowIntensity = isActive ? 0.3 : 0;

              return (
                <React.Fragment key={i}>
                  <div
                    style={{
                      padding: "12px 16px",
                      backgroundColor: `${layer.color}15`,
                      border: `2px solid ${layer.color}60`,
                      borderRadius: 10,
                      textAlign: "center",
                      opacity: layerOpacity,
                      boxShadow: glowIntensity
                        ? `0 0 16px ${layer.color}${Math.round(glowIntensity * 255).toString(16).padStart(2, "0")}`
                        : "none",
                      minWidth: 80,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 14,
                        fontWeight: 700,
                        color: layer.color,
                        fontFamily: fontFamilyBody,
                      }}
                    >
                      {layer.label}
                    </div>
                    {layer.size && (
                      <div
                        style={{
                          fontSize: 12,
                          color: THEME.colors.textMuted,
                          fontFamily: fontFamilyCode,
                          marginTop: 2,
                        }}
                      >
                        ({layer.size})
                      </div>
                    )}
                  </div>
                  {i < layers.length - 1 && (
                    <div
                      style={{
                        fontSize: 16,
                        color: THEME.colors.textMuted,
                        padding: "0 6px",
                        opacity: layerOpacity,
                      }}
                    >
                      {"\u2192"}
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>

          {/* Performance bar */}
          <div
            style={{
              marginTop: 16,
              opacity: perfOpacity,
              padding: "16px 20px",
              backgroundColor: "rgba(255,255,255,0.03)",
              borderRadius: 10,
              border: `1px solid rgba(255,255,255,0.08)`,
              width: 660,
            }}
          >
            <div
              style={{
                fontSize: 16,
                fontWeight: 700,
                color: THEME.colors.accentCyan,
                fontFamily: fontFamilyBody,
                marginBottom: 14,
              }}
            >
              Performance Comparison
            </div>

            {/* Naive bar */}
            <div style={{ marginBottom: 12 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: 4,
                  width: BAR_MAX_WIDTH,
                }}
              >
                <span
                  style={{
                    fontSize: 14,
                    color: THEME.colors.textPrimary,
                    fontFamily: fontFamilyBody,
                  }}
                >
                  Naive kernels
                </span>
                <span
                  style={{
                    fontSize: 14,
                    color: THEME.colors.accentRed,
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                  }}
                >
                  {Math.round(NAIVE_MS * perfProgress)} ms
                </span>
              </div>
              <div
                style={{
                  width: BAR_MAX_WIDTH,
                  height: 22,
                  backgroundColor: "rgba(255,255,255,0.04)",
                  borderRadius: 6,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: (NAIVE_MS / MAX_MS) * BAR_MAX_WIDTH * perfProgress,
                    backgroundColor: THEME.colors.accentRed + "CC",
                    borderRadius: 6,
                  }}
                />
              </div>
            </div>

            {/* cuBLAS bar */}
            <div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: 4,
                  width: BAR_MAX_WIDTH,
                }}
              >
                <span
                  style={{
                    fontSize: 14,
                    color: THEME.colors.textPrimary,
                    fontFamily: fontFamilyBody,
                  }}
                >
                  cuBLAS MLP
                </span>
                <span
                  style={{
                    fontSize: 14,
                    color: THEME.colors.nvidiaGreen,
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                  }}
                >
                  {(CUBLAS_MS * perfProgress).toFixed(1)} ms
                </span>
              </div>
              <div
                style={{
                  width: BAR_MAX_WIDTH,
                  height: 22,
                  backgroundColor: "rgba(255,255,255,0.04)",
                  borderRadius: 6,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: Math.max(
                      (CUBLAS_MS / MAX_MS) * BAR_MAX_WIDTH * perfProgress,
                      4
                    ),
                    backgroundColor: THEME.colors.nvidiaGreen + "CC",
                    borderRadius: 6,
                  }}
                />
              </div>
            </div>

            {/* Speedup label */}
            <div
              style={{
                marginTop: 10,
                fontSize: 18,
                fontWeight: 700,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                textAlign: "center",
              }}
            >
              56x faster
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ width: 520, marginTop: 60 }}>
          <div
            style={{
              fontSize: 20,
              fontWeight: 700,
              color: THEME.colors.accentCyan,
              fontFamily: fontFamilyBody,
              marginBottom: 20,
              opacity: interpolate(
                frame - 5 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            Implementation Breakdown
          </div>

          {implSteps.map((step, i) => {
            const stepDelay = 5.5 * fps + i * 1 * fps;
            const stepSpring = spring({
              frame: frame - stepDelay,
              fps,
              config: { damping: 200 },
            });
            const stepOpacity = interpolate(stepSpring, [0, 1], [0, 1]);
            const stepX = interpolate(stepSpring, [0, 1], [15, 0]);

            return (
              <div
                key={i}
                style={{
                  padding: "12px 16px",
                  backgroundColor: `${step.color}08`,
                  borderLeft: `4px solid ${step.color}`,
                  borderRadius: 8,
                  marginBottom: 12,
                  opacity: stepOpacity,
                  transform: `translateX(${stepX}px)`,
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 4,
                  }}
                >
                  <span
                    style={{
                      fontSize: 16,
                      fontWeight: 700,
                      color: step.color,
                      fontFamily: fontFamilyBody,
                    }}
                  >
                    {step.label}
                  </span>
                  <span
                    style={{
                      fontSize: 12,
                      color: step.color,
                      fontFamily: fontFamilyBody,
                      padding: "2px 8px",
                      backgroundColor: `${step.color}15`,
                      borderRadius: 4,
                    }}
                  >
                    {step.library}
                  </span>
                </div>
                <div
                  style={{
                    fontSize: 13,
                    color: THEME.colors.textCode,
                    fontFamily: fontFamilyCode,
                    lineHeight: 1.5,
                  }}
                >
                  {step.code}
                </div>
              </div>
            );
          })}
        </div>
      }
    />
  );
};
