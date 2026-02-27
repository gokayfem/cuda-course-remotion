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

interface WorkflowStep {
  number: string;
  label: string;
  api: string;
  detail: string;
  color: string;
  delay: number;
}

const WORKFLOW_STEPS: WorkflowStep[] = [
  {
    number: "1",
    label: "Create Handle",
    api: "cudnnCreate",
    detail: "Initialize cuDNN context",
    color: THEME.colors.accentBlue,
    delay: 0,
  },
  {
    number: "2",
    label: "Describe Tensors",
    api: "cudnnSetTensor4dDescriptor",
    detail: "NCHW format, dimensions, type",
    color: THEME.colors.accentPurple,
    delay: 0.6,
  },
  {
    number: "3",
    label: "Describe Convolution",
    api: "cudnnSetConvolution2dDescriptor",
    detail: "padding, stride, dilation",
    color: THEME.colors.accentOrange,
    delay: 1.2,
  },
  {
    number: "4",
    label: "Find Best Algorithm",
    api: "cudnnGetConvolutionFwdAlgorithm",
    detail: "Auto-benchmark all options",
    color: THEME.colors.accentCyan,
    delay: 1.8,
  },
  {
    number: "5",
    label: "Allocate Workspace",
    api: "cudnnGetConvolutionFwdWorkspaceSize",
    detail: "Some algorithms need temp memory",
    color: THEME.colors.accentYellow,
    delay: 2.4,
  },
  {
    number: "6",
    label: "Execute!",
    api: "cudnnConvolutionForward",
    detail: "Run the optimized convolution",
    color: THEME.colors.nvidiaGreen,
    delay: 3.0,
  },
];

interface OpItem {
  name: string;
  detail: string;
}

const SUPPORTED_OPS: OpItem[] = [
  { name: "Convolution", detail: "forward + backward" },
  { name: "Pooling", detail: "max, average" },
  { name: "BatchNorm", detail: "LayerNorm" },
  { name: "Activation", detail: "ReLU, sigmoid, tanh" },
  { name: "Softmax", detail: "forward + backward" },
  { name: "Dropout", detail: "training mode" },
];

export const M7S06_CudnnIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={7}
      leftWidth="55%"
      left={
        <div style={{ width: 620 }}>
          <SlideTitle
            title="cuDNN -- Deep Learning Primitives"
            subtitle="6-step workflow for GPU-accelerated convolutions"
          />

          <div style={{ marginTop: 4, width: 600 }}>
            {WORKFLOW_STEPS.map((step, i) => {
              const stepDelay = (1.5 + step.delay) * fps;
              const stepSpring = spring({
                frame: frame - stepDelay,
                fps,
                config: { damping: 200, stiffness: 100 },
              });
              const stepOpacity = interpolate(stepSpring, [0, 1], [0, 1]);
              const stepX = interpolate(stepSpring, [0, 1], [-30, 0]);

              // Arrow between steps
              const arrowDelay = stepDelay + 0.3 * fps;
              const arrowOpacity =
                i < WORKFLOW_STEPS.length - 1
                  ? interpolate(
                      frame - arrowDelay,
                      [0, 0.2 * fps],
                      [0, 0.5],
                      { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                    )
                  : 0;

              return (
                <div key={`step-${i}`}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 12,
                      opacity: stepOpacity,
                      transform: `translateX(${stepX}px)`,
                      marginBottom: 2,
                      width: 580,
                    }}
                  >
                    {/* Step number circle */}
                    <div
                      style={{
                        width: 28,
                        height: 28,
                        borderRadius: 14,
                        backgroundColor: `${step.color}30`,
                        border: `2px solid ${step.color}`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 14,
                        fontFamily: fontFamilyBody,
                        fontWeight: 700,
                        color: step.color,
                        flexShrink: 0,
                      }}
                    >
                      {step.number}
                    </div>

                    {/* Step content */}
                    <div style={{ flex: 1 }}>
                      <div
                        style={{
                          display: "flex",
                          alignItems: "baseline",
                          gap: 10,
                          width: 520,
                        }}
                      >
                        <span
                          style={{
                            fontSize: 16,
                            fontFamily: fontFamilyBody,
                            fontWeight: 700,
                            color: step.color,
                          }}
                        >
                          {step.label}
                        </span>
                        <span
                          style={{
                            fontSize: 13,
                            fontFamily: fontFamilyCode,
                            color: THEME.colors.textMuted,
                          }}
                        >
                          {step.api}
                        </span>
                      </div>
                      <div
                        style={{
                          fontSize: 13,
                          fontFamily: fontFamilyBody,
                          color: THEME.colors.textSecondary,
                          marginTop: 2,
                        }}
                      >
                        {step.detail}
                      </div>
                    </div>
                  </div>

                  {/* Connector arrow */}
                  {i < WORKFLOW_STEPS.length - 1 && (
                    <div
                      style={{
                        marginLeft: 13,
                        width: 2,
                        height: 14,
                        backgroundColor: THEME.colors.textMuted,
                        opacity: arrowOpacity,
                        marginBottom: 2,
                      }}
                    />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 50, width: 420 }}>
          <div
            style={{
              fontSize: 14,
              fontFamily: fontFamilyBody,
              fontWeight: 700,
              color: THEME.colors.nvidiaGreen,
              marginBottom: 16,
              letterSpacing: "1px",
            }}
          >
            SUPPORTED OPERATIONS
          </div>

          {SUPPORTED_OPS.map((op, i) => {
            const opDelay = (5 + i * 0.4) * fps;
            const opSpring = spring({
              frame: frame - opDelay,
              fps,
              config: { damping: 200 },
            });
            const opOpacity = interpolate(opSpring, [0, 1], [0, 1]);
            const opX = interpolate(opSpring, [0, 1], [-15, 0]);

            return (
              <div
                key={`op-${i}`}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  marginBottom: 12,
                  opacity: opOpacity,
                  transform: `translateX(${opX}px)`,
                  width: 400,
                }}
              >
                <div
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: 4,
                    backgroundColor: THEME.colors.nvidiaGreen,
                    flexShrink: 0,
                  }}
                />
                <span
                  style={{
                    fontSize: 18,
                    fontFamily: fontFamilyBody,
                    fontWeight: 600,
                    color: THEME.colors.textPrimary,
                  }}
                >
                  {op.name}
                </span>
                <span
                  style={{
                    fontSize: 14,
                    fontFamily: fontFamilyBody,
                    color: THEME.colors.textMuted,
                  }}
                >
                  ({op.detail})
                </span>
              </div>
            );
          })}
        </div>
      }
    />
  );
};
