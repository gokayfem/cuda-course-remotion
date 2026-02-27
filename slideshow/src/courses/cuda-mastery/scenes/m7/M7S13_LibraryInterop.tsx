import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint } from "../../../../components/AnimatedText";
import { fontFamilyBody } from "../../../../styles/fonts";

type PipelineStep = {
  label: string;
  library: string;
  color: string;
};

const pipelineSteps: PipelineStep[] = [
  { label: "Initialize weights", library: "cuRAND", color: THEME.colors.accentYellow },
  { label: "Forward pass GEMM", library: "cuBLAS", color: THEME.colors.accentBlue },
  { label: "ReLU activation", library: "Custom kernel", color: THEME.colors.accentRed },
  { label: "BatchNorm", library: "cuDNN", color: THEME.colors.nvidiaGreen },
  { label: "Output GEMM", library: "cuBLAS", color: THEME.colors.accentBlue },
  { label: "Loss function", library: "Custom kernel", color: THEME.colors.accentRed },
];

export const M7S13_LibraryInterop: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={7}
      leftWidth="50%"
      left={
        <div style={{ width: 740 }}>
          <SlideTitle title="Mixing Libraries with Custom Kernels" />

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 0,
              alignItems: "flex-start",
            }}
          >
            {pipelineSteps.map((step, i) => {
              const stepDelay = 1 * fps + i * 1 * fps;
              const stepSpring = spring({
                frame: frame - stepDelay,
                fps,
                config: { damping: 200 },
              });
              const stepOpacity = interpolate(stepSpring, [0, 1], [0, 1]);
              const stepY = interpolate(stepSpring, [0, 1], [15, 0]);

              const showArrow = i < pipelineSteps.length - 1;
              const arrowOpacity = interpolate(
                frame - (stepDelay + 0.5 * fps),
                [0, 0.2 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              return (
                <React.Fragment key={i}>
                  {/* Step block */}
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 14,
                      padding: "10px 18px",
                      backgroundColor: `${step.color}10`,
                      border: `2px solid ${step.color}50`,
                      borderRadius: 10,
                      opacity: stepOpacity,
                      transform: `translateY(${stepY}px)`,
                      width: 520,
                    }}
                  >
                    {/* Step number */}
                    <div
                      style={{
                        width: 28,
                        height: 28,
                        borderRadius: 14,
                        backgroundColor: `${step.color}30`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 14,
                        fontWeight: 700,
                        color: step.color,
                        fontFamily: fontFamilyBody,
                        flexShrink: 0,
                      }}
                    >
                      {i + 1}
                    </div>

                    {/* Library badge */}
                    <div
                      style={{
                        padding: "3px 10px",
                        backgroundColor: `${step.color}20`,
                        borderRadius: 6,
                        fontSize: 13,
                        fontWeight: 700,
                        color: step.color,
                        fontFamily: fontFamilyBody,
                        flexShrink: 0,
                        minWidth: 110,
                        textAlign: "center",
                      }}
                    >
                      {step.library}
                    </div>

                    {/* Label */}
                    <div
                      style={{
                        fontSize: 16,
                        color: THEME.colors.textPrimary,
                        fontFamily: fontFamilyBody,
                      }}
                    >
                      {step.label}
                    </div>
                  </div>

                  {/* Arrow connector */}
                  {showArrow && (
                    <div
                      style={{
                        marginLeft: 36,
                        opacity: arrowOpacity,
                        display: "flex",
                        alignItems: "center",
                        height: 20,
                      }}
                    >
                      <div
                        style={{
                          width: 2,
                          height: 20,
                          backgroundColor: THEME.colors.textMuted + "60",
                        }}
                      />
                      <span
                        style={{
                          fontSize: 14,
                          color: THEME.colors.textMuted,
                          marginLeft: 8,
                        }}
                      >
                        {"\u2193"}
                      </span>
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>
      }
      right={
        <div style={{ width: 460, marginTop: 80 }}>
          <BulletPoint
            text="Libraries work on raw device pointers -- easy interop"
            index={0}
            delay={4 * fps}
          />
          <BulletPoint
            text="Use streams for concurrent library calls"
            index={1}
            delay={4 * fps}
          />
          <BulletPoint
            text="cuBLAS + custom kernels = best of both worlds"
            index={2}
            delay={4 * fps}
            highlight
          />
          <BulletPoint
            text="Profile: library calls often dominate runtime"
            index={3}
            delay={4 * fps}
          />
        </div>
      }
    />
  );
};
