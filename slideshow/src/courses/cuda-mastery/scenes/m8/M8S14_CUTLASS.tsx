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

type ArchLayer = {
  label: string;
  sublabel: string;
  color: string;
};

const layers: ArchLayer[] = [
  {
    label: "Your Code",
    sublabel: "Custom fused epilogues, data types, layouts",
    color: THEME.colors.nvidiaGreen,
  },
  {
    label: "CUTLASS Templates",
    sublabel: "Configurable tile sizes, warp arrangements, pipeline stages",
    color: THEME.colors.accentBlue,
  },
  {
    label: "Tensor Cores / CUDA Cores",
    sublabel: "Hardware execution units",
    color: THEME.colors.textMuted,
  },
];

export const M8S14_CUTLASS: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const bottomOpacity = interpolate(
    frame - 10 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={8}
      leftWidth="50%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle
            title="CUTLASS -- Customizable MatMul Templates"
            subtitle="Near-cuBLAS performance with full customization"
          />

          {/* Architecture layers */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 0,
              width: 640,
              marginTop: 16,
            }}
          >
            {layers.map((layer, i) => {
              const layerDelay = 1.5 * fps + i * 1.5 * fps;
              const layerSpring = spring({
                frame: frame - layerDelay,
                fps,
                config: { damping: 200 },
              });
              const layerOpacity = interpolate(layerSpring, [0, 1], [0, 1]);
              const layerY = interpolate(layerSpring, [0, 1], [20, 0]);

              const showArrow = i < layers.length - 1;
              const arrowDelay = layerDelay + 0.8 * fps;
              const arrowOpacity = interpolate(
                frame - arrowDelay,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              return (
                <React.Fragment key={layer.label}>
                  <div
                    style={{
                      padding: "18px 24px",
                      backgroundColor: `${layer.color}10`,
                      border: `2px solid ${layer.color}50`,
                      borderRadius: 10,
                      opacity: layerOpacity,
                      transform: `translateY(${layerY}px)`,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 20,
                        fontWeight: 700,
                        color: layer.color,
                        fontFamily: fontFamilyBody,
                        marginBottom: 4,
                      }}
                    >
                      {layer.label}
                    </div>
                    <div
                      style={{
                        fontSize: 14,
                        color: THEME.colors.textSecondary,
                        fontFamily: fontFamilyBody,
                        lineHeight: 1.4,
                      }}
                    >
                      {layer.sublabel}
                    </div>
                  </div>

                  {/* Arrow */}
                  {showArrow && (
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "center",
                        opacity: arrowOpacity,
                        height: 32,
                        alignItems: "center",
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          flexDirection: "column",
                          alignItems: "center",
                        }}
                      >
                        <div
                          style={{
                            width: 2,
                            height: 16,
                            backgroundColor: THEME.colors.textMuted + "60",
                          }}
                        />
                        <div
                          style={{
                            fontSize: 14,
                            color: THEME.colors.textMuted,
                          }}
                        >
                          {i === 0 ? "customization" : "execution"}
                        </div>
                      </div>
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>
      }
      right={
        <div style={{ width: 480, marginTop: 80 }}>
          <BulletPoint
            text="NVIDIA's open-source GEMM template library"
            index={0}
            delay={3 * fps}
          />
          <BulletPoint
            text="C++ templates: configure tile sizes, warp arrangements, stages"
            index={1}
            delay={3 * fps}
          />
          <BulletPoint
            text="Near-cuBLAS performance with full customization"
            index={2}
            delay={3 * fps}
            highlight
          />
          <BulletPoint
            text="Supports fused epilogues (bias, activation, scaling)"
            index={3}
            delay={3 * fps}
          />

          {/* Bottom insight */}
          <div
            style={{
              marginTop: 32,
              padding: "14px 20px",
              backgroundColor: "rgba(79,195,247,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.accentBlue}30`,
              opacity: bottomOpacity,
            }}
          >
            <div
              style={{
                fontSize: 16,
                color: THEME.colors.accentBlue,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
                lineHeight: 1.5,
              }}
            >
              CUTLASS = when you need cuBLAS performance with custom behavior
            </div>
          </div>
        </div>
      }
    />
  );
};
