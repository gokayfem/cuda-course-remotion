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
import { fontFamilyBody } from "../../../../styles/fonts";

type LayerItem = {
  label: string;
  color: string;
};

type StackLayer = {
  name: string;
  items: LayerItem[];
  layerColor: string;
};

const layers: StackLayer[] = [
  {
    name: "ML Frameworks",
    layerColor: THEME.colors.textMuted,
    items: [
      { label: "PyTorch", color: THEME.colors.textMuted },
      { label: "JAX", color: THEME.colors.textMuted },
      { label: "TensorFlow", color: THEME.colors.textMuted },
    ],
  },
  {
    name: "Compilers",
    layerColor: THEME.colors.accentPurple,
    items: [
      { label: "torch.compile", color: THEME.colors.accentPurple },
      { label: "XLA", color: THEME.colors.accentPurple },
      { label: "TVM", color: THEME.colors.accentPurple },
    ],
  },
  {
    name: "Kernel Languages",
    layerColor: THEME.colors.accentBlue,
    items: [
      { label: "Triton", color: THEME.colors.accentBlue },
      { label: "CUDA C", color: THEME.colors.accentBlue },
      { label: "CUTLASS", color: THEME.colors.accentBlue },
    ],
  },
  {
    name: "Libraries",
    layerColor: THEME.colors.nvidiaGreen,
    items: [
      { label: "cuBLAS", color: THEME.colors.nvidiaGreen },
      { label: "cuDNN", color: THEME.colors.nvidiaGreen },
      { label: "NCCL", color: THEME.colors.nvidiaGreen },
      { label: "Flash Attention", color: THEME.colors.nvidiaGreen },
    ],
  },
  {
    name: "Hardware",
    layerColor: THEME.colors.textSecondary,
    items: [
      { label: "Tensor Cores", color: THEME.colors.textSecondary },
      { label: "CUDA Cores", color: THEME.colors.textSecondary },
      { label: "HBM", color: THEME.colors.textSecondary },
      { label: "NVLink", color: THEME.colors.textSecondary },
    ],
  },
];

export const M10S12_LandscapeOverview: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const calloutDelay = 8 * fps;
  const calloutOpacity = interpolate(
    frame - calloutDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={10}>
      <SlideTitle title="The Modern GPU Programming Landscape" />

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 12,
          flex: 1,
          width: 1776,
        }}
      >
        {layers.map((layer, layerIdx) => {
          const layerDelay = 0.8 * fps + layerIdx * 1.2 * fps;
          const layerSpring = spring({
            frame: frame - layerDelay,
            fps,
            config: { damping: 200 },
          });
          const layerOpacity = interpolate(layerSpring, [0, 1], [0, 1]);
          const layerY = interpolate(layerSpring, [0, 1], [-20, 0]);

          return (
            <div
              key={layer.name}
              style={{
                opacity: layerOpacity,
                transform: `translateY(${layerY}px)`,
                display: "flex",
                alignItems: "center",
                gap: 20,
              }}
            >
              {/* Layer label */}
              <div
                style={{
                  width: 160,
                  flexShrink: 0,
                  fontSize: 14,
                  fontWeight: 700,
                  color: layer.layerColor,
                  fontFamily: fontFamilyBody,
                  textAlign: "right",
                }}
              >
                {layer.name}
              </div>

              {/* Arrow down indicator */}
              {layerIdx > 0 && (
                <div
                  style={{
                    position: "absolute",
                    left: 170,
                    marginTop: -24,
                    fontSize: 14,
                    color: THEME.colors.textMuted,
                    opacity: 0.4,
                  }}
                >
                  {"\u25BC"}
                </div>
              )}

              {/* Layer items */}
              <div
                style={{
                  display: "flex",
                  gap: 12,
                  flex: 1,
                }}
              >
                {layer.items.map((item, itemIdx) => {
                  const itemDelay = layerDelay + itemIdx * 0.2 * fps;
                  const itemSpring = spring({
                    frame: frame - itemDelay,
                    fps,
                    config: { damping: 200 },
                  });
                  const itemScale = interpolate(itemSpring, [0, 1], [0.85, 1]);
                  const itemOpacity = interpolate(itemSpring, [0, 1], [0, 1]);

                  return (
                    <div
                      key={item.label}
                      style={{
                        flex: 1,
                        padding: "14px 18px",
                        backgroundColor: `${item.color}10`,
                        border: `1px solid ${item.color}40`,
                        borderRadius: 8,
                        textAlign: "center",
                        opacity: itemOpacity,
                        transform: `scale(${itemScale})`,
                      }}
                    >
                      <div
                        style={{
                          fontSize: 16,
                          fontWeight: 600,
                          color: item.color,
                          fontFamily: fontFamilyBody,
                        }}
                      >
                        {item.label}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      {/* Center callout */}
      <div
        style={{
          marginTop: 8,
          padding: "14px 28px",
          background: "linear-gradient(135deg, rgba(118,185,0,0.12), rgba(79,195,247,0.08))",
          borderRadius: 12,
          border: `1px solid ${THEME.colors.nvidiaGreen}30`,
          textAlign: "center",
          width: 1776,
          opacity: calloutOpacity,
        }}
      >
        <div
          style={{
            fontSize: 22,
            fontWeight: 700,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyBody,
          }}
        >
          You now understand ALL of these layers!
        </div>
      </div>
    </SlideLayout>
  );
};
