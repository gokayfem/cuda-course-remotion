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
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

interface ArchLayer {
  readonly label: string;
  readonly sublabel: string;
  readonly color: string;
  readonly width: number;
  readonly height: number;
}

const LAYERS: readonly ArchLayer[] = [
  {
    label: "Device Level",
    sublabel: "Grid of threadblocks",
    color: THEME.colors.accentBlue,
    width: 620,
    height: 70,
  },
  {
    label: "Threadblock Level",
    sublabel: "Tiles: global -> shared memory",
    color: THEME.colors.accentPurple,
    width: 500,
    height: 70,
  },
  {
    label: "Warp Level",
    sublabel: "Warp-level MMA (WMMA/MMA)",
    color: THEME.colors.accentOrange,
    width: 380,
    height: 70,
  },
  {
    label: "Instruction Level",
    sublabel: "Tensor Core operations",
    color: THEME.colors.nvidiaGreen,
    width: 260,
    height: 70,
  },
];

export const M10S07_CUTLASSIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const bottomDelay = 10 * fps;
  const bottomOpacity = interpolate(
    frame - bottomDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={10}
      leftWidth="55%"
      left={
        <div style={{ width: 860 }}>
          <SlideTitle
            title="CUTLASS -- Customizable GEMM Templates"
            subtitle="Open-source C++ template library for high-performance linear algebra"
          />

          {/* Architecture diagram - stacked layers zooming in */}
          <div
            style={{
              position: "relative",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 12,
              marginTop: 12,
            }}
          >
            {LAYERS.map((layer, i) => {
              const layerDelay = 1 * fps + i * 0.6 * fps;
              const layerSpring = spring({
                frame: frame - layerDelay,
                fps,
                config: { damping: 200 },
              });
              const layerOpacity = interpolate(layerSpring, [0, 1], [0, 1]);
              const layerScale = interpolate(layerSpring, [0, 1], [0.9, 1]);

              const zoomDelay = 3 * fps + i * 0.5 * fps;
              const zoomGlow = interpolate(
                frame - zoomDelay,
                [0, 0.4 * fps, 1 * fps],
                [0, 0.6, 0.2],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              return (
                <div
                  key={layer.label}
                  style={{
                    width: layer.width,
                    height: layer.height,
                    backgroundColor: `${layer.color}12`,
                    border: `2px solid ${layer.color}60`,
                    borderRadius: 10,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    opacity: layerOpacity,
                    transform: `scale(${layerScale})`,
                    boxShadow: `0 0 ${8 + zoomGlow * 16}px ${layer.color}${Math.round(zoomGlow * 80).toString(16).padStart(2, "0")}`,
                    position: "relative",
                  }}
                >
                  {/* Level number */}
                  <div
                    style={{
                      position: "absolute",
                      left: 12,
                      top: "50%",
                      transform: "translateY(-50%)",
                      width: 28,
                      height: 28,
                      borderRadius: 14,
                      backgroundColor: `${layer.color}30`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 14,
                      fontWeight: 800,
                      color: layer.color,
                      fontFamily: fontFamilyBody,
                    }}
                  >
                    {i + 1}
                  </div>

                  <span
                    style={{
                      fontSize: 16,
                      fontWeight: 700,
                      color: layer.color,
                      fontFamily: fontFamilyBody,
                    }}
                  >
                    {layer.label}
                  </span>
                  <span
                    style={{
                      fontSize: 13,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyCode,
                      marginTop: 2,
                    }}
                  >
                    {layer.sublabel}
                  </span>

                  {/* Zoom-in connector line */}
                  {i < LAYERS.length - 1 && (
                    <div
                      style={{
                        position: "absolute",
                        bottom: -14,
                        left: "50%",
                        transform: "translateX(-50%)",
                        fontSize: 16,
                        color: THEME.colors.textMuted,
                        opacity: layerOpacity * 0.6,
                      }}
                    >
                      {"\u25BC"}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Bottom insight */}
          <div
            style={{
              marginTop: 24,
              padding: "12px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              borderLeft: `4px solid ${THEME.colors.nvidiaGreen}`,
              opacity: bottomOpacity,
              textAlign: "center",
            }}
          >
            <span
              style={{
                fontSize: 18,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
              }}
            >
              When cuBLAS is too rigid but you need{" "}
              <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>
                peak performance
              </span>
            </span>
          </div>
        </div>
      }
      right={
        <div style={{ width: 540, marginTop: 80 }}>
          <BulletPoint
            text="Open-source C++ template library"
            index={0}
            delay={5 * fps}
            highlight
            subtext="NVIDIA-maintained, header-only, composable building blocks"
          />
          <BulletPoint
            text="Near-cuBLAS performance with full customization"
            index={1}
            delay={5 * fps}
            subtext="Tune tile sizes, memory access patterns, epilogues"
          />
          <BulletPoint
            text="Fused epilogues: GEMM + bias + activation"
            index={2}
            delay={5 * fps}
            highlight
            subtext="Avoid writing intermediate results to global memory"
          />
          <BulletPoint
            text="CUTLASS 3.x: CuTe tensor layout algebra"
            index={3}
            delay={5 * fps}
            subtext="New abstraction for describing tensor layouts and operations"
          />
        </div>
      }
    />
  );
};
