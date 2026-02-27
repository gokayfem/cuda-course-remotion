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

type LibRow = {
  category: string;
  color: string;
  libs: string[];
};

const rows: LibRow[] = [
  {
    category: "Math",
    color: THEME.colors.accentBlue,
    libs: ["cuBLAS", "cuSPARSE", "cuSOLVER", "cuFFT"],
  },
  {
    category: "ML / DL",
    color: THEME.colors.nvidiaGreen,
    libs: ["cuDNN", "TensorRT", "cuML"],
  },
  {
    category: "Algorithms",
    color: THEME.colors.accentPurple,
    libs: ["Thrust", "CUB", "cuRAND"],
  },
  {
    category: "Communication",
    color: THEME.colors.accentOrange,
    libs: ["NCCL", "GPUDirect"],
  },
];

export const M7S12_LibraryEcosystem: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const centerOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const bottomOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={7}>
      <SlideTitle title="The CUDA Library Ecosystem" />

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 20,
          flex: 1,
          width: 1776,
        }}
      >
        {rows.map((row, rowIdx) => {
          const rowDelay = 1 * fps + rowIdx * 1.2 * fps;
          const rowSpring = spring({
            frame: frame - rowDelay,
            fps,
            config: { damping: 200 },
          });
          const rowOpacity = interpolate(rowSpring, [0, 1], [0, 1]);
          const rowX = interpolate(rowSpring, [0, 1], [-30, 0]);

          return (
            <div
              key={row.category}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 24,
                opacity: rowOpacity,
                transform: `translateX(${rowX}px)`,
              }}
            >
              {/* Category label */}
              <div
                style={{
                  width: 160,
                  flexShrink: 0,
                  padding: "10px 16px",
                  backgroundColor: `${row.color}15`,
                  borderRadius: 8,
                  border: `1px solid ${row.color}40`,
                  textAlign: "center",
                }}
              >
                <div
                  style={{
                    fontSize: 16,
                    fontWeight: 700,
                    color: row.color,
                    fontFamily: fontFamilyBody,
                  }}
                >
                  {row.category}
                </div>
              </div>

              {/* Library boxes */}
              <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                {row.libs.map((lib, libIdx) => {
                  const libDelay = rowDelay + 0.2 * fps + libIdx * 0.15 * fps;
                  const libSpring = spring({
                    frame: frame - libDelay,
                    fps,
                    config: { damping: 200 },
                  });
                  const libOpacity = interpolate(libSpring, [0, 1], [0, 1]);
                  const libScale = interpolate(libSpring, [0, 1], [0.9, 1]);

                  return (
                    <div
                      key={lib}
                      style={{
                        padding: "14px 32px",
                        backgroundColor: `${row.color}12`,
                        border: `2px solid ${row.color}50`,
                        borderRadius: 10,
                        opacity: libOpacity,
                        transform: `scale(${libScale})`,
                      }}
                    >
                      <div
                        style={{
                          fontSize: 18,
                          fontWeight: 700,
                          color: THEME.colors.textPrimary,
                          fontFamily: fontFamilyBody,
                          textAlign: "center",
                        }}
                      >
                        {lib}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}

        {/* Center label */}
        <div
          style={{
            textAlign: "center",
            padding: "12px 24px",
            opacity: centerOpacity,
            marginTop: 8,
          }}
        >
          <span
            style={{
              fontSize: 18,
              color: THEME.colors.accentCyan,
              fontFamily: fontFamilyBody,
              fontWeight: 600,
            }}
          >
            All optimized for NVIDIA hardware, all using Tensor Cores
          </span>
        </div>
      </div>

      {/* Bottom */}
      <div
        style={{
          padding: "14px 24px",
          backgroundColor: "rgba(118,185,0,0.08)",
          borderRadius: 10,
          border: `1px solid ${THEME.colors.nvidiaGreen}30`,
          opacity: bottomOpacity,
          width: 1776,
          textAlign: "center",
        }}
      >
        <span
          style={{
            fontSize: 18,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyBody,
            fontWeight: 700,
          }}
        >
          Choose the right library {"\u2192"} instant 10-100x over hand-written kernels
        </span>
      </div>
    </SlideLayout>
  );
};
