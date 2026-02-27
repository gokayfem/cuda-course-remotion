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

type Resource = {
  title: string;
  description: string;
  url: string;
  color: string;
};

const resources: Resource[] = [
  {
    title: "CUDA Programming Guide",
    description: "NVIDIA's official documentation",
    url: "nvidia.com",
    color: THEME.colors.nvidiaGreen,
  },
  {
    title: "GPU Gems / GPU Pro",
    description: "Advanced rendering and compute techniques",
    url: "developer.nvidia.com",
    color: THEME.colors.accentBlue,
  },
  {
    title: "Triton Tutorials",
    description: "Python GPU programming",
    url: "triton-lang.org",
    color: THEME.colors.accentPurple,
  },
  {
    title: "CUTLASS GitHub",
    description: "GEMM templates library",
    url: "github.com/NVIDIA/cutlass",
    color: THEME.colors.accentOrange,
  },
  {
    title: "Flash Attention Paper",
    description: "Dao et al. 2022 -- must-read for ML",
    url: "arxiv.org",
    color: THEME.colors.accentCyan,
  },
  {
    title: "Nsight Compute Docs",
    description: "Profiling deep-dives",
    url: "developer.nvidia.com",
    color: THEME.colors.accentYellow,
  },
];

export const M10S14_ResourcesGuide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="dark" moduleNumber={10}>
      <SlideTitle
        title="Essential Resources"
        subtitle="Where to continue your GPU programming journey"
      />

      {/* 2x3 grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: 18,
          flex: 1,
          width: 1776,
        }}
      >
        {resources.map((resource, i) => {
          const cardDelay = 0.8 * fps + i * 0.4 * fps;
          const cardSpring = spring({
            frame: frame - cardDelay,
            fps,
            config: { damping: 200 },
          });
          const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
          const cardScale = interpolate(cardSpring, [0, 1], [0.9, 1]);

          return (
            <div
              key={resource.title}
              style={{
                padding: "20px 24px",
                backgroundColor: `${resource.color}06`,
                borderLeft: `4px solid ${resource.color}`,
                borderRadius: 10,
                opacity: cardOpacity,
                transform: `scale(${cardScale})`,
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
              }}
            >
              <div>
                <div
                  style={{
                    fontSize: 19,
                    fontWeight: 700,
                    color: resource.color,
                    fontFamily: fontFamilyBody,
                    marginBottom: 10,
                  }}
                >
                  {resource.title}
                </div>
                <div
                  style={{
                    fontSize: 15,
                    color: THEME.colors.textSecondary,
                    fontFamily: fontFamilyBody,
                    lineHeight: 1.5,
                  }}
                >
                  {resource.description}
                </div>
              </div>
              <div
                style={{
                  marginTop: 12,
                  fontSize: 13,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyCode,
                  padding: "4px 8px",
                  backgroundColor: "rgba(255,255,255,0.04)",
                  borderRadius: 4,
                  display: "inline-block",
                  alignSelf: "flex-start",
                }}
              >
                {resource.url}
              </div>
            </div>
          );
        })}
      </div>

      {/* Bottom advice */}
      <div
        style={{
          marginTop: 14,
          padding: "14px 24px",
          backgroundColor: "rgba(118,185,0,0.08)",
          borderRadius: 10,
          border: `1px solid ${THEME.colors.nvidiaGreen}30`,
          width: 1776,
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
            fontSize: 17,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyBody,
            fontWeight: 600,
            lineHeight: 1.5,
          }}
        >
          The best way to learn: profile real workloads, identify bottlenecks, optimize
        </div>
      </div>
    </SlideLayout>
  );
};
