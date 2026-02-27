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

type CareerPath = {
  title: string;
  color: string;
  skills: string[];
  companies: string;
};

const paths: CareerPath[] = [
  {
    title: "ML Infrastructure",
    color: THEME.colors.nvidiaGreen,
    skills: [
      "Optimize training pipelines",
      "Custom kernels for novel architectures",
      "Distributed training (NCCL, FSDP)",
    ],
    companies: "NVIDIA, Meta, Google",
  },
  {
    title: "Inference Optimization",
    color: THEME.colors.accentBlue,
    skills: [
      "TensorRT, vLLM, TGI",
      "Quantization (INT8, INT4, FP8)",
      "KV cache optimization",
    ],
    companies: "Together, Anyscale, Modal",
  },
  {
    title: "GPU Systems",
    color: THEME.colors.accentPurple,
    skills: [
      "Multi-GPU/multi-node",
      "Memory management",
      "Custom hardware integration",
    ],
    companies: "Cerebras, Groq, d-Matrix",
  },
];

export const M10S13_CareerPath: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="gradient" moduleNumber={10}>
      <SlideTitle
        title="Your CUDA Learning Path -- What's Next?"
        subtitle="Three career directions where CUDA skills are in highest demand"
      />

      <div
        style={{
          display: "flex",
          gap: 24,
          flex: 1,
          width: 1776,
        }}
      >
        {paths.map((path, colIdx) => {
          const cardDelay = 1.2 * fps + colIdx * 0.6 * fps;
          const cardSpring = spring({
            frame: frame - cardDelay,
            fps,
            config: { damping: 200 },
          });
          const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
          const cardScale = interpolate(cardSpring, [0, 1], [0.92, 1]);

          return (
            <div
              key={path.title}
              style={{
                flex: 1,
                padding: "24px 28px",
                backgroundColor: `${path.color}06`,
                border: `1px solid ${path.color}30`,
                borderRadius: 14,
                opacity: cardOpacity,
                transform: `scale(${cardScale})`,
                display: "flex",
                flexDirection: "column",
              }}
            >
              {/* Card header */}
              <div
                style={{
                  fontSize: 22,
                  fontWeight: 800,
                  color: path.color,
                  fontFamily: fontFamilyBody,
                  marginBottom: 20,
                  paddingBottom: 14,
                  borderBottom: `2px solid ${path.color}30`,
                }}
              >
                {path.title}
              </div>

              {/* Skills */}
              <div style={{ flex: 1 }}>
                {path.skills.map((skill, skillIdx) => {
                  const skillDelay = cardDelay + 0.8 * fps + skillIdx * 0.4 * fps;
                  const skillSpring = spring({
                    frame: frame - skillDelay,
                    fps,
                    config: { damping: 200 },
                  });
                  const skillOpacity = interpolate(skillSpring, [0, 1], [0, 1]);
                  const skillX = interpolate(skillSpring, [0, 1], [-12, 0]);

                  return (
                    <div
                      key={skill}
                      style={{
                        display: "flex",
                        alignItems: "flex-start",
                        gap: 10,
                        marginBottom: 14,
                        opacity: skillOpacity,
                        transform: `translateX(${skillX}px)`,
                      }}
                    >
                      <span
                        style={{
                          color: path.color,
                          fontSize: 16,
                          lineHeight: 1.5,
                          flexShrink: 0,
                        }}
                      >
                        {"\u25B8"}
                      </span>
                      <span
                        style={{
                          fontSize: 17,
                          color: THEME.colors.textPrimary,
                          fontFamily: fontFamilyBody,
                          lineHeight: 1.5,
                        }}
                      >
                        {skill}
                      </span>
                    </div>
                  );
                })}
              </div>

              {/* Companies */}
              <div
                style={{
                  marginTop: 16,
                  padding: "10px 14px",
                  backgroundColor: `${path.color}10`,
                  borderRadius: 8,
                  opacity: interpolate(
                    frame - (cardDelay + 3 * fps),
                    [0, 0.3 * fps],
                    [0, 1],
                    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                  ),
                }}
              >
                <div
                  style={{
                    fontSize: 12,
                    color: THEME.colors.textMuted,
                    fontFamily: fontFamilyBody,
                    marginBottom: 4,
                    letterSpacing: "0.5px",
                    textTransform: "uppercase" as const,
                  }}
                >
                  Companies
                </div>
                <div
                  style={{
                    fontSize: 15,
                    color: path.color,
                    fontFamily: fontFamilyBody,
                    fontWeight: 600,
                  }}
                >
                  {path.companies}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </SlideLayout>
  );
};
