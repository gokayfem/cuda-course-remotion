import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint, FadeInText } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

type ApproachCard = {
  title: string;
  code: string;
  description: string;
  color: string;
};

const approaches: ApproachCard[] = [
  {
    title: "JIT Compilation",
    code: "torch.utils.cpp_extension.load()",
    description: "Compile at runtime, easiest",
    color: THEME.colors.accentBlue,
  },
  {
    title: "setup.py",
    code: "CUDAExtension",
    description: "Pre-compiled package, production",
    color: THEME.colors.nvidiaGreen,
  },
  {
    title: "torch.compile",
    code: "Custom ops + Triton",
    description: "Newest approach, Python-native",
    color: THEME.colors.accentPurple,
  },
];

export const M10S10_PyTorchExtIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={10}
      leftWidth="50%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle title="Custom PyTorch CUDA Extensions" />

          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {approaches.map((approach, i) => {
              const cardDelay = 1 * fps + i * 0.8 * fps;
              const cardSpring = spring({
                frame: frame - cardDelay,
                fps,
                config: { damping: 200 },
              });
              const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
              const cardX = interpolate(cardSpring, [0, 1], [-40, 0]);

              return (
                <div
                  key={approach.title}
                  style={{
                    padding: "16px 20px",
                    backgroundColor: `${approach.color}08`,
                    borderLeft: `4px solid ${approach.color}`,
                    borderRadius: 10,
                    opacity: cardOpacity,
                    transform: `translateX(${cardX}px)`,
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 12,
                      marginBottom: 8,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 14,
                        fontWeight: 700,
                        color: THEME.colors.textPrimary,
                        fontFamily: fontFamilyBody,
                        padding: "2px 10px",
                        backgroundColor: `${approach.color}20`,
                        borderRadius: 6,
                      }}
                    >
                      {i + 1}
                    </div>
                    <div
                      style={{
                        fontSize: 18,
                        fontWeight: 700,
                        color: approach.color,
                        fontFamily: fontFamilyBody,
                      }}
                    >
                      {approach.title}
                    </div>
                  </div>
                  <div
                    style={{
                      fontSize: 14,
                      color: THEME.colors.accentCyan,
                      fontFamily: fontFamilyCode,
                      marginBottom: 6,
                      padding: "4px 10px",
                      backgroundColor: "rgba(13,17,23,0.5)",
                      borderRadius: 4,
                      display: "inline-block",
                    }}
                  >
                    {approach.code}
                  </div>
                  <div
                    style={{
                      fontSize: 15,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyBody,
                      lineHeight: 1.4,
                      marginTop: 4,
                    }}
                  >
                    {approach.description}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      }
      right={
        <div style={{ width: 560, marginTop: 10 }}>
          <FadeInText
            text="When to Use"
            delay={4 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 18 }}
          />

          <BulletPoint
            text="When PyTorch built-ins aren't fast enough"
            index={0}
            delay={4.5 * fps}
            highlight
          />
          <BulletPoint
            text="Write CUDA kernel + C++ binding + Python wrapper"
            index={1}
            delay={4.5 * fps}
          />
          <BulletPoint
            text="pybind11 for C++ to Python interface"
            index={2}
            delay={4.5 * fps}
          />
          <BulletPoint
            text="torch.autograd.Function for custom backward"
            index={3}
            delay={4.5 * fps}
          />

          {/* Bottom callout */}
          <div
            style={{
              marginTop: 28,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}30`,
              opacity: interpolate(
                frame - 8 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div
              style={{
                fontSize: 16,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                lineHeight: 1.5,
              }}
            >
              Most common use: fused operations not available in PyTorch
            </div>
          </div>
        </div>
      }
    />
  );
};
