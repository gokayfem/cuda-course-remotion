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

interface StageDef {
  readonly label: string;
  readonly formula: string;
  readonly color: string;
}

const STAGES: readonly StageDef[] = [
  {
    label: "Compute Mean",
    formula: "\u03BC = (1/D) \u03A3 x\u1D62",
    color: THEME.colors.accentBlue,
  },
  {
    label: "Compute Variance",
    formula: "\u03C3\u00B2 = (1/D) \u03A3 (x\u1D62 - \u03BC)\u00B2",
    color: THEME.colors.accentOrange,
  },
  {
    label: "Normalize",
    formula: "x\u0302 = (x - \u03BC) / \u221A(\u03C3\u00B2 + \u03B5)",
    color: THEME.colors.accentPurple,
  },
  {
    label: "Scale + Shift",
    formula: "y = \u03B3 \u00D7 x\u0302 + \u03B2",
    color: THEME.colors.nvidiaGreen,
  },
];

export const M9S06_LayerNormIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={9}
      leftWidth="50%"
      left={
        <div style={{ width: 800 }}>
          <SlideTitle
            title="The LayerNorm Kernel"
            subtitle="Normalizing each token's hidden dimension independently"
          />

          {/* Main formula */}
          <div
            style={{
              padding: "20px 28px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 12,
              borderLeft: `4px solid ${THEME.colors.nvidiaGreen}`,
              marginBottom: 28,
              opacity: interpolate(
                frame - 0.8 * fps,
                [0, 0.4 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <span
              style={{
                fontSize: 28,
                fontFamily: fontFamilyCode,
                color: THEME.colors.nvidiaGreen,
                fontWeight: 700,
              }}
            >
              y = (x - {"\u03BC"}) / {"\u221A"}({"\u03C3\u00B2"} + {"\u03B5"}) {"\u00D7"} {"\u03B3"} + {"\u03B2"}
            </span>
          </div>

          {/* Animated stages */}
          <div
            style={{
              fontSize: 14,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
              marginBottom: 12,
              fontWeight: 600,
              letterSpacing: "0.5px",
              opacity: interpolate(
                frame - 2 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            NORMALIZATION STAGES
          </div>

          {STAGES.map((stage, i) => {
            const stageDelay = 2.5 * fps + i * 0.8 * fps;
            const stageSpring = spring({
              frame: frame - stageDelay,
              fps,
              config: { damping: 200 },
            });
            const opacity = interpolate(stageSpring, [0, 1], [0, 1]);
            const translateX = interpolate(stageSpring, [0, 1], [25, 0]);

            const isActive =
              frame >= stageDelay &&
              (i === STAGES.length - 1 ||
                frame < stageDelay + 0.8 * fps);

            return (
              <div
                key={stage.label}
                style={{
                  opacity,
                  transform: `translateX(${translateX}px)`,
                  marginBottom: 10,
                  display: "flex",
                  alignItems: "center",
                  gap: 16,
                }}
              >
                {/* Step number */}
                <div
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: 16,
                    backgroundColor: isActive
                      ? `${stage.color}30`
                      : `${stage.color}15`,
                    border: `2px solid ${stage.color}${isActive ? "" : "60"}`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 14,
                    fontWeight: 700,
                    color: stage.color,
                    fontFamily: fontFamilyBody,
                    flexShrink: 0,
                  }}
                >
                  {i + 1}
                </div>

                {/* Content */}
                <div
                  style={{
                    flex: 1,
                    padding: "10px 16px",
                    backgroundColor: isActive
                      ? `${stage.color}10`
                      : "rgba(255,255,255,0.02)",
                    borderRadius: 8,
                    border: `1px solid ${stage.color}${isActive ? "60" : "20"}`,
                  }}
                >
                  <span
                    style={{
                      fontSize: 16,
                      fontWeight: 700,
                      color: stage.color,
                      fontFamily: fontFamilyBody,
                    }}
                  >
                    {stage.label}
                  </span>
                  <span
                    style={{
                      fontSize: 16,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyCode,
                      marginLeft: 16,
                    }}
                  >
                    {stage.formula}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      }
      right={
        <div style={{ width: 580, marginTop: 80 }}>
          <BulletPoint
            text="Each token normalized independently"
            index={0}
            delay={3 * fps}
            highlight
            subtext="No cross-token dependencies = perfect parallelism"
          />
          <BulletPoint
            text="Hidden dimension D = 768 to 12288+"
            index={1}
            delay={3 * fps}
            subtext="GPT-2: 768, LLaMA-70B: 8192, GPT-4 class: 12288+"
          />
          <BulletPoint
            text="Appears after every attention & FFN block"
            index={2}
            delay={3 * fps}
            subtext="2 LayerNorms per transformer layer = high invocation count"
          />
          <BulletPoint
            text="Bottleneck: memory-bound reduction"
            index={3}
            delay={3 * fps}
            subtext="Two reductions (mean, variance) require reading all D elements"
          />
          <BulletPoint
            text="Learnable parameters: \u03B3 (scale), \u03B2 (bias)"
            index={4}
            delay={3 * fps}
            subtext="Element-wise multiply and add after normalization"
          />
        </div>
      }
    />
  );
};
