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

type ApiMode = {
  title: string;
  color: string;
  steps: string[];
  label: string;
};

const apiModes: ApiMode[] = [
  {
    title: "Host API",
    color: THEME.colors.accentBlue,
    steps: [
      "curandCreateGenerator(&gen, ...)",
      "curandGenerateUniform(gen, d_random, N)",
      "// use d_random array in kernels",
    ],
    label: "Generate array on GPU, use in subsequent kernels",
  },
  {
    title: "Device API",
    color: THEME.colors.accentPurple,
    steps: [
      "curand_init(seed, tid, 0, &state)",
      "float r = curand_uniform(&state)",
      "// use r inline in this kernel",
    ],
    label: "Generate inline within your kernel",
  },
];

const ApiBox: React.FC<{
  mode: ApiMode;
  delay: number;
}> = ({ mode, delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const boxSpring = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });
  const boxOpacity = interpolate(boxSpring, [0, 1], [0, 1]);
  const boxScale = interpolate(boxSpring, [0, 1], [0.95, 1]);

  return (
    <div
      style={{
        padding: "16px 20px",
        backgroundColor: `${mode.color}08`,
        borderLeft: `4px solid ${mode.color}`,
        borderRadius: 10,
        opacity: boxOpacity,
        transform: `scale(${boxScale})`,
        width: 680,
      }}
    >
      <div
        style={{
          fontSize: 20,
          fontWeight: 700,
          color: mode.color,
          fontFamily: fontFamilyBody,
          marginBottom: 6,
        }}
      >
        {mode.title}
      </div>
      <div
        style={{
          fontSize: 14,
          color: THEME.colors.textSecondary,
          fontFamily: fontFamilyBody,
          marginBottom: 12,
        }}
      >
        {mode.label}
      </div>
      <div
        style={{
          backgroundColor: "rgba(13,17,23,0.6)",
          borderRadius: 6,
          padding: "10px 14px",
        }}
      >
        {mode.steps.map((step, i) => {
          const stepDelay = delay + 0.3 * fps + i * 0.2 * fps;
          const stepOpacity = interpolate(
            frame - stepDelay,
            [0, 0.2 * fps],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );

          return (
            <div
              key={i}
              style={{
                fontSize: 14,
                color: step.startsWith("//")
                  ? THEME.colors.syntaxComment
                  : THEME.colors.textCode,
                fontFamily: fontFamilyCode,
                lineHeight: 1.6,
                opacity: stepOpacity,
              }}
            >
              {step}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const M7S10_CurandIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const arrowOpacity = interpolate(
    frame - 5 * fps,
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
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={7}
      leftWidth="55%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle title="cuRAND -- Random Numbers on GPU" />

          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <ApiBox mode={apiModes[0]} delay={1 * fps} />
            <ApiBox mode={apiModes[1]} delay={3 * fps} />

            {/* Arrow between them */}
            <div
              style={{
                padding: "10px 16px",
                backgroundColor: "rgba(255,171,64,0.08)",
                borderRadius: 8,
                border: `1px solid ${THEME.colors.accentOrange}30`,
                opacity: arrowOpacity,
                width: 680,
                textAlign: "center",
              }}
            >
              <span
                style={{
                  fontSize: 15,
                  color: THEME.colors.accentOrange,
                  fontFamily: fontFamilyBody,
                  fontWeight: 600,
                }}
              >
                Host = bulk generation {"  "}|{"  "} Device = per-thread generation
              </span>
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ width: 460, marginTop: 80 }}>
          <BulletPoint
            text="Multiple generators: XORWOW, Philox, MRG32k3a, MT"
            index={0}
            delay={4 * fps}
          />
          <BulletPoint
            text="Uniform, Normal, Log-normal, Poisson distributions"
            index={1}
            delay={4 * fps}
          />
          <BulletPoint
            text="Reproducible with seed control"
            index={2}
            delay={4 * fps}
          />
          <BulletPoint
            text="Billions of random numbers per second"
            index={3}
            delay={4 * fps}
            highlight
          />

          {/* Bottom recommendation */}
          <div
            style={{
              marginTop: 32,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.nvidiaGreen}30`,
              opacity: bottomOpacity,
            }}
          >
            <div
              style={{
                fontSize: 16,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
              }}
            >
              Philox recommended for ML: good quality + fast
            </div>
          </div>
        </div>
      }
    />
  );
};
