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

type ExampleCard = {
  title: string;
  color: string;
  diagram: React.ReactNode;
  code: string;
};

const DropoutDiagram: React.FC = () => {
  const cells = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1];
  return (
    <div style={{ display: "flex", gap: 4, flexWrap: "wrap", width: 180 }}>
      {cells.map((val, i) => (
        <div
          key={i}
          style={{
            width: 28,
            height: 28,
            borderRadius: 4,
            backgroundColor: val
              ? THEME.colors.accentBlue + "40"
              : THEME.colors.accentRed + "40",
            border: `1px solid ${val ? THEME.colors.accentBlue : THEME.colors.accentRed}60`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 13,
            color: val ? THEME.colors.accentBlue : THEME.colors.accentRed,
            fontFamily: fontFamilyCode,
            fontWeight: 700,
          }}
        >
          {val ? "x" : "0"}
        </div>
      ))}
    </div>
  );
};

const GaussianDiagram: React.FC = () => {
  const bars = [2, 5, 10, 18, 28, 32, 28, 18, 10, 5, 2];
  const maxH = 32;
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-end",
        gap: 3,
        height: 60,
        width: 180,
      }}
    >
      {bars.map((h, i) => (
        <div
          key={i}
          style={{
            width: 13,
            height: (h / maxH) * 50,
            backgroundColor: THEME.colors.nvidiaGreen + "80",
            borderRadius: 2,
            borderTop: `2px solid ${THEME.colors.nvidiaGreen}`,
          }}
        />
      ))}
    </div>
  );
};

const AugmentDiagram: React.FC = () => {
  return (
    <div style={{ display: "flex", gap: 8, alignItems: "center", width: 180 }}>
      {/* Original image placeholder */}
      <div
        style={{
          width: 50,
          height: 50,
          border: `2px solid ${THEME.colors.accentPurple}60`,
          borderRadius: 6,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 22,
        }}
      >
        <span style={{ color: THEME.colors.accentPurple }}>IMG</span>
      </div>
      <span style={{ fontSize: 18, color: THEME.colors.textMuted }}>{"\u2192"}</span>
      {/* Augmented versions */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <div
          style={{
            width: 40,
            height: 24,
            border: `1px solid ${THEME.colors.accentPurple}40`,
            borderRadius: 4,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 10,
            color: THEME.colors.accentPurple,
            fontFamily: fontFamilyCode,
          }}
        >
          crop
        </div>
        <div
          style={{
            width: 40,
            height: 24,
            border: `1px solid ${THEME.colors.accentPurple}40`,
            borderRadius: 4,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 10,
            color: THEME.colors.accentPurple,
            fontFamily: fontFamilyCode,
          }}
        >
          flip
        </div>
      </div>
    </div>
  );
};

const examples: ExampleCard[] = [
  {
    title: "Dropout",
    color: THEME.colors.accentBlue,
    diagram: <DropoutDiagram />,
    code: "if (curand_uniform(&state) < p) output[i] = 0;",
  },
  {
    title: "Weight Init",
    color: THEME.colors.nvidiaGreen,
    diagram: <GaussianDiagram />,
    code: "curandGenerateNormal(gen, d_weights, n, 0.0f, std);",
  },
  {
    title: "Data Augmentation",
    color: THEME.colors.accentPurple,
    diagram: <AugmentDiagram />,
    code: "float angle = curand_uniform(&state) * 360.0f;",
  },
];

export const M7S11_CurandML: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const bottomOpacity = interpolate(
    frame - 10 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="dark" moduleNumber={7}>
      <SlideTitle title="cuRAND in Machine Learning" />

      <div
        style={{
          display: "flex",
          gap: 24,
          flex: 1,
          width: 1776,
        }}
      >
        {examples.map((card, i) => {
          const cardDelay = 1.5 * fps + i * 2 * fps;
          const cardSpring = spring({
            frame: frame - cardDelay,
            fps,
            config: { damping: 200 },
          });
          const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
          const cardScale = interpolate(cardSpring, [0, 1], [0.92, 1]);

          return (
            <div
              key={card.title}
              style={{
                flex: 1,
                padding: "24px 24px",
                backgroundColor: `${card.color}08`,
                borderTop: `4px solid ${card.color}`,
                borderRadius: 12,
                opacity: cardOpacity,
                transform: `scale(${cardScale})`,
                display: "flex",
                flexDirection: "column",
              }}
            >
              {/* Card title */}
              <div
                style={{
                  fontSize: 22,
                  fontWeight: 700,
                  color: card.color,
                  fontFamily: fontFamilyBody,
                  marginBottom: 16,
                }}
              >
                {card.title}
              </div>

              {/* Diagram */}
              <div
                style={{
                  marginBottom: 20,
                  display: "flex",
                  justifyContent: "center",
                  minHeight: 70,
                  alignItems: "center",
                }}
              >
                {card.diagram}
              </div>

              {/* Code snippet */}
              <div
                style={{
                  padding: "10px 14px",
                  backgroundColor: "rgba(13,17,23,0.7)",
                  borderRadius: 8,
                  fontFamily: fontFamilyCode,
                  fontSize: 13,
                  color: THEME.colors.textCode,
                  lineHeight: 1.5,
                  wordBreak: "break-all",
                }}
              >
                {card.code}
              </div>
            </div>
          );
        })}
      </div>

      {/* Bottom note */}
      <div
        style={{
          marginTop: 20,
          padding: "14px 24px",
          backgroundColor: "rgba(255,255,255,0.03)",
          borderRadius: 10,
          border: `1px solid rgba(255,255,255,0.08)`,
          opacity: bottomOpacity,
          width: 1776,
          textAlign: "center",
        }}
      >
        <span
          style={{
            fontSize: 17,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
          }}
        >
          PyTorch/TensorFlow use cuRAND under the hood for all random ops
        </span>
      </div>
    </SlideLayout>
  );
};
