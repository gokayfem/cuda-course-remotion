import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

type Mistake = {
  title: string;
  explanation: string;
};

const mistakes: Mistake[] = [
  {
    title: "Using pageable memory with async",
    explanation: "Falls back to synchronous! Always use cudaMallocHost.",
  },
  {
    title: "Forgetting stream sync",
    explanation:
      "Kernel results not ready when CPU reads them. Use events or sync.",
  },
  {
    title: "Default stream blocking",
    explanation:
      "Stream 0 synchronizes with all streams by default. Use --default-stream per-thread.",
  },
];

const bestPractices: string[] = [
  "Always pin memory for async",
  "Use events for dependencies, not device sync",
  "Profile with nsys to verify actual overlap",
  "Keep chunks large enough to hide launch overhead",
];

export const M6S14_CommonMistakes: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={6}
      leftWidth="50%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle title="Common Stream Pitfalls" />

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 16,
            }}
          >
            {mistakes.map((mistake, i) => {
              const cardDelay = 1 * fps + i * 1.5 * fps;
              const cardSpring = spring({
                frame: frame - cardDelay,
                fps,
                config: { damping: 200 },
              });
              const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
              const cardX = interpolate(cardSpring, [0, 1], [-20, 0]);

              return (
                <div
                  key={mistake.title}
                  style={{
                    padding: "16px 20px",
                    backgroundColor: `${THEME.colors.accentRed}08`,
                    border: `2px solid ${THEME.colors.accentRed}40`,
                    borderRadius: 10,
                    opacity: cardOpacity,
                    transform: `translateX(${cardX}px)`,
                    width: 680,
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                      marginBottom: 8,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 16,
                        color: THEME.colors.accentRed,
                        fontWeight: 700,
                      }}
                    >
                      X
                    </span>
                    <span
                      style={{
                        fontSize: 17,
                        fontWeight: 700,
                        color: THEME.colors.accentRed,
                        fontFamily: fontFamilyBody,
                      }}
                    >
                      {mistake.title}
                    </span>
                  </div>
                  <div
                    style={{
                      fontSize: 15,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyBody,
                      lineHeight: 1.5,
                      paddingLeft: 26,
                    }}
                  >
                    {mistake.explanation}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      }
      right={
        <div style={{ width: 500, marginTop: 80 }}>
          <div
            style={{
              padding: "20px 24px",
              backgroundColor: `${THEME.colors.nvidiaGreen}08`,
              border: `2px solid ${THEME.colors.nvidiaGreen}40`,
              borderRadius: 12,
              opacity: interpolate(
                frame - 5.5 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div
              style={{
                fontSize: 20,
                fontWeight: 700,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                marginBottom: 18,
              }}
            >
              Best Practices
            </div>

            {bestPractices.map((practice, i) => {
              const practiceDelay = 6 * fps + i * 0.4 * fps;
              const practiceSpring = spring({
                frame: frame - practiceDelay,
                fps,
                config: { damping: 200 },
              });
              const practiceOpacity = interpolate(
                practiceSpring,
                [0, 1],
                [0, 1]
              );
              const practiceX = interpolate(
                practiceSpring,
                [0, 1],
                [-15, 0]
              );

              return (
                <div
                  key={practice}
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 10,
                    marginBottom: 14,
                    opacity: practiceOpacity,
                    transform: `translateX(${practiceX}px)`,
                  }}
                >
                  <span
                    style={{
                      color: THEME.colors.nvidiaGreen,
                      fontSize: 16,
                      fontWeight: 700,
                      flexShrink: 0,
                      lineHeight: 1.5,
                    }}
                  >
                    +
                  </span>
                  <span
                    style={{
                      fontSize: 17,
                      color: THEME.colors.textPrimary,
                      fontFamily: fontFamilyBody,
                      lineHeight: 1.5,
                    }}
                  >
                    {practice}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      }
    />
  );
};
