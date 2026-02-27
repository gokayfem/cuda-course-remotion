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
import { fontFamilyBody, fontFamilyCode, fontFamilyHeading } from "../../../../styles/fonts";

interface PassDef {
  readonly label: string;
  readonly detail: string;
  readonly color: string;
}

const PASSES: readonly PassDef[] = [
  {
    label: "Pass 1: Find max",
    detail: "max_val = max(x[0..N])",
    color: THEME.colors.accentBlue,
  },
  {
    label: "Pass 2: Compute exp + sum",
    detail: "sum += exp(x[i] - max_val)",
    color: THEME.colors.accentOrange,
  },
  {
    label: "Pass 3: Normalize",
    detail: "out[i] = exp(x[i] - max_val) / sum",
    color: THEME.colors.nvidiaGreen,
  },
];

export const M9S03_SoftmaxIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Formula animations
  const naiveDelay = 0.8 * fps;
  const naiveOpacity = interpolate(
    frame - naiveDelay,
    [0, 0.4 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const problemDelay = 2.5 * fps;
  const problemOpacity = interpolate(
    frame - problemDelay,
    [0, 0.4 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const safeDelay = 4.5 * fps;
  const safeOpacity = interpolate(
    frame - safeDelay,
    [0, 0.4 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={9}
      leftWidth="50%"
      left={
        <div style={{ width: 800 }}>
          <SlideTitle
            title="The Softmax Kernel"
            subtitle="Numerically stable softmax is critical for attention"
          />

          {/* Naive formula */}
          <div style={{ marginBottom: 28, opacity: naiveOpacity }}>
            <div
              style={{
                fontSize: 14,
                color: THEME.colors.textMuted,
                fontFamily: fontFamilyBody,
                marginBottom: 8,
                fontWeight: 600,
                letterSpacing: "0.5px",
              }}
            >
              NAIVE SOFTMAX
            </div>
            <div
              style={{
                padding: "16px 24px",
                backgroundColor: "rgba(79,195,247,0.08)",
                borderRadius: 10,
                borderLeft: `4px solid ${THEME.colors.accentBlue}`,
              }}
            >
              <span
                style={{
                  fontSize: 26,
                  fontFamily: fontFamilyCode,
                  color: THEME.colors.accentBlue,
                }}
              >
                softmax(x
                <sub style={{ fontSize: 18 }}>i</sub>) = exp(x
                <sub style={{ fontSize: 18 }}>i</sub>) /{" "}
                <span style={{ fontSize: 22 }}>{"\u03A3"}</span> exp(x
                <sub style={{ fontSize: 18 }}>j</sub>)
              </span>
            </div>
          </div>

          {/* Problem callout */}
          <div style={{ marginBottom: 28, opacity: problemOpacity }}>
            <div
              style={{
                padding: "14px 24px",
                backgroundColor: "rgba(255,82,82,0.1)",
                borderRadius: 10,
                borderLeft: `4px solid ${THEME.colors.accentRed}`,
              }}
            >
              <span
                style={{
                  fontSize: 22,
                  fontFamily: fontFamilyCode,
                  color: THEME.colors.accentRed,
                  fontWeight: 700,
                }}
              >
                Problem: exp(1000) = Inf!
              </span>
              <div
                style={{
                  fontSize: 16,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  marginTop: 6,
                }}
              >
                FP32 overflows at exp(88.7). Logits in transformers often
                exceed this.
              </div>
            </div>
          </div>

          {/* Safe formula */}
          <div style={{ opacity: safeOpacity }}>
            <div
              style={{
                fontSize: 14,
                color: THEME.colors.textMuted,
                fontFamily: fontFamilyBody,
                marginBottom: 8,
                fontWeight: 600,
                letterSpacing: "0.5px",
              }}
            >
              SAFE SOFTMAX (subtract max)
            </div>
            <div
              style={{
                padding: "16px 24px",
                backgroundColor: "rgba(118,185,0,0.08)",
                borderRadius: 10,
                borderLeft: `4px solid ${THEME.colors.nvidiaGreen}`,
              }}
            >
              <span
                style={{
                  fontSize: 24,
                  fontFamily: fontFamilyCode,
                  color: THEME.colors.nvidiaGreen,
                }}
              >
                m = max(x)
              </span>
              <br />
              <span
                style={{
                  fontSize: 24,
                  fontFamily: fontFamilyCode,
                  color: THEME.colors.nvidiaGreen,
                }}
              >
                softmax(x
                <sub style={{ fontSize: 16 }}>i</sub>) = exp(x
                <sub style={{ fontSize: 16 }}>i</sub> - m) /{" "}
                <span style={{ fontSize: 20 }}>{"\u03A3"}</span> exp(x
                <sub style={{ fontSize: 16 }}>j</sub> - m)
              </span>
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ width: 700 }}>
          <div
            style={{
              fontSize: 14,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
              marginBottom: 16,
              marginTop: 80,
              fontWeight: 600,
              letterSpacing: "0.5px",
              opacity: interpolate(
                frame - 5.5 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            THREE-PASS IMPLEMENTATION
          </div>

          {/* Three passes as pipeline blocks */}
          {PASSES.map((pass, i) => {
            const passDelay = 6 * fps + i * 0.8 * fps;
            const passSpring = spring({
              frame: frame - passDelay,
              fps,
              config: { damping: 200 },
            });
            const opacity = interpolate(passSpring, [0, 1], [0, 1]);
            const translateX = interpolate(passSpring, [0, 1], [30, 0]);

            return (
              <div key={pass.label} style={{ marginBottom: 16 }}>
                <div
                  style={{
                    opacity,
                    transform: `translateX(${translateX}px)`,
                    display: "flex",
                    alignItems: "stretch",
                    gap: 0,
                  }}
                >
                  {/* Pass block */}
                  <div
                    style={{
                      width: 320,
                      padding: "14px 20px",
                      backgroundColor: `${pass.color}12`,
                      border: `2px solid ${pass.color}60`,
                      borderRadius: "10px 0 0 10px",
                    }}
                  >
                    <div
                      style={{
                        fontSize: 18,
                        fontWeight: 700,
                        color: pass.color,
                        fontFamily: fontFamilyBody,
                      }}
                    >
                      {pass.label}
                    </div>
                    <div
                      style={{
                        fontSize: 15,
                        color: THEME.colors.textSecondary,
                        fontFamily: fontFamilyCode,
                        marginTop: 4,
                      }}
                    >
                      {pass.detail}
                    </div>
                  </div>

                  {/* Global memory arrow */}
                  <div
                    style={{
                      width: 140,
                      backgroundColor: "rgba(255,255,255,0.03)",
                      border: "2px solid rgba(255,255,255,0.1)",
                      borderLeft: "none",
                      borderRadius: "0 10px 10px 0",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                      padding: "8px 12px",
                    }}
                  >
                    <span
                      style={{
                        fontSize: 20,
                        color: THEME.colors.textMuted,
                      }}
                    >
                      {"\u2194"}
                    </span>
                    <span
                      style={{
                        fontSize: 12,
                        color: THEME.colors.textMuted,
                        fontFamily: fontFamilyCode,
                      }}
                    >
                      Global Mem
                    </span>
                  </div>
                </div>
              </div>
            );
          })}

          {/* Bottom callout */}
          <div
            style={{
              marginTop: 24,
              padding: "14px 24px",
              backgroundColor: "rgba(255,171,64,0.08)",
              borderRadius: 10,
              borderLeft: `4px solid ${THEME.colors.accentOrange}`,
              opacity: interpolate(
                frame - 9 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <span
              style={{
                fontSize: 20,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
              }}
            >
              3 passes ={" "}
              <span
                style={{
                  color: THEME.colors.accentOrange,
                  fontWeight: 700,
                }}
              >
                3 global memory round-trips
              </span>
              . Can we do better?
            </span>
          </div>
        </div>
      }
    />
  );
};
