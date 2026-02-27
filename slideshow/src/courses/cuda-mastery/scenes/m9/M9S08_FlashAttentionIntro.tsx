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
import { fontFamilyBody, fontFamilyCode, fontFamilyHeading } from "../../../../styles/fonts";

interface MethodStep {
  readonly label: string;
  readonly detail: string;
}

const STANDARD_STEPS: readonly MethodStep[] = [
  { label: "Q x K\u1D40", detail: "N x N score matrix" },
  { label: "Softmax", detail: "row-wise normalization" },
  { label: "x V", detail: "weighted value sum" },
];

const FLASH_STEPS: readonly MethodStep[] = [
  { label: "Tile Q blocks", detail: "process B_r rows at a time" },
  { label: "Iterate K,V tiles", detail: "B_c columns per tile" },
  { label: "Online softmax", detail: "never materialize N x N" },
  { label: "Accumulate O", detail: "write once to HBM" },
];

export const M9S08_FlashAttentionIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const leftDelay = 1 * fps;
  const rightDelay = 3 * fps;
  const arrowDelay = 5.5 * fps;

  const leftOpacity = interpolate(
    frame - leftDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const rightOpacity = interpolate(
    frame - rightDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const arrowSpring = spring({
    frame: frame - arrowDelay,
    fps,
    config: { damping: 200 },
  });
  const arrowOpacity = interpolate(arrowSpring, [0, 1], [0, 1]);

  return (
    <SlideLayout variant="gradient" moduleNumber={9}>
      <SlideTitle
        title="Flash Attention \u2014 The Breakthrough"
        subtitle="Same exact output, fundamentally different memory access pattern"
      />

      <div
        style={{
          flex: 1,
          display: "flex",
          gap: 40,
          alignItems: "flex-start",
          width: 1776,
        }}
      >
        {/* Standard Attention (left) */}
        <div
          style={{
            flex: 1,
            opacity: leftOpacity,
          }}
        >
          <div
            style={{
              fontSize: 14,
              color: THEME.colors.accentRed,
              fontFamily: fontFamilyBody,
              fontWeight: 700,
              letterSpacing: "0.5px",
              marginBottom: 12,
            }}
          >
            STANDARD ATTENTION
          </div>

          <div
            style={{
              padding: "20px 24px",
              backgroundColor: "rgba(255,82,82,0.08)",
              border: `2px solid ${THEME.colors.accentRed}50`,
              borderRadius: 12,
            }}
          >
            {STANDARD_STEPS.map((step, i) => {
              const stepDelay = leftDelay + 0.5 * fps + i * 0.5 * fps;
              const stepOpacity = interpolate(
                frame - stepDelay,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              return (
                <div
                  key={step.label}
                  style={{
                    opacity: stepOpacity,
                    display: "flex",
                    alignItems: "center",
                    gap: 14,
                    marginBottom: i < STANDARD_STEPS.length - 1 ? 14 : 0,
                  }}
                >
                  <div
                    style={{
                      width: 28,
                      height: 28,
                      borderRadius: 14,
                      backgroundColor: `${THEME.colors.accentRed}20`,
                      border: `2px solid ${THEME.colors.accentRed}60`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 13,
                      fontWeight: 700,
                      color: THEME.colors.accentRed,
                      fontFamily: fontFamilyBody,
                      flexShrink: 0,
                    }}
                  >
                    {i + 1}
                  </div>
                  <div>
                    <span
                      style={{
                        fontSize: 20,
                        fontWeight: 700,
                        color: THEME.colors.accentRed,
                        fontFamily: fontFamilyCode,
                      }}
                    >
                      {step.label}
                    </span>
                    <span
                      style={{
                        fontSize: 16,
                        color: THEME.colors.textMuted,
                        fontFamily: fontFamilyBody,
                        marginLeft: 12,
                      }}
                    >
                      {step.detail}
                    </span>
                  </div>
                </div>
              );
            })}

            {/* Memory badge */}
            <div
              style={{
                marginTop: 18,
                padding: "10px 18px",
                backgroundColor: "rgba(255,82,82,0.12)",
                borderRadius: 8,
                display: "inline-block",
                opacity: interpolate(
                  frame - (leftDelay + 2 * fps),
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
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
                Memory: O(N{"\u00B2"})
              </span>
              <span
                style={{
                  fontSize: 16,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  marginLeft: 12,
                }}
              >
                Stores full N x N attention matrix in HBM
              </span>
            </div>
          </div>
        </div>

        {/* Arrow in the middle */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            alignSelf: "center",
            opacity: arrowOpacity,
            width: 160,
            flexShrink: 0,
          }}
        >
          <div
            style={{
              fontSize: 48,
              color: THEME.colors.nvidiaGreen,
              fontFamily: fontFamilyBody,
            }}
          >
            {"\u2192"}
          </div>
          <div
            style={{
              fontSize: 15,
              color: THEME.colors.textSecondary,
              fontFamily: fontFamilyBody,
              textAlign: "center",
              marginTop: 8,
              lineHeight: 1.4,
            }}
          >
            Same output
            <br />
            <span
              style={{
                color: THEME.colors.nvidiaGreen,
                fontWeight: 700,
              }}
            >
              2-4x faster
            </span>
          </div>
          <div
            style={{
              fontSize: 14,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyCode,
              textAlign: "center",
              marginTop: 6,
            }}
          >
            O(N{"\u00B2"}) {"\u2192"} O(N)
            <br />
            memory
          </div>
        </div>

        {/* Flash Attention (right) */}
        <div
          style={{
            flex: 1,
            opacity: rightOpacity,
          }}
        >
          <div
            style={{
              fontSize: 14,
              color: THEME.colors.nvidiaGreen,
              fontFamily: fontFamilyBody,
              fontWeight: 700,
              letterSpacing: "0.5px",
              marginBottom: 12,
            }}
          >
            FLASH ATTENTION
          </div>

          <div
            style={{
              padding: "20px 24px",
              backgroundColor: "rgba(118,185,0,0.08)",
              border: `2px solid ${THEME.colors.nvidiaGreen}50`,
              borderRadius: 12,
            }}
          >
            {FLASH_STEPS.map((step, i) => {
              const stepDelay = rightDelay + 0.5 * fps + i * 0.5 * fps;
              const stepOpacity = interpolate(
                frame - stepDelay,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              return (
                <div
                  key={step.label}
                  style={{
                    opacity: stepOpacity,
                    display: "flex",
                    alignItems: "center",
                    gap: 14,
                    marginBottom: i < FLASH_STEPS.length - 1 ? 14 : 0,
                  }}
                >
                  <div
                    style={{
                      width: 28,
                      height: 28,
                      borderRadius: 14,
                      backgroundColor: `${THEME.colors.nvidiaGreen}20`,
                      border: `2px solid ${THEME.colors.nvidiaGreen}60`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 13,
                      fontWeight: 700,
                      color: THEME.colors.nvidiaGreen,
                      fontFamily: fontFamilyBody,
                      flexShrink: 0,
                    }}
                  >
                    {i + 1}
                  </div>
                  <div>
                    <span
                      style={{
                        fontSize: 20,
                        fontWeight: 700,
                        color: THEME.colors.nvidiaGreen,
                        fontFamily: fontFamilyCode,
                      }}
                    >
                      {step.label}
                    </span>
                    <span
                      style={{
                        fontSize: 16,
                        color: THEME.colors.textMuted,
                        fontFamily: fontFamilyBody,
                        marginLeft: 12,
                      }}
                    >
                      {step.detail}
                    </span>
                  </div>
                </div>
              );
            })}

            {/* Memory badge */}
            <div
              style={{
                marginTop: 18,
                padding: "10px 18px",
                backgroundColor: "rgba(118,185,0,0.12)",
                borderRadius: 8,
                display: "inline-block",
                opacity: interpolate(
                  frame - (rightDelay + 2.5 * fps),
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
              }}
            >
              <span
                style={{
                  fontSize: 22,
                  fontFamily: fontFamilyCode,
                  color: THEME.colors.nvidiaGreen,
                  fontWeight: 700,
                }}
              >
                Memory: O(N)
              </span>
              <span
                style={{
                  fontSize: 16,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  marginLeft: 12,
                }}
              >
                Never materializes N x N matrix
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom callout */}
      <div
        style={{
          marginTop: 16,
          padding: "14px 32px",
          backgroundColor: "rgba(118,185,0,0.08)",
          borderRadius: 10,
          borderLeft: `4px solid ${THEME.colors.nvidiaGreen}`,
          textAlign: "center",
          opacity: interpolate(
            frame - 8 * fps,
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
          Enables{" "}
          <span
            style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}
          >
            4K {"\u2192"} 128K+ context
          </span>
          . The most important CUDA optimization in modern LLMs.
        </span>
      </div>
    </SlideLayout>
  );
};
