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

interface StepDef {
  readonly label: string;
  readonly runningMax: string;
  readonly denom: string;
  readonly note: string;
}

const STEPS: readonly StepDef[] = [
  {
    label: "x[0] = 3.0",
    runningMax: "m = 3.0",
    denom: "d = exp(3.0 - 3.0) = 1.0",
    note: "Initialize",
  },
  {
    label: "x[1] = 7.0",
    runningMax: "m = 7.0",
    denom: "d = 1.0 * exp(3.0-7.0) + exp(7.0-7.0) = 1.018",
    note: "Correction factor applied",
  },
  {
    label: "x[2] = 1.0",
    runningMax: "m = 7.0",
    denom: "d = 1.018 + exp(1.0-7.0) = 1.021",
    note: "Max unchanged",
  },
  {
    label: "x[3] = 5.0",
    runningMax: "m = 7.0",
    denom: "d = 1.021 + exp(5.0-7.0) = 1.156",
    note: "Accumulate",
  },
];

export const M9S04_OnlineSoftmax: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={9}
      leftWidth="55%"
      left={
        <div style={{ width: 860 }}>
          <SlideTitle
            title="Online Softmax \u2014 Single Pass"
            subtitle="Compute max and denominator simultaneously as we scan"
          />

          {/* Algorithm steps */}
          <div
            style={{
              fontSize: 14,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
              marginBottom: 12,
              fontWeight: 600,
              letterSpacing: "0.5px",
              opacity: interpolate(
                frame - 0.8 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            STEP-BY-STEP WITH RUNNING MAX & DENOMINATOR
          </div>

          {STEPS.map((step, i) => {
            const stepDelay = 1.5 * fps + i * 1.5 * fps;
            const stepSpring = spring({
              frame: frame - stepDelay,
              fps,
              config: { damping: 200 },
            });
            const opacity = interpolate(stepSpring, [0, 1], [0, 1]);
            const translateY = interpolate(stepSpring, [0, 1], [15, 0]);

            const isActive =
              frame >= stepDelay &&
              (i === STEPS.length - 1 || frame < stepDelay + 1.5 * fps);
            const borderColor = isActive
              ? THEME.colors.nvidiaGreen
              : "rgba(255,255,255,0.1)";
            const bgColor = isActive
              ? "rgba(118,185,0,0.06)"
              : "rgba(255,255,255,0.02)";

            return (
              <div
                key={step.label}
                style={{
                  opacity,
                  transform: `translateY(${translateY}px)`,
                  marginBottom: 10,
                  padding: "12px 18px",
                  backgroundColor: bgColor,
                  border: `2px solid ${borderColor}`,
                  borderRadius: 10,
                  display: "flex",
                  gap: 20,
                  alignItems: "center",
                }}
              >
                <div
                  style={{
                    width: 90,
                    flexShrink: 0,
                    fontSize: 16,
                    fontFamily: fontFamilyCode,
                    color: THEME.colors.accentBlue,
                    fontWeight: 700,
                  }}
                >
                  {step.label}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontSize: 15,
                      fontFamily: fontFamilyCode,
                      color: THEME.colors.accentOrange,
                    }}
                  >
                    {step.runningMax}
                  </div>
                  <div
                    style={{
                      fontSize: 14,
                      fontFamily: fontFamilyCode,
                      color: THEME.colors.textSecondary,
                      marginTop: 3,
                    }}
                  >
                    {step.denom}
                  </div>
                </div>
                <div
                  style={{
                    width: 140,
                    flexShrink: 0,
                    fontSize: 13,
                    fontFamily: fontFamilyBody,
                    color: THEME.colors.textMuted,
                    textAlign: "right",
                  }}
                >
                  {step.note}
                </div>
              </div>
            );
          })}

          {/* Final normalize step */}
          <div
            style={{
              marginTop: 12,
              padding: "12px 18px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              borderLeft: `4px solid ${THEME.colors.nvidiaGreen}`,
              opacity: interpolate(
                frame - 8 * fps,
                [0, 0.4 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <span
              style={{
                fontSize: 17,
                fontFamily: fontFamilyCode,
                color: THEME.colors.nvidiaGreen,
                fontWeight: 700,
              }}
            >
              Pass 2: out[i] = exp(x[i] - m) / d
            </span>
            <span
              style={{
                fontSize: 14,
                fontFamily: fontFamilyBody,
                color: THEME.colors.textMuted,
                marginLeft: 16,
              }}
            >
              (only 2 passes total)
            </span>
          </div>
        </div>
      }
      right={
        <div style={{ width: 580, marginTop: 80 }}>
          <BulletPoint
            text="Running max tracks global maximum incrementally"
            index={0}
            delay={2 * fps}
            highlight
            subtext="No separate pass needed for max computation"
          />
          <BulletPoint
            text="Correction factor rescales old denominator"
            index={1}
            delay={2 * fps}
            subtext="d_new = d_old * exp(m_old - m_new) + exp(x_i - m_new)"
          />
          <BulletPoint
            text="Total: 2 passes instead of 3"
            index={2}
            delay={2 * fps}
            highlight
            subtext="Pass 1: online scan (max + denom). Pass 2: normalize"
          />
          <BulletPoint
            text="Foundation for Flash Attention's tiled approach"
            index={3}
            delay={2 * fps}
            subtext="Online softmax enables block-by-block attention computation"
          />

          {/* Bottom callout */}
          <div
            style={{
              marginTop: 36,
              padding: "14px 24px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              borderLeft: `4px solid ${THEME.colors.nvidiaGreen}`,
              opacity: interpolate(
                frame - 10 * fps,
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
              <span
                style={{
                  color: THEME.colors.nvidiaGreen,
                  fontWeight: 700,
                }}
              >
                1.5-2x faster
              </span>{" "}
              than 3-pass for long sequences
            </span>
          </div>
        </div>
      }
    />
  );
};
