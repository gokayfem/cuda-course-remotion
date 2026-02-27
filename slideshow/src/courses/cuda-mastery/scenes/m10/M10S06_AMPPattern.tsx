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

interface FlowBox {
  readonly label: string;
  readonly sublabel: string;
  readonly color: string;
  readonly precision: "FP32" | "FP16" | "both";
}

const FLOW_BOXES: readonly FlowBox[] = [
  { label: "FP32 Master Weights", sublabel: "persistent", color: THEME.colors.accentBlue, precision: "FP32" },
  { label: "Cast to FP16", sublabel: "", color: THEME.colors.textMuted, precision: "both" },
  { label: "Forward Pass", sublabel: "Tensor Cores!", color: THEME.colors.nvidiaGreen, precision: "FP16" },
  { label: "Loss (FP32)", sublabel: "x Scale Factor", color: THEME.colors.accentOrange, precision: "FP32" },
  { label: "Backward Pass", sublabel: "FP16 gradients", color: THEME.colors.nvidiaGreen, precision: "FP16" },
  { label: "Unscale Grads", sublabel: "Check Inf/NaN", color: THEME.colors.accentYellow, precision: "both" },
  { label: "Optimizer Step", sublabel: "FP32 update", color: THEME.colors.accentBlue, precision: "FP32" },
];

const BOX_WIDTH = 180;
const BOX_HEIGHT = 80;
const ARROW_WIDTH = 36;

export const M10S06_AMPPattern: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const totalFlowWidth = FLOW_BOXES.length * BOX_WIDTH + (FLOW_BOXES.length - 1) * ARROW_WIDTH;
  const availableWidth = 1920 - 2 * 72;
  const startX = Math.max(0, (availableWidth - totalFlowWidth) / 2);

  const loopArrowDelay = 6 * fps;
  const loopOpacity = interpolate(
    frame - loopArrowDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const codeDelay = 8 * fps;
  const codeOpacity = interpolate(
    frame - codeDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const precisionBarDelay = 7 * fps;
  const precisionOpacity = interpolate(
    frame - precisionBarDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={10}>
      <SlideTitle
        title="The AMP Training Pattern"
        subtitle="Automatic Mixed Precision: the full training loop"
      />

      <div style={{ flex: 1, position: "relative", width: 1776 }}>
        {/* Main flow diagram */}
        <div
          style={{
            position: "absolute",
            top: 20,
            left: startX,
            display: "flex",
            alignItems: "center",
          }}
        >
          {FLOW_BOXES.map((box, i) => {
            const boxDelay = 0.8 * fps + i * 0.4 * fps;
            const boxSpring = spring({
              frame: frame - boxDelay,
              fps,
              config: { damping: 200 },
            });
            const boxOpacity = interpolate(boxSpring, [0, 1], [0, 1]);
            const boxScale = interpolate(boxSpring, [0, 1], [0.85, 1]);

            const isFP16 = box.precision === "FP16";
            const bgAlpha = isFP16 ? "18" : "12";

            return (
              <React.Fragment key={box.label}>
                <div
                  style={{
                    width: BOX_WIDTH,
                    height: BOX_HEIGHT,
                    backgroundColor: `${box.color}${bgAlpha}`,
                    border: `2px solid ${box.color}60`,
                    borderRadius: 10,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    opacity: boxOpacity,
                    transform: `scale(${boxScale})`,
                    flexShrink: 0,
                    position: "relative",
                  }}
                >
                  <span
                    style={{
                      fontSize: 14,
                      fontWeight: 700,
                      color: box.color,
                      fontFamily: fontFamilyBody,
                      textAlign: "center",
                      lineHeight: 1.2,
                    }}
                  >
                    {box.label}
                  </span>
                  {box.sublabel && (
                    <span
                      style={{
                        fontSize: 11,
                        color: isFP16 ? THEME.colors.nvidiaGreen : THEME.colors.textMuted,
                        fontFamily: fontFamilyCode,
                        marginTop: 4,
                        textAlign: "center",
                        fontWeight: isFP16 ? 600 : 400,
                      }}
                    >
                      {box.sublabel}
                    </span>
                  )}

                  {/* Precision tag */}
                  <div
                    style={{
                      position: "absolute",
                      top: -10,
                      right: -6,
                      padding: "2px 6px",
                      backgroundColor: isFP16
                        ? `${THEME.colors.nvidiaGreen}30`
                        : `${THEME.colors.accentBlue}30`,
                      borderRadius: 4,
                      fontSize: 10,
                      fontWeight: 700,
                      color: isFP16 ? THEME.colors.nvidiaGreen : THEME.colors.accentBlue,
                      fontFamily: fontFamilyCode,
                      opacity: boxOpacity,
                    }}
                  >
                    {box.precision}
                  </div>
                </div>

                {/* Arrow */}
                {i < FLOW_BOXES.length - 1 && (
                  <div
                    style={{
                      width: ARROW_WIDTH,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      flexShrink: 0,
                      opacity: interpolate(
                        frame - (boxDelay + 0.2 * fps),
                        [0, 0.2 * fps],
                        [0, 0.6],
                        { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                      ),
                    }}
                  >
                    <span
                      style={{
                        color: THEME.colors.textMuted,
                        fontSize: 20,
                        fontFamily: fontFamilyBody,
                      }}
                    >
                      {"\u2192"}
                    </span>
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Loop-back arrow from Optimizer back to Master Weights */}
        <div
          style={{
            position: "absolute",
            top: BOX_HEIGHT + 30,
            left: startX,
            width: totalFlowWidth,
            height: 50,
            opacity: loopOpacity,
          }}
        >
          <svg width={totalFlowWidth} height={50}>
            <path
              d={`M ${totalFlowWidth - BOX_WIDTH / 2} 0
                  L ${totalFlowWidth - BOX_WIDTH / 2} 30
                  L ${BOX_WIDTH / 2} 30
                  L ${BOX_WIDTH / 2} 0`}
              fill="none"
              stroke={THEME.colors.accentBlue}
              strokeWidth={2}
              strokeDasharray="8,4"
            />
            <polygon
              points={`${BOX_WIDTH / 2 - 5},8 ${BOX_WIDTH / 2},0 ${BOX_WIDTH / 2 + 5},8`}
              fill={THEME.colors.accentBlue}
            />
            <text
              x={totalFlowWidth / 2}
              y={46}
              fill={THEME.colors.accentBlue}
              fontSize={13}
              fontFamily={fontFamilyBody}
              textAnchor="middle"
              fontWeight={600}
            >
              Loop: update master weights, repeat
            </text>
          </svg>
        </div>

        {/* Precision boundary legend */}
        <div
          style={{
            position: "absolute",
            top: 160,
            left: startX,
            width: totalFlowWidth,
            display: "flex",
            justifyContent: "center",
            gap: 40,
            opacity: precisionOpacity,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div
              style={{
                width: 16,
                height: 16,
                borderRadius: 3,
                backgroundColor: `${THEME.colors.nvidiaGreen}30`,
                border: `2px solid ${THEME.colors.nvidiaGreen}60`,
              }}
            />
            <span
              style={{
                fontSize: 15,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              FP16 (fast, Tensor Cores)
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div
              style={{
                width: 16,
                height: 16,
                borderRadius: 3,
                backgroundColor: `${THEME.colors.accentBlue}30`,
                border: `2px solid ${THEME.colors.accentBlue}60`,
              }}
            />
            <span
              style={{
                fontSize: 15,
                color: THEME.colors.accentBlue,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              FP32 (accurate, stable)
            </span>
          </div>
        </div>

        {/* PyTorch code snippet */}
        <div
          style={{
            position: "absolute",
            bottom: 20,
            left: 0,
            right: 0,
            display: "flex",
            justifyContent: "center",
            opacity: codeOpacity,
          }}
        >
          <div
            style={{
              padding: "14px 28px",
              backgroundColor: "rgba(13,17,23,0.9)",
              borderRadius: 10,
              border: "1px solid rgba(255,255,255,0.1)",
            }}
          >
            <span
              style={{
                fontSize: 16,
                fontFamily: fontFamilyCode,
                color: THEME.colors.syntaxKeyword,
              }}
            >
              {"with "}
            </span>
            <span
              style={{
                fontSize: 16,
                fontFamily: fontFamilyCode,
                color: THEME.colors.syntaxFunction,
              }}
            >
              torch.autocast
            </span>
            <span
              style={{
                fontSize: 16,
                fontFamily: fontFamilyCode,
                color: THEME.colors.textCode,
              }}
            >
              {"("}
            </span>
            <span
              style={{
                fontSize: 16,
                fontFamily: fontFamilyCode,
                color: THEME.colors.syntaxString,
              }}
            >
              {"'cuda'"}
            </span>
            <span
              style={{
                fontSize: 16,
                fontFamily: fontFamilyCode,
                color: THEME.colors.textCode,
              }}
            >
              {"): ... "}
            </span>
            <span
              style={{
                fontSize: 16,
                fontFamily: fontFamilyCode,
                color: THEME.colors.syntaxFunction,
              }}
            >
              scaler.scale
            </span>
            <span
              style={{
                fontSize: 16,
                fontFamily: fontFamilyCode,
                color: THEME.colors.textCode,
              }}
            >
              {"(loss)."}
            </span>
            <span
              style={{
                fontSize: 16,
                fontFamily: fontFamilyCode,
                color: THEME.colors.syntaxFunction,
              }}
            >
              backward
            </span>
            <span
              style={{
                fontSize: 16,
                fontFamily: fontFamilyCode,
                color: THEME.colors.textCode,
              }}
            >
              {"()"}
            </span>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
