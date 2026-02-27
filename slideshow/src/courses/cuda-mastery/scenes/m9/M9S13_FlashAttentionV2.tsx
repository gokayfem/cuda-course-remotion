import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

type Improvement = {
  change: string;
  benefit: string;
};

const improvements: Improvement[] = [
  {
    change: "Swap loop order: outer=K/V blocks, inner=Q blocks",
    benefit: "Better parallelism",
  },
  {
    change: "Reduce non-matmul FLOPs",
    benefit: "Less rescaling overhead",
  },
  {
    change: "Better work partitioning across warps",
    benefit: "Fewer syncs",
  },
];

type EcoItem = {
  name: string;
  description: string;
  color: string;
};

const ecosystem: EcoItem[] = [
  {
    name: "FlashAttention-2",
    description: "2x faster than v1",
    color: THEME.colors.nvidiaGreen,
  },
  {
    name: "FlashAttention-3",
    description: "Hopper (H100) optimized",
    color: THEME.colors.accentCyan,
  },
  {
    name: "xFormers",
    description: "Meta's attention library",
    color: THEME.colors.accentBlue,
  },
  {
    name: "FlexAttention",
    description: "PyTorch native (torch 2.5+)",
    color: THEME.colors.accentOrange,
  },
];

export const M9S13_FlashAttentionV2: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={9}
      leftWidth="50%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle title="Flash Attention v2 & Beyond" />

          <FadeInText
            text="Improvements over v1"
            delay={0.5 * fps}
            fontSize={18}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 16 }}
          />

          {improvements.map((item, i) => {
            const itemDelay = 1 * fps + i * 1 * fps;
            const itemSpring = spring({
              frame: frame - itemDelay,
              fps,
              config: { damping: 200 },
            });
            const itemOpacity = interpolate(itemSpring, [0, 1], [0, 1]);
            const itemX = interpolate(itemSpring, [0, 1], [-20, 0]);

            return (
              <div
                key={item.change}
                style={{
                  display: "flex",
                  gap: 14,
                  marginBottom: 16,
                  opacity: itemOpacity,
                  transform: `translateX(${itemX}px)`,
                  alignItems: "flex-start",
                }}
              >
                {/* Step number */}
                <div
                  style={{
                    width: 28,
                    height: 28,
                    borderRadius: 14,
                    backgroundColor: `${THEME.colors.nvidiaGreen}20`,
                    border: `1px solid ${THEME.colors.nvidiaGreen}50`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 14,
                    fontWeight: 700,
                    color: THEME.colors.nvidiaGreen,
                    fontFamily: fontFamilyCode,
                    flexShrink: 0,
                  }}
                >
                  {i + 1}
                </div>

                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      fontSize: 16,
                      color: THEME.colors.textPrimary,
                      fontFamily: fontFamilyBody,
                      fontWeight: 500,
                      lineHeight: 1.4,
                    }}
                  >
                    {item.change}
                  </div>
                  <div
                    style={{
                      fontSize: 14,
                      color: THEME.colors.nvidiaGreen,
                      fontFamily: fontFamilyBody,
                      marginTop: 4,
                    }}
                  >
                    {item.benefit}
                  </div>
                </div>

                {/* Animated arrow */}
                {i < improvements.length - 1 && (
                  <div
                    style={{
                      position: "absolute",
                      left: 13,
                      marginTop: 30,
                    }}
                  />
                )}
              </div>
            );
          })}

          {/* v1 -> v2 improvement arrow */}
          <div
            style={{
              marginTop: 20,
              display: "flex",
              alignItems: "center",
              gap: 12,
              opacity: interpolate(
                frame - 4 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div
              style={{
                padding: "6px 14px",
                backgroundColor: "rgba(255,255,255,0.05)",
                borderRadius: 6,
                fontSize: 14,
                fontWeight: 600,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyCode,
              }}
            >
              v1
            </div>
            <div
              style={{
                flex: 1,
                maxWidth: 200,
                height: 2,
                background: `linear-gradient(90deg, ${THEME.colors.textMuted}, ${THEME.colors.nvidiaGreen})`,
                position: "relative",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  right: 0,
                  top: -4,
                  width: 0,
                  height: 0,
                  borderLeft: `8px solid ${THEME.colors.nvidiaGreen}`,
                  borderTop: "5px solid transparent",
                  borderBottom: "5px solid transparent",
                }}
              />
            </div>
            <div
              style={{
                padding: "6px 14px",
                backgroundColor: `${THEME.colors.nvidiaGreen}15`,
                border: `1px solid ${THEME.colors.nvidiaGreen}40`,
                borderRadius: 6,
                fontSize: 14,
                fontWeight: 700,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyCode,
              }}
            >
              v2 (2x faster)
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ width: 520, marginTop: 80 }}>
          <FadeInText
            text="Flash Attention Ecosystem"
            delay={5 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentPurple}
            style={{ marginBottom: 20 }}
          />

          {ecosystem.map((item, i) => {
            const ecoDelay = 5.5 * fps + i * 0.5 * fps;
            const ecoSpring = spring({
              frame: frame - ecoDelay,
              fps,
              config: { damping: 200 },
            });
            const ecoOpacity = interpolate(ecoSpring, [0, 1], [0, 1]);

            return (
              <div
                key={item.name}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  marginBottom: 14,
                  padding: "10px 16px",
                  backgroundColor: `${item.color}08`,
                  borderLeft: `3px solid ${item.color}`,
                  borderRadius: 8,
                  opacity: ecoOpacity,
                }}
              >
                <span
                  style={{
                    fontSize: 16,
                    fontWeight: 700,
                    color: item.color,
                    fontFamily: fontFamilyBody,
                    width: 170,
                    flexShrink: 0,
                  }}
                >
                  {item.name}
                </span>
                <span
                  style={{
                    fontSize: 15,
                    color: THEME.colors.textSecondary,
                    fontFamily: fontFamilyBody,
                  }}
                >
                  {item.description}
                </span>
              </div>
            );
          })}

          {/* Bottom advice */}
          <div
            style={{
              marginTop: 24,
              padding: "14px 20px",
              backgroundColor: "rgba(179,136,255,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.accentPurple}30`,
              opacity: interpolate(
                frame - 9 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div
              style={{
                fontSize: 15,
                color: THEME.colors.accentPurple,
                fontFamily: fontFamilyBody,
                fontWeight: 500,
                lineHeight: 1.5,
              }}
            >
              You don't need to implement Flash Attention -- use the library. But understanding it makes you a better engineer.
            </div>
          </div>
        </div>
      }
    />
  );
};
