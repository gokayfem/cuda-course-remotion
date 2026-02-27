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

interface ComparisonItem {
  readonly text: string;
}

const TRITON_ITEMS: readonly ComparisonItem[] = [
  { text: "Element-wise and reduction ops" },
  { text: "Fused operations" },
  { text: "Rapid prototyping" },
  { text: "90% of custom kernel needs" },
  { text: "Python-native workflow" },
];

const CUDA_ITEMS: readonly ComparisonItem[] = [
  { text: "Tensor Core programming (WMMA)" },
  { text: "Warp-level primitives (__shfl)" },
  { text: "Complex shared memory patterns" },
  { text: "Maximum control and performance" },
  { text: "Integration with C++ codebases" },
];

interface ColumnCardProps {
  readonly title: string;
  readonly color: string;
  readonly items: readonly ComparisonItem[];
  readonly delay: number;
  readonly frame: number;
  readonly fps: number;
  readonly width: number;
}

const ColumnCard: React.FC<ColumnCardProps> = ({
  title,
  color,
  items,
  delay,
  frame: f,
  fps: fp,
  width,
}) => {
  const cardSpring = spring({
    frame: f - delay,
    fps: fp,
    config: { damping: 200 },
  });
  const cardOpacity = interpolate(cardSpring, [0, 1], [0, 1]);
  const cardY = interpolate(cardSpring, [0, 1], [30, 0]);

  return (
    <div
      style={{
        width,
        backgroundColor: `${color}08`,
        border: `2px solid ${color}40`,
        borderRadius: 14,
        padding: "24px 28px",
        opacity: cardOpacity,
        transform: `translateY(${cardY}px)`,
      }}
    >
      {/* Title */}
      <div
        style={{
          fontSize: 26,
          fontWeight: 800,
          color,
          fontFamily: fontFamilyBody,
          marginBottom: 24,
          textAlign: "center",
        }}
      >
        {title}
      </div>

      {/* Items */}
      {items.map((item, i) => {
        const itemDelay = delay + 0.5 * fp + i * 0.25 * fp;
        const itemSpring = spring({
          frame: f - itemDelay,
          fps: fp,
          config: { damping: 200 },
        });
        const itemOpacity = interpolate(itemSpring, [0, 1], [0, 1]);
        const itemX = interpolate(itemSpring, [0, 1], [-16, 0]);

        return (
          <div
            key={item.text}
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 12,
              marginBottom: 16,
              opacity: itemOpacity,
              transform: `translateX(${itemX}px)`,
            }}
          >
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: 4,
                backgroundColor: color,
                marginTop: 8,
                flexShrink: 0,
              }}
            />
            <span
              style={{
                fontSize: 20,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
                lineHeight: 1.5,
              }}
            >
              {item.text}
            </span>
          </div>
        );
      })}
    </div>
  );
};

export const M10S09_TritonVsCuda: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const perfDelay = 8 * fps;
  const perfOpacity = interpolate(
    frame - perfDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const perfBarDelay = 8.5 * fps;
  const tritonBarWidth = interpolate(
    frame - perfBarDelay,
    [0, 1 * fps],
    [0, 87.5],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const cudaBarWidth = interpolate(
    frame - perfBarDelay,
    [0, 1 * fps],
    [0, 100],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={10}>
      <SlideTitle
        title="Triton vs CUDA -- When to Use Which"
        subtitle="Choosing the right tool for your GPU kernel"
      />

      <div style={{ flex: 1, position: "relative", width: 1776 }}>
        {/* Two columns */}
        <div
          style={{
            display: "flex",
            gap: 48,
            justifyContent: "center",
            marginTop: 8,
          }}
        >
          <ColumnCard
            title="Use Triton"
            color={THEME.colors.nvidiaGreen}
            items={TRITON_ITEMS}
            delay={0.5 * fps}
            frame={frame}
            fps={fps}
            width={700}
          />
          <ColumnCard
            title="Use CUDA"
            color={THEME.colors.accentBlue}
            items={CUDA_ITEMS}
            delay={1 * fps}
            frame={frame}
            fps={fps}
            width={700}
          />
        </div>

        {/* Performance comparison bar */}
        <div
          style={{
            position: "absolute",
            bottom: 20,
            left: 0,
            right: 0,
            opacity: perfOpacity,
          }}
        >
          <div
            style={{
              maxWidth: 900,
              margin: "0 auto",
              padding: "18px 28px",
              backgroundColor: "rgba(255,255,255,0.04)",
              borderRadius: 12,
              border: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <div
              style={{
                fontSize: 16,
                fontWeight: 600,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                marginBottom: 14,
                textAlign: "center",
              }}
            >
              Typical Performance Comparison
            </div>

            {/* Triton bar */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                marginBottom: 10,
              }}
            >
              <span
                style={{
                  width: 80,
                  fontSize: 14,
                  fontWeight: 700,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyCode,
                  flexShrink: 0,
                  textAlign: "right",
                }}
              >
                Triton
              </span>
              <div
                style={{
                  flex: 1,
                  height: 24,
                  backgroundColor: "rgba(255,255,255,0.05)",
                  borderRadius: 6,
                  overflow: "hidden",
                  position: "relative",
                }}
              >
                <div
                  style={{
                    width: `${tritonBarWidth}%`,
                    height: "100%",
                    background: `linear-gradient(90deg, ${THEME.colors.nvidiaGreen}80, ${THEME.colors.nvidiaGreen})`,
                    borderRadius: 6,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "flex-end",
                    paddingRight: 8,
                  }}
                >
                  {tritonBarWidth > 20 && (
                    <span
                      style={{
                        fontSize: 12,
                        fontWeight: 700,
                        color: THEME.colors.textPrimary,
                        fontFamily: fontFamilyCode,
                      }}
                    >
                      80-95%
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* CUDA bar */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
              }}
            >
              <span
                style={{
                  width: 80,
                  fontSize: 14,
                  fontWeight: 700,
                  color: THEME.colors.accentBlue,
                  fontFamily: fontFamilyCode,
                  flexShrink: 0,
                  textAlign: "right",
                }}
              >
                CUDA
              </span>
              <div
                style={{
                  flex: 1,
                  height: 24,
                  backgroundColor: "rgba(255,255,255,0.05)",
                  borderRadius: 6,
                  overflow: "hidden",
                  position: "relative",
                }}
              >
                <div
                  style={{
                    width: `${cudaBarWidth}%`,
                    height: "100%",
                    background: `linear-gradient(90deg, ${THEME.colors.accentBlue}80, ${THEME.colors.accentBlue})`,
                    borderRadius: 6,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "flex-end",
                    paddingRight: 8,
                  }}
                >
                  {cudaBarWidth > 20 && (
                    <span
                      style={{
                        fontSize: 12,
                        fontWeight: 700,
                        color: THEME.colors.textPrimary,
                        fontFamily: fontFamilyCode,
                      }}
                    >
                      100%
                    </span>
                  )}
                </div>
              </div>
            </div>

            <div
              style={{
                fontSize: 14,
                color: THEME.colors.textMuted,
                fontFamily: fontFamilyBody,
                marginTop: 10,
                textAlign: "center",
              }}
            >
              Triton achieves{" "}
              <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>
                80-95%
              </span>{" "}
              of hand-tuned CUDA for most kernels
            </div>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
