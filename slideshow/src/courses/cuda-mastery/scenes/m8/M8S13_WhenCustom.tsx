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
import { fontFamilyBody } from "../../../../styles/fonts";

type ColumnItem = {
  text: string;
};

const cublasItems: ColumnItem[] = [
  { text: "Standard dense GEMM" },
  { text: "Batch of same-size matrices" },
  { text: "Standard dtypes (FP16, FP32, TF32)" },
  { text: "Production code" },
];

const customItems: ColumnItem[] = [
  { text: "Fused operations (matmul + activation + bias)" },
  { text: "Non-standard sparsity patterns" },
  { text: "Custom quantization (INT4, FP8)" },
  { text: "Specialized shapes (very tall/thin matrices)" },
  { text: "Learning / understanding GPU performance" },
];

const ColumnCard: React.FC<{
  title: string;
  items: ColumnItem[];
  color: string;
  startDelay: number;
}> = ({ title, items, color, startDelay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const headerSpring = spring({
    frame: frame - startDelay,
    fps,
    config: { damping: 200 },
  });
  const headerOpacity = interpolate(headerSpring, [0, 1], [0, 1]);
  const headerScale = interpolate(headerSpring, [0, 1], [0.95, 1]);

  return (
    <div
      style={{
        flex: 1,
        padding: "24px 28px",
        backgroundColor: `${color}06`,
        border: `2px solid ${color}40`,
        borderRadius: 12,
        opacity: headerOpacity,
        transform: `scale(${headerScale})`,
      }}
    >
      <div
        style={{
          fontSize: 24,
          fontWeight: 700,
          color,
          fontFamily: fontFamilyBody,
          marginBottom: 20,
          textAlign: "center",
          paddingBottom: 14,
          borderBottom: `1px solid ${color}30`,
        }}
      >
        {title}
      </div>

      {items.map((item, i) => {
        const itemDelay = startDelay + 0.5 * fps + i * 0.4 * fps;
        const itemSpring = spring({
          frame: frame - itemDelay,
          fps,
          config: { damping: 200 },
        });
        const itemOpacity = interpolate(itemSpring, [0, 1], [0, 1]);
        const itemX = interpolate(itemSpring, [0, 1], [-12, 0]);

        return (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 12,
              marginBottom: 14,
              opacity: itemOpacity,
              transform: `translateX(${itemX}px)`,
            }}
          >
            <div
              style={{
                width: 24,
                height: 24,
                borderRadius: 12,
                backgroundColor: `${color}20`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 13,
                fontWeight: 700,
                color,
                fontFamily: fontFamilyBody,
                flexShrink: 0,
                marginTop: 2,
              }}
            >
              {i + 1}
            </div>
            <span
              style={{
                fontSize: 18,
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

export const M8S13_WhenCustom: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const bottomOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={8}>
      <SlideTitle title="When to Write Custom MatMul?" />

      <div
        style={{
          display: "flex",
          gap: 32,
          flex: 1,
          width: 1776,
        }}
      >
        <ColumnCard
          title="Use cuBLAS"
          items={cublasItems}
          color={THEME.colors.nvidiaGreen}
          startDelay={1 * fps}
        />
        <ColumnCard
          title="Write Custom"
          items={customItems}
          color={THEME.colors.accentBlue}
          startDelay={1.5 * fps}
        />
      </div>

      {/* Bottom callout */}
      <div
        style={{
          marginTop: 20,
          padding: "16px 28px",
          backgroundColor: "rgba(118,185,0,0.08)",
          borderRadius: 10,
          border: `1px solid ${THEME.colors.nvidiaGreen}30`,
          opacity: bottomOpacity,
          width: 1776,
          textAlign: "center",
        }}
      >
        <span
          style={{
            fontSize: 20,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyBody,
            fontWeight: 700,
          }}
        >
          99% of the time: use cuBLAS. The 1% is what makes you a GPU expert.
        </span>
      </div>
    </SlideLayout>
  );
};
