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

const MATRIX_SIZE = 4;

const MatrixBlock: React.FC<{
  label: string;
  color: string;
  dtype: string;
  opacity: number;
  scale: number;
}> = ({ label, color, dtype, opacity, scale }) => (
  <div
    style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 6,
      opacity,
      transform: `scale(${scale})`,
    }}
  >
    <div
      style={{
        display: "grid",
        gridTemplateColumns: `repeat(${MATRIX_SIZE}, 1fr)`,
        gap: 2,
        width: 100,
      }}
    >
      {Array.from({ length: MATRIX_SIZE * MATRIX_SIZE }).map((_, i) => (
        <div
          key={i}
          style={{
            width: 22,
            height: 22,
            backgroundColor: `${color}25`,
            border: `1px solid ${color}50`,
            borderRadius: 2,
          }}
        />
      ))}
    </div>
    <div
      style={{
        fontSize: 16,
        fontWeight: 700,
        color,
        fontFamily: fontFamilyBody,
      }}
    >
      {label}
    </div>
    <div
      style={{
        fontSize: 12,
        color: THEME.colors.textMuted,
        fontFamily: fontFamilyCode,
      }}
    >
      {dtype}
    </div>
  </div>
);

const OperatorSymbol: React.FC<{
  symbol: string;
  opacity: number;
}> = ({ symbol, opacity }) => (
  <div
    style={{
      fontSize: 28,
      color: THEME.colors.textSecondary,
      fontWeight: 700,
      opacity,
      display: "flex",
      alignItems: "center",
      paddingTop: 10,
    }}
  >
    {symbol}
  </div>
);

export const M8S12_TensorCores: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const aSpring = spring({
    frame: frame - 1.5 * fps,
    fps,
    config: { damping: 200 },
  });
  const aOpacity = interpolate(aSpring, [0, 1], [0, 1]);
  const aScale = interpolate(aSpring, [0, 1], [0.85, 1]);

  const bSpring = spring({
    frame: frame - 2.5 * fps,
    fps,
    config: { damping: 200 },
  });
  const bOpacity = interpolate(bSpring, [0, 1], [0, 1]);
  const bScale = interpolate(bSpring, [0, 1], [0.85, 1]);

  const cSpring = spring({
    frame: frame - 3.5 * fps,
    fps,
    config: { damping: 200 },
  });
  const cOpacity = interpolate(cSpring, [0, 1], [0, 1]);
  const cScale = interpolate(cSpring, [0, 1], [0.85, 1]);

  const dSpring = spring({
    frame: frame - 5 * fps,
    fps,
    config: { damping: 200 },
  });
  const dOpacity = interpolate(dSpring, [0, 1], [0, 1]);
  const dScale = interpolate(dSpring, [0, 1], [0.85, 1]);

  const opOpacity = interpolate(
    frame - 3 * fps,
    [0, 0.3 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const resultGlow =
    frame > 5 * fps ? 0.3 + 0.2 * Math.sin((frame - 5 * fps) * 0.15) : 0;

  const bottomOpacity = interpolate(
    frame - 11 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={8}
      leftWidth="50%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle
            title="Tensor Cores -- The Ultimate Weapon"
            subtitle="4x4x4 matrix multiply-accumulate in ONE cycle"
          />

          {/* D = A x B + C diagram */}
          <div
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 16,
              marginTop: 24,
              padding: "24px 20px",
              backgroundColor: "rgba(13,17,23,0.6)",
              borderRadius: 12,
              border: `1px solid rgba(255,255,255,0.08)`,
              width: 700,
              justifyContent: "center",
            }}
          >
            <MatrixBlock
              label="A"
              color={THEME.colors.accentBlue}
              dtype="FP16"
              opacity={aOpacity}
              scale={aScale}
            />
            <OperatorSymbol symbol="x" opacity={opOpacity} />
            <MatrixBlock
              label="B"
              color={THEME.colors.accentPurple}
              dtype="FP16"
              opacity={bOpacity}
              scale={bScale}
            />
            <OperatorSymbol symbol="+" opacity={opOpacity} />
            <MatrixBlock
              label="C"
              color={THEME.colors.accentOrange}
              dtype="FP32"
              opacity={cOpacity}
              scale={cScale}
            />
            <OperatorSymbol symbol="=" opacity={opOpacity} />
            <div style={{ position: "relative" }}>
              <MatrixBlock
                label="D"
                color={THEME.colors.nvidiaGreen}
                dtype="FP32"
                opacity={dOpacity}
                scale={dScale}
              />
              {frame > 5 * fps && (
                <div
                  style={{
                    position: "absolute",
                    inset: -8,
                    borderRadius: 8,
                    border: `2px solid ${THEME.colors.nvidiaGreen}`,
                    opacity: resultGlow,
                    boxShadow: `0 0 20px ${THEME.colors.nvidiaGreen}40`,
                  }}
                />
              )}
            </div>
          </div>

          {/* One cycle callout */}
          <div
            style={{
              marginTop: 16,
              textAlign: "center",
              width: 700,
              opacity: dOpacity,
            }}
          >
            <span
              style={{
                fontSize: 18,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
                padding: "6px 20px",
                backgroundColor: "rgba(118,185,0,0.1)",
                borderRadius: 20,
              }}
            >
              D = A x B + C in a single clock cycle
            </span>
          </div>
        </div>
      }
      right={
        <div style={{ width: 480, marginTop: 60 }}>
          <BulletPoint
            text="A100: 312 TFLOPS FP16 (vs 19.5 TFLOPS FP32)"
            index={0}
            delay={4 * fps}
            highlight
          />
          <BulletPoint
            text="16x more throughput than regular CUDA cores"
            index={1}
            delay={4 * fps}
          />
          <BulletPoint
            text="Requires: FP16 inputs, specific matrix sizes (multiples of 16)"
            index={2}
            delay={4 * fps}
          />
          <BulletPoint
            text="WMMA API: wmmaLoadMatrix, wmmaComputeMMA, wmmaStoreMatrix"
            index={3}
            delay={4 * fps}
          />

          {/* Bottom insight */}
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
                lineHeight: 1.5,
              }}
            >
              cuBLAS automatically uses Tensor Cores when possible.
            </div>
            <div
              style={{
                fontSize: 14,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                marginTop: 4,
              }}
            >
              That's why it's so fast!
            </div>
          </div>
        </div>
      }
    />
  );
};
