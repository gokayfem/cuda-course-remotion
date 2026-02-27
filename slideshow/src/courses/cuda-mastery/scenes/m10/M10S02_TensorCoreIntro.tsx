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
const CELL = 36;
const GAP = 3;

interface MatrixProps {
  readonly label: string;
  readonly type: string;
  readonly color: string;
  readonly opacity: number;
  readonly offsetX: number;
}

const MatrixGrid: React.FC<MatrixProps> = ({ label, type, color, opacity, offsetX }) => (
  <div
    style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      opacity,
      transform: `translateX(${offsetX}px)`,
    }}
  >
    <div
      style={{
        fontSize: 14,
        fontWeight: 700,
        color,
        fontFamily: fontFamilyBody,
        marginBottom: 6,
      }}
    >
      {label}
    </div>
    <div
      style={{
        display: "grid",
        gridTemplateColumns: `repeat(${MATRIX_SIZE}, ${CELL}px)`,
        gap: GAP,
      }}
    >
      {Array.from({ length: MATRIX_SIZE * MATRIX_SIZE }).map((_, idx) => (
        <div
          key={idx}
          style={{
            width: CELL,
            height: CELL,
            backgroundColor: `${color}25`,
            border: `1px solid ${color}50`,
            borderRadius: 3,
          }}
        />
      ))}
    </div>
    <div
      style={{
        fontSize: 12,
        color: THEME.colors.textMuted,
        fontFamily: fontFamilyCode,
        marginTop: 4,
      }}
    >
      {type}
    </div>
  </div>
);

const OpSymbol: React.FC<{ text: string; opacity: number }> = ({ text, opacity }) => (
  <div
    style={{
      fontSize: 28,
      fontWeight: 800,
      color: THEME.colors.textSecondary,
      fontFamily: fontFamilyBody,
      alignSelf: "center",
      opacity,
      padding: "0 6px",
    }}
  >
    {text}
  </div>
);

export const M10S02_TensorCoreIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const matADelay = 1 * fps;
  const matASpring = spring({ frame: frame - matADelay, fps, config: { damping: 200 } });
  const matAOpacity = interpolate(matASpring, [0, 1], [0, 1]);

  const matBDelay = 1.5 * fps;
  const matBSpring = spring({ frame: frame - matBDelay, fps, config: { damping: 200 } });
  const matBOpacity = interpolate(matBSpring, [0, 1], [0, 1]);

  const matCDelay = 2 * fps;
  const matCSpring = spring({ frame: frame - matCDelay, fps, config: { damping: 200 } });
  const matCOpacity = interpolate(matCSpring, [0, 1], [0, 1]);

  const matDDelay = 2.8 * fps;
  const matDSpring = spring({ frame: frame - matDDelay, fps, config: { damping: 200 } });
  const matDOpacity = interpolate(matDSpring, [0, 1], [0, 1]);

  const combineDelay = 3.2 * fps;
  const combineGlow = interpolate(
    frame - combineDelay,
    [0, 0.3 * fps, 0.8 * fps],
    [0, 1, 0.4],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const cycleDelay = 3.5 * fps;
  const cycleOpacity = interpolate(
    frame - cycleDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const bottomDelay = 8 * fps;
  const bottomOpacity = interpolate(
    frame - bottomDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={10}
      leftWidth="55%"
      left={
        <div style={{ width: 860 }}>
          <SlideTitle
            title="Tensor Cores -- Dedicated Matrix Units"
            subtitle="Hardware-accelerated matrix multiply-accumulate"
          />

          {/* Matrix equation: A x B + C = D */}
          <div
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 8,
              marginTop: 16,
              flexWrap: "wrap",
            }}
          >
            <MatrixGrid
              label="A (4x4)"
              type="FP16"
              color={THEME.colors.accentBlue}
              opacity={matAOpacity}
              offsetX={0}
            />
            <OpSymbol text="x" opacity={matBOpacity} />
            <MatrixGrid
              label="B (4x4)"
              type="FP16"
              color={THEME.colors.accentPurple}
              opacity={matBOpacity}
              offsetX={0}
            />
            <OpSymbol text="+" opacity={matCOpacity} />
            <MatrixGrid
              label="C (4x4)"
              type="FP32"
              color={THEME.colors.accentOrange}
              opacity={matCOpacity}
              offsetX={0}
            />
            <OpSymbol text="=" opacity={matDOpacity} />
            <MatrixGrid
              label="D (4x4)"
              type="FP32"
              color={THEME.colors.nvidiaGreen}
              opacity={matDOpacity}
              offsetX={0}
            />
          </div>

          {/* Glow highlight on result */}
          {combineGlow > 0 && (
            <div
              style={{
                marginTop: 12,
                height: 3,
                background: `linear-gradient(90deg, transparent, ${THEME.colors.nvidiaGreen}${Math.round(combineGlow * 200).toString(16).padStart(2, "0")}, transparent)`,
                borderRadius: 2,
              }}
            />
          )}

          {/* Cycle label */}
          <div
            style={{
              marginTop: 16,
              padding: "10px 20px",
              backgroundColor: "rgba(118,185,0,0.1)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              textAlign: "center",
              opacity: cycleOpacity,
            }}
          >
            <span
              style={{
                fontSize: 20,
                fontWeight: 700,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
              }}
            >
              64 FMA operations in 1 cycle
            </span>
          </div>

          {/* Bottom insight */}
          <div
            style={{
              marginTop: 24,
              padding: "12px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              borderLeft: `4px solid ${THEME.colors.nvidiaGreen}`,
              opacity: bottomOpacity,
            }}
          >
            <span
              style={{
                fontSize: 18,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
              }}
            >
              Tensor Cores are why modern GPUs are{" "}
              <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>
                ML powerhouses
              </span>
            </span>
          </div>
        </div>
      }
      right={
        <div style={{ width: 540, marginTop: 80 }}>
          <BulletPoint
            text="Dedicated hardware for matrix multiply-accumulate"
            index={0}
            delay={4 * fps}
            highlight
          />
          <BulletPoint
            text="Volta: FP16, Turing: +INT8, Ampere: +TF32/BF16, Hopper: +FP8"
            index={1}
            delay={4 * fps}
            subtext="Each generation adds more data types"
          />
          <BulletPoint
            text="A100: 312 TFLOPS FP16 vs 19.5 TFLOPS FP32 (16x!)"
            index={2}
            delay={4 * fps}
            highlight
            subtext="Tensor Cores deliver an order of magnitude more throughput"
          />
          <BulletPoint
            text="Used automatically by cuBLAS, cuDNN"
            index={3}
            delay={4 * fps}
            subtext="Library calls dispatch to Tensor Cores when possible"
          />
        </div>
      }
    />
  );
};
