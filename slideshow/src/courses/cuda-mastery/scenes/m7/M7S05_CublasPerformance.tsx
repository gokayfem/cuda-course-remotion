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

interface MatrixSize {
  label: string;
  tflops: number;
  delay: number;
}

const MATRIX_SIZES: MatrixSize[] = [
  { label: "256 x 256", tflops: 2, delay: 0 },
  { label: "512 x 512", tflops: 8, delay: 0.4 },
  { label: "1024 x 1024", tflops: 15, delay: 0.8 },
  { label: "2048 x 2048", tflops: 18, delay: 1.2 },
  { label: "4096 x 4096", tflops: 19, delay: 1.6 },
];

const PEAK_TFLOPS = 19.5;
const BAR_MAX_WIDTH = 400;
const BAR_HEIGHT = 32;

export const M7S05_CublasPerformance: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Peak line
  const peakOpacity = interpolate(
    frame - 1 * fps,
    [0, 0.5 * fps],
    [0, 0.6],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Bottom rule of thumb
  const bottomOpacity = interpolate(
    frame - 10 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={7}
      leftWidth="55%"
      left={
        <div style={{ width: 620 }}>
          <SlideTitle
            title="cuBLAS Performance"
            subtitle="TFLOPS vs matrix size (A100, FP32 SGEMM)"
          />

          <div style={{ position: "relative", marginTop: 8, width: 580 }}>
            {/* Peak line */}
            <div
              style={{
                position: "absolute",
                right: 40,
                top: 0,
                bottom: 20,
                width: 2,
                borderLeft: `2px dashed ${THEME.colors.textMuted}`,
                opacity: peakOpacity,
              }}
            />
            <div
              style={{
                position: "absolute",
                right: 0,
                top: -8,
                fontSize: 12,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                color: THEME.colors.textMuted,
                opacity: peakOpacity,
                width: 36,
                textAlign: "center",
              }}
            >
              Peak
            </div>

            {/* Bars */}
            {MATRIX_SIZES.map((size, i) => {
              const barDelay = (2 + size.delay) * fps;
              const barSpring = spring({
                frame: frame - barDelay,
                fps,
                config: { damping: 200, stiffness: 60 },
              });

              const barWidth = interpolate(
                barSpring,
                [0, 1],
                [0, (size.tflops / PEAK_TFLOPS) * BAR_MAX_WIDTH]
              );
              const labelOpacity = interpolate(barSpring, [0, 1], [0, 1]);

              // Color gradient from blue to green based on performance
              const greenAmount = size.tflops / PEAK_TFLOPS;
              const barColor =
                greenAmount > 0.8
                  ? THEME.colors.nvidiaGreen
                  : greenAmount > 0.5
                    ? THEME.colors.accentBlue
                    : greenAmount > 0.3
                      ? THEME.colors.accentOrange
                      : THEME.colors.accentRed;

              return (
                <div
                  key={`size-${i}`}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    marginBottom: 14,
                    width: 560,
                  }}
                >
                  <div
                    style={{
                      width: 110,
                      fontSize: 14,
                      fontFamily: fontFamilyCode,
                      color: THEME.colors.textSecondary,
                      textAlign: "right",
                      flexShrink: 0,
                      opacity: labelOpacity,
                    }}
                  >
                    {size.label}
                  </div>
                  <div
                    style={{
                      width: barWidth,
                      height: BAR_HEIGHT,
                      backgroundColor: barColor,
                      borderRadius: 5,
                      minWidth: 4,
                    }}
                  />
                  <span
                    style={{
                      fontSize: 14,
                      fontFamily: fontFamilyCode,
                      fontWeight: 700,
                      color: barColor,
                      opacity: labelOpacity,
                      whiteSpace: "nowrap",
                    }}
                  >
                    ~{size.tflops} TFLOPS
                  </span>
                </div>
              );
            })}
          </div>

          {/* Bottom rule */}
          <div
            style={{
              marginTop: 20,
              padding: "12px 18px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: bottomOpacity,
              width: 560,
            }}
          >
            <span
              style={{
                fontSize: 16,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              Rule of thumb: if{" "}
              <span style={{ color: THEME.colors.nvidiaGreen }}>N &gt; 512</span>,
              cuBLAS is near-optimal
            </span>
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 60, width: 420 }}>
          <BulletPoint
            index={0}
            delay={4 * fps}
            text="Tensor Cores: FP16 = 312 TFLOPS (A100)"
            icon="1"
            highlight
          />
          <BulletPoint
            index={1}
            delay={4 * fps}
            text="TF32: 156 TFLOPS (automatic on Ampere+)"
            icon="2"
          />
          <BulletPoint
            index={2}
            delay={4 * fps}
            text="Batched GEMM for small matrices (attention heads)"
            icon="3"
          />
          <BulletPoint
            index={3}
            delay={4 * fps}
            text="cublasGemmEx for mixed-precision"
            icon="4"
            highlight
          />
        </div>
      }
    />
  );
};
