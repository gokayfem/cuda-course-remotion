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
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody } from "../../../../styles/fonts";

export const M8S06_TiledCode: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const perfOpacity = interpolate(
    frame - 10 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={8}
      leftWidth="52%"
      left={
        <div style={{ width: 600 }}>
          <SlideTitle
            title="Tiled Implementation"
            subtitle="Shared memory eliminates redundant global reads"
          />
          <CodeBlock
            delay={0.5 * fps}
            title="matmul_tiled.cu"
            fontSize={14}
            code={`__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];
float sum = 0.0f;

for (int t = 0; t < K; t += TILE) {
  // Load tile into shared memory
  As[ty][tx] = A[row*K + (t+tx)];
  Bs[ty][tx] = B[(t+ty)*N + col];
  __syncthreads();

  // Compute partial dot product
  for (int k = 0; k < TILE; k++)
    sum += As[ty][k] * Bs[k][tx];
  __syncthreads();
}
C[row * N + col] = sum;`}
            highlightLines={[1, 2, 7, 8, 9, 13]}
          />
        </div>
      }
      right={
        <div style={{ paddingTop: 60, width: 480 }}>
          <BulletPoint
            index={0}
            delay={3 * fps}
            text="TILE = 32: each tile loads 32x32 = 1024 elements"
            icon="1"
          />
          <BulletPoint
            index={1}
            delay={3 * fps}
            text="__syncthreads() ensures all loads complete"
            icon="2"
            highlight
          />
          <BulletPoint
            index={2}
            delay={3 * fps}
            text="Inner loop uses only shared memory (fast!)"
            icon="3"
          />
          <BulletPoint
            index={3}
            delay={3 * fps}
            text="Global reads reduced by 32x (TILE factor)"
            icon="4"
            highlight
          />

          {/* Performance badge */}
          <div
            style={{
              marginTop: 40,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.nvidiaGreen}40`,
              opacity: perfOpacity,
              width: 440,
            }}
          >
            <div
              style={{
                fontSize: 17,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                lineHeight: 1.5,
              }}
            >
              Performance:{" "}
              <span style={{ color: THEME.colors.accentOrange, fontWeight: 800 }}>
                ~500 GFLOPS
              </span>{" "}
              (~3% of peak)
            </div>
            <div
              style={{
                fontSize: 15,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
                marginTop: 4,
              }}
            >
              10x improvement over naive!
            </div>
          </div>
        </div>
      }
    />
  );
};
