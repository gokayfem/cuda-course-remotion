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

export const M10S03_WMMA: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const code = `wmma::fragment<wmma::matrix_a,
  16, 16, 16, half, wmma::row_major>
  a_frag;
wmma::fragment<wmma::matrix_b,
  16, 16, 16, half, wmma::col_major>
  b_frag;
wmma::fragment<wmma::accumulator,
  16, 16, 16, float> c_frag;

wmma::fill_fragment(c_frag, 0.0f);
wmma::load_matrix_sync(
  a_frag, A_ptr, lda);
wmma::load_matrix_sync(
  b_frag, B_ptr, ldb);
wmma::mma_sync(
  c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(
  C_ptr, c_frag, ldc,
  wmma::mem_row_major);`;

  const bottomDelay = 10 * fps;
  const bottomOpacity = interpolate(
    frame - bottomDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={10}
      leftWidth="55%"
      left={
        <div style={{ width: 860 }}>
          <SlideTitle
            title="WMMA API -- Programming Tensor Cores"
            subtitle="Warp Matrix Multiply-Accumulate intrinsics"
          />
          <CodeBlock
            code={code}
            title="wmma_example.cu"
            delay={0.5 * fps}
            fontSize={14}
            highlightLines={[1, 2, 3, 10, 11, 15, 16]}
          />

          {/* Bottom callout */}
          <div
            style={{
              marginTop: 20,
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
              WMMA = low-level Tensor Core access.{" "}
              <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>
                cuBLAS wraps this for you.
              </span>
            </span>
          </div>
        </div>
      }
      right={
        <div style={{ width: 540, marginTop: 80 }}>
          <BulletPoint
            text="Warp-level operation (all 32 threads cooperate)"
            index={0}
            delay={3 * fps}
            highlight
            subtext="Entire warp executes one matrix multiply together"
          />
          <BulletPoint
            text="Fragment = distributed across warp threads"
            index={1}
            delay={3 * fps}
            subtext="Each thread holds a portion of the matrix tile"
          />
          <BulletPoint
            text="16x16x16 tiles for FP16"
            index={2}
            delay={3 * fps}
            highlight
            subtext="M=16, N=16, K=16 -- other sizes available per architecture"
          />
          <BulletPoint
            text="Must use half precision inputs"
            index={3}
            delay={3 * fps}
            subtext="Accumulator can be FP32 for numerical stability"
          />
        </div>
      }
    />
  );
};
