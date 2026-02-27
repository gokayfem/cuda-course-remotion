import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../components/AnimatedText";
import { CodeBlock } from "../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../styles/fonts";

export const S08_VectorAdd: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="code" slideNumber={8} totalSlides={18}>
      <SlideTitle
        title='Vector Addition — The "Hello World" of CUDA'
        subtitle="Each thread computes ONE element: C[i] = A[i] + B[i]"
      />

      <div style={{ display: "flex", gap: 36, flex: 1 }}>
        {/* Code */}
        <div style={{ flex: 1.3 }}>
          <CodeBlock
            delay={0.5 * fps}
            title="02_vector_add.cu — The Kernel"
            fontSize={18}
            code={`__global__ void vector_add(
    const float *a, const float *b,
    float *c, int n
) {
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;

    if (idx < n) {       // Bounds check!
        c[idx] = a[idx] + b[idx];
    }
}

// Launch configuration
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;
vector_add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);`}
            highlightLines={[5, 6, 8, 14, 15, 16]}
          />
        </div>

        {/* Explanation */}
        <div style={{ flex: 0.7 }}>
          <FadeInText
            text="Thread Index Formula"
            delay={3 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.nvidiaGreen}
            style={{ marginBottom: 12 }}
          />

          {/* Index calculation visual */}
          <div
            style={{
              padding: "16px 20px",
              backgroundColor: "rgba(118,185,0,0.06)",
              borderRadius: 10,
              marginBottom: 20,
              opacity: interpolate(
                frame - 3.5 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div style={{ fontFamily: fontFamilyCode, fontSize: 18, color: THEME.colors.nvidiaGreen, lineHeight: 2 }}>
              idx = blockIdx.x * blockDim.x<br />
              {"    "}+ threadIdx.x
            </div>
            <div style={{ fontFamily: fontFamilyCode, fontSize: 16, color: THEME.colors.textMuted, marginTop: 12, lineHeight: 1.8 }}>
              Example: block=3, thread=7,<br />
              blockDim=32<br />
              idx = 3 * 32 + 7 = <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>103</span>
            </div>
          </div>

          <FadeInText
            text="Why Bounds Check?"
            delay={5 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentRed}
            style={{ marginBottom: 12 }}
          />

          <div
            style={{
              padding: "14px 18px",
              backgroundColor: "rgba(255,82,82,0.06)",
              borderRadius: 10,
              opacity: interpolate(
                frame - 5.5 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div style={{ fontFamily: fontFamilyCode, fontSize: 15, color: THEME.colors.textSecondary, lineHeight: 1.8 }}>
              N = 1,000,003<br />
              blocks = ceil(1000003/256) = 3907<br />
              total threads = 3907 * 256 = 1,000,192<br />
              <span style={{ color: THEME.colors.accentRed }}>extra 189 threads!</span><br />
              Without bounds check → <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>crash</span>
            </div>
          </div>

          <FadeInText
            text="Why blockSize = 256?"
            delay={7 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentBlue}
            style={{ marginBottom: 8, marginTop: 16 }}
          />
          <FadeInText
            text="Must be multiple of 32 (warp size). 256 gives good occupancy on most GPUs. Max is 1024."
            delay={7.3 * fps}
            fontSize={16}
            color={THEME.colors.textSecondary}
          />
        </div>
      </div>
    </SlideLayout>
  );
};
