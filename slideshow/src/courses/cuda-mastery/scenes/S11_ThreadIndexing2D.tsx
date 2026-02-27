import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../components/AnimatedText";
import { CodeBlock } from "../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../styles/fonts";

export const S11_ThreadIndexing2D: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const ROWS = 4;
  const COLS = 6;

  return (
    <SlideLayout variant="gradient" slideNumber={11} totalSlides={18}>
      <SlideTitle
        title="2D Thread Indexing — For Matrices"
        subtitle="Essential for images, feature maps, attention matrices"
      />

      <div style={{ display: "flex", gap: 48, flex: 1 }}>
        {/* Left: Matrix visualization */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text="6x4 Matrix (row-major in memory)"
            delay={0.5 * fps}
            fontSize={20}
            fontWeight={600}
            color={THEME.colors.textSecondary}
            style={{ marginBottom: 16 }}
          />

          {/* Matrix grid */}
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {/* Column headers */}
            <div style={{ display: "flex", gap: 4, marginLeft: 60 }}>
              {Array.from({ length: COLS }).map((_, c) => (
                <div
                  key={c}
                  style={{
                    width: 80,
                    textAlign: "center",
                    fontSize: 14,
                    color: THEME.colors.textMuted,
                    fontFamily: fontFamilyCode,
                    opacity: interpolate(
                      frame - 1 * fps,
                      [0, 0.3 * fps],
                      [0, 1],
                      { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                    ),
                  }}
                >
                  col {c}
                </div>
              ))}
            </div>

            {Array.from({ length: ROWS }).map((_, r) => (
              <div key={r} style={{ display: "flex", gap: 4, alignItems: "center" }}>
                {/* Row label */}
                <div
                  style={{
                    width: 56,
                    textAlign: "right",
                    fontSize: 14,
                    color: THEME.colors.textMuted,
                    fontFamily: fontFamilyCode,
                    paddingRight: 4,
                    opacity: interpolate(
                      frame - 1 * fps,
                      [0, 0.3 * fps],
                      [0, 1],
                      { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                    ),
                  }}
                >
                  row {r}
                </div>
                {Array.from({ length: COLS }).map((_, c) => {
                  const idx = r * COLS + c;
                  const cellDelay = 1.5 * fps + idx * 1.5;
                  const cellSpring = spring({
                    frame: frame - cellDelay,
                    fps,
                    config: { damping: 200 },
                  });

                  const isHighlighted = r === 2 && c === 3;

                  return (
                    <div
                      key={c}
                      style={{
                        width: 80,
                        height: 50,
                        backgroundColor: isHighlighted
                          ? `${THEME.colors.nvidiaGreen}30`
                          : `${THEME.colors.accentBlue}15`,
                        border: isHighlighted
                          ? `2px solid ${THEME.colors.nvidiaGreen}`
                          : `1px solid ${THEME.colors.accentBlue}40`,
                        borderRadius: 6,
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                        opacity: interpolate(cellSpring, [0, 1], [0, 1]),
                      }}
                    >
                      <span
                        style={{
                          fontSize: 18,
                          fontWeight: 700,
                          color: isHighlighted
                            ? THEME.colors.nvidiaGreen
                            : THEME.colors.textPrimary,
                          fontFamily: fontFamilyCode,
                        }}
                      >
                        {idx}
                      </span>
                      {isHighlighted && (
                        <span style={{ fontSize: 12, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyCode }}>
                          [2,3]
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>

          {/* Memory layout */}
          <div
            style={{
              marginTop: 24,
              opacity: interpolate(
                frame - 5 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <FadeInText
              text="In memory (row-major): [0, 1, 2, 3, 4, 5, 6, 7, 8, ...]"
              delay={5 * fps}
              fontSize={16}
              color={THEME.colors.textMuted}
            />
          </div>
        </div>

        {/* Right: Code and formula */}
        <div style={{ flex: 0.9 }}>
          <CodeBlock
            delay={3 * fps}
            title="2D Indexing Pattern"
            fontSize={17}
            code={`__global__ void kernel_2d(
    float *out, int width, int height
) {
    int col = blockIdx.x * blockDim.x
            + threadIdx.x;
    int row = blockIdx.y * blockDim.y
            + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;
        out[idx] = process(idx);
    }
}

// Launch with 2D config:
dim3 block(16, 16);  // 256 threads
dim3 grid(
    (width + 15) / 16,
    (height + 15) / 16
);
kernel_2d<<<grid, block>>>(d_out, W, H);`}
            highlightLines={[4, 5, 6, 7, 10, 16]}
          />

          <div
            style={{
              marginTop: 16,
              padding: "14px 18px",
              backgroundColor: "rgba(118,185,0,0.06)",
              borderRadius: 8,
              opacity: interpolate(
                frame - 7 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div style={{ fontFamily: fontFamilyCode, fontSize: 16, color: THEME.colors.nvidiaGreen, lineHeight: 2 }}>
              Example: [2,3] highlighted<br />
              idx = <span style={{ color: THEME.colors.accentOrange }}>2</span> * <span style={{ color: THEME.colors.accentBlue }}>6</span> + <span style={{ color: THEME.colors.accentPurple }}>3</span> = <span style={{ fontWeight: 700 }}>15</span>
            </div>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
