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
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

const skeletonCode = `cudaStream_t streams[4];
float *h_in, *h_out;  // pinned
float *d_buf[4];
cudaMallocHost(&h_in, total_size);
cudaMallocHost(&h_out, total_size);
for (int i = 0; i < 4; i++) {
    cudaStreamCreate(&streams[i]);
    cudaMalloc(&d_buf[i], chunk_size);
}
// TODO: Launch pipeline`;

const solutionCode = `for (int i = 0; i < 4; i++) {
    int off = i * chunk_elems;
    cudaMemcpyAsync(d_buf[i], h_in + off,
        chunk_size, H2D, streams[i]);
    square<<<grid, block, 0, streams[i]>>>
        (d_buf[i], chunk_elems);
    cudaMemcpyAsync(h_out + off, d_buf[i],
        chunk_size, D2H, streams[i]);
}`;

const stages = [
  { num: "1", text: "H2D copy (cudaMemcpyAsync)", color: THEME.colors.accentBlue },
  { num: "2", text: "Kernel: square each element", color: THEME.colors.nvidiaGreen },
  { num: "3", text: "D2H copy (cudaMemcpyAsync)", color: THEME.colors.accentOrange },
];

export const M6S16_Exercise: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const showSolution = frame > 8 * fps;

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={6}
      leftWidth="50%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle
            title="Exercise: Build a Stream Pipeline"
            subtitle="Process N=4 chunks through 3 stages using separate streams"
          />

          {/* Stage cards */}
          <div
            style={{
              display: "flex",
              gap: 12,
              marginBottom: 18,
            }}
          >
            {stages.map((stage, i) => {
              const stageDelay = 1 * fps + i * 0.3 * fps;
              const stageSpring = spring({
                frame: frame - stageDelay,
                fps,
                config: { damping: 200 },
              });
              const stageOpacity = interpolate(stageSpring, [0, 1], [0, 1]);

              return (
                <div
                  key={stage.num}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "8px 14px",
                    backgroundColor: `${stage.color}10`,
                    border: `1px solid ${stage.color}50`,
                    borderRadius: 8,
                    opacity: stageOpacity,
                  }}
                >
                  <span
                    style={{
                      fontSize: 14,
                      fontWeight: 700,
                      color: stage.color,
                      fontFamily: fontFamilyBody,
                    }}
                  >
                    Stage {stage.num}:
                  </span>
                  <span
                    style={{
                      fontSize: 13,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyCode,
                    }}
                  >
                    {stage.text}
                  </span>
                </div>
              );
            })}
          </div>

          <FadeInText
            text="Use separate streams. Measure speedup vs serial."
            delay={2 * fps}
            fontSize={16}
            color={THEME.colors.textSecondary}
            style={{ marginBottom: 16 }}
          />

          <CodeBlock
            delay={2.5 * fps}
            title="stream_pipeline.cu"
            fontSize={15}
            code={skeletonCode}
            highlightLines={[10]}
          />
        </div>
      }
      right={
        <div style={{ width: 560 }}>
          {!showSolution && (
            <div
              style={{
                marginTop: 100,
                padding: "24px 28px",
                backgroundColor: "rgba(255,255,255,0.03)",
                borderRadius: 12,
                border: `1px solid rgba(255,255,255,0.08)`,
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontSize: 18,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyBody,
                }}
              >
                Solution appears in a moment...
              </div>
            </div>
          )}

          {showSolution && (
            <div style={{ marginTop: 10 }}>
              <FadeInText
                text="Solution Approach"
                delay={8 * fps}
                fontSize={22}
                fontWeight={700}
                color={THEME.colors.nvidiaGreen}
                style={{ marginBottom: 16 }}
              />

              <CodeBlock
                delay={8.5 * fps}
                title="solution.cu"
                fontSize={14}
                code={solutionCode}
                highlightLines={[3, 4, 5, 6, 7, 8]}
              />

              {/* Timing results */}
              <div
                style={{
                  marginTop: 16,
                  padding: "14px 20px",
                  backgroundColor: "rgba(118,185,0,0.08)",
                  borderRadius: 8,
                  border: `1px solid ${THEME.colors.nvidiaGreen}30`,
                  opacity: interpolate(
                    frame - 10.5 * fps,
                    [0, 0.5 * fps],
                    [0, 1],
                    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                  ),
                }}
              >
                <div
                  style={{
                    fontSize: 14,
                    fontWeight: 700,
                    color: THEME.colors.nvidiaGreen,
                    fontFamily: fontFamilyBody,
                    marginBottom: 8,
                  }}
                >
                  Expected Timing Results
                </div>
                {[
                  { label: "Serial", time: "4.2 ms", color: THEME.colors.accentRed },
                  { label: "4 streams", time: "1.6 ms", color: THEME.colors.nvidiaGreen },
                  { label: "Speedup", time: "~2.6x", color: THEME.colors.accentCyan },
                ].map((row) => (
                  <div
                    key={row.label}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      padding: "4px 0",
                    }}
                  >
                    <span
                      style={{
                        fontSize: 14,
                        color: THEME.colors.textSecondary,
                        fontFamily: fontFamilyCode,
                      }}
                    >
                      {row.label}
                    </span>
                    <span
                      style={{
                        fontSize: 14,
                        fontWeight: 700,
                        color: row.color,
                        fontFamily: fontFamilyCode,
                      }}
                    >
                      {row.time}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      }
    />
  );
};
