import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

const STAGES = [
  { label: "Input", color: THEME.colors.accentBlue },
  { label: "Predicates", color: THEME.colors.accentOrange },
  { label: "Scan", color: THEME.colors.accentCyan },
  { label: "Scatter", color: THEME.colors.accentPurple },
  { label: "Output", color: THEME.colors.nvidiaGreen },
];

export const M4S14_CompactionCode: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const diagramDelay = 4 * fps;
  const diagramOpacity = interpolate(
    frame - diagramDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const insightDelay = 8 * fps;
  const insightOpacity = interpolate(
    frame - insightDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={4}
      slideNumber={14}
      totalSlides={18}
      leftWidth="55%"
      left={
        <>
          <SlideTitle
            title="Stream Compaction Code"
            subtitle="Three kernels: predicate, scan, scatter"
          />

          <CodeBlock
            delay={0.5 * fps}
            title="compaction_kernels.cu"
            fontSize={13}
            code={`// Kernel 1: Compute predicates
__global__ void compute_predicates(
    const float *data, int *pred, int N,
    float threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        pred[i] = (data[i] > threshold) ? 1 : 0;
}

// Kernel 2: Exclusive scan (reuse scan kernel)
// scan(pred, scan_result, N);

// Kernel 3: Scatter selected elements
__global__ void scatter(
    const float *data, const int *pred,
    const int *scan_idx, float *output, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && pred[i] == 1)
        output[scan_idx[i]] = data[i];
}

// Host launch:
// compute_predicates<<<grid, block>>>(d, pred, N, thresh);
// exclusive_scan(pred, scan_idx, N);
// scatter<<<grid, block>>>(d, pred, scan_idx, out, N);
// total_selected = scan_idx[N-1] + pred[N-1];`}
            highlightLines={[8, 20, 21, 27]}
          />
        </>
      }
      right={
        <>
          <div style={{ marginTop: 100 }}>
            {/* Data flow diagram */}
            <div
              style={{
                fontSize: 16,
                fontWeight: 700,
                color: THEME.colors.textPrimary,
                fontFamily: fontFamilyBody,
                marginBottom: 16,
              }}
            >
              Data Flow Pipeline
            </div>

            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 0,
                opacity: diagramOpacity,
              }}
            >
              {STAGES.map((stage, i) => {
                const stageDelay = diagramDelay + i * 0.35 * fps;
                const stageSpring = spring({
                  frame: frame - stageDelay,
                  fps,
                  config: { damping: 200 },
                });
                const stageOpacity = interpolate(stageSpring, [0, 1], [0, 1]);
                const stageScale = interpolate(stageSpring, [0, 1], [0.9, 1]);

                return (
                  <React.Fragment key={i}>
                    <div
                      style={{
                        width: 280,
                        padding: "12px 20px",
                        backgroundColor: `${stage.color}12`,
                        border: `2px solid ${stage.color}50`,
                        borderRadius: 10,
                        textAlign: "center",
                        opacity: stageOpacity,
                        transform: `scale(${stageScale})`,
                      }}
                    >
                      <div
                        style={{
                          fontSize: 16,
                          fontWeight: 700,
                          color: stage.color,
                          fontFamily: fontFamilyBody,
                        }}
                      >
                        {stage.label}
                      </div>
                    </div>

                    {i < STAGES.length - 1 && (
                      <div
                        style={{
                          fontSize: 20,
                          color: THEME.colors.textMuted,
                          opacity: stageOpacity,
                          margin: "4px 0",
                        }}
                      >
                        {"\u2193"}
                      </div>
                    )}
                  </React.Fragment>
                );
              })}
            </div>

            {/* Kernel labels next to flow */}
            <div
              style={{
                marginTop: 20,
                display: "flex",
                flexDirection: "column",
                gap: 6,
                opacity: diagramOpacity,
              }}
            >
              {[
                { kernel: "compute_predicates", connects: "Input \u2192 Predicates" },
                { kernel: "exclusive_scan", connects: "Predicates \u2192 Scan" },
                { kernel: "scatter", connects: "Scan \u2192 Output" },
              ].map((k, i) => {
                const kDelay = diagramDelay + 2 * fps + i * 0.3 * fps;
                const kOpacity = interpolate(
                  frame - kDelay,
                  [0, 0.3 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                );

                return (
                  <div
                    key={i}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                      opacity: kOpacity,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 13,
                        fontFamily: fontFamilyCode,
                        color: THEME.colors.syntaxFunction,
                        backgroundColor: "#0d1117",
                        padding: "3px 8px",
                        borderRadius: 4,
                      }}
                    >
                      {k.kernel}
                    </span>
                    <span
                      style={{
                        fontSize: 13,
                        color: THEME.colors.textMuted,
                        fontFamily: fontFamilyBody,
                      }}
                    >
                      {k.connects}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Insight box */}
          <div
            style={{
              marginTop: 24,
              padding: "14px 18px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.nvidiaGreen}30`,
              opacity: insightOpacity,
            }}
          >
            <div
              style={{
                fontSize: 14,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                lineHeight: 1.5,
              }}
            >
              <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>
                Key insight:
              </span>{" "}
              The scan result acts as a{" "}
              <span style={{ color: THEME.colors.accentCyan, fontWeight: 700 }}>
                scatter map
              </span>
              . Each selected element knows exactly where to write in the output without
              any coordination between threads.
            </div>
          </div>
        </>
      }
    />
  );
};
