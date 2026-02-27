import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint, FadeInText } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode, fontFamilyHeading } from "../../../../styles/fonts";

const codeLines = [
  { text: "cudaEvent_t start, stop;", color: THEME.colors.textCode },
  { text: "cudaEventCreate(&start);", color: THEME.colors.syntaxFunction },
  { text: "cudaEventCreate(&stop);", color: THEME.colors.syntaxFunction },
  { text: "", color: "transparent" },
  { text: "cudaEventRecord(start, stream);", color: THEME.colors.accentCyan },
  { text: "myKernel<<<grid, block, 0, stream>>>(..);", color: THEME.colors.syntaxFunction },
  { text: "cudaEventRecord(stop, stream);", color: THEME.colors.accentCyan },
  { text: "", color: "transparent" },
  { text: "cudaEventSynchronize(stop);", color: THEME.colors.syntaxKeyword },
  { text: "", color: "transparent" },
  { text: "float ms;", color: THEME.colors.textCode },
  { text: "cudaEventElapsedTime(&ms, start, stop);", color: THEME.colors.nvidiaGreen },
  { text: '// ms = kernel execution time', color: THEME.colors.syntaxComment },
];

interface CompRow {
  method: string;
  precision: string;
  asyncSafe: string;
  overhead: string;
  color: string;
}

const comparisonRows: CompRow[] = [
  {
    method: "clock() (CPU)",
    precision: "~ms",
    asyncSafe: "No",
    overhead: "High",
    color: THEME.colors.accentRed,
  },
  {
    method: "CUDA Events",
    precision: "~0.5\u00B5s",
    asyncSafe: "Yes",
    overhead: "Low",
    color: THEME.colors.nvidiaGreen,
  },
  {
    method: "nsys profiler",
    precision: "~ns",
    asyncSafe: "Yes",
    overhead: "Varies",
    color: THEME.colors.accentBlue,
  },
];

export const M6S09_EventTiming: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Code block
  const codeOpacity = interpolate(
    frame - 0.5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Comparison table
  const tableOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={6}
      leftWidth="48%"
      left={
        <div style={{ width: 520 }}>
          <SlideTitle
            title="Precise GPU Timing with Events"
            subtitle="The standard way to benchmark CUDA kernels"
          />

          {/* Code block */}
          <div
            style={{
              backgroundColor: THEME.colors.bgCode,
              borderRadius: 10,
              padding: "16px 20px",
              border: `1px solid rgba(255,255,255,0.08)`,
              opacity: codeOpacity,
              width: 500,
            }}
          >
            {codeLines.map((line, i) => {
              const lineDelay = (1 + i * 0.25) * fps;
              const lineOpacity = interpolate(
                frame - lineDelay,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              return (
                <div
                  key={i}
                  style={{
                    fontSize: 14,
                    fontFamily: fontFamilyCode,
                    color: line.color,
                    opacity: lineOpacity,
                    lineHeight: 1.7,
                    minHeight: line.text === "" ? 10 : "auto",
                    whiteSpace: "nowrap",
                  }}
                >
                  {line.text}
                </div>
              );
            })}
          </div>

          {/* Annotations on the code */}
          <div
            style={{
              marginTop: 16,
              display: "flex",
              gap: 12,
              width: 500,
            }}
          >
            {[
              {
                label: "Record start",
                color: THEME.colors.accentCyan,
                delay: 3,
              },
              {
                label: "Launch kernel",
                color: THEME.colors.syntaxFunction,
                delay: 3.5,
              },
              {
                label: "Record stop",
                color: THEME.colors.accentCyan,
                delay: 4,
              },
              {
                label: "Get elapsed",
                color: THEME.colors.nvidiaGreen,
                delay: 4.5,
              },
            ].map((ann) => {
              const annOpacity = interpolate(
                frame - ann.delay * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );
              return (
                <div
                  key={ann.label}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    opacity: annOpacity,
                  }}
                >
                  <div
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      backgroundColor: ann.color,
                    }}
                  />
                  <span
                    style={{
                      fontSize: 12,
                      color: ann.color,
                      fontFamily: fontFamilyBody,
                      fontWeight: 600,
                    }}
                  >
                    {ann.label}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 40, width: 480 }}>
          <BulletPoint
            index={0}
            delay={2 * fps}
            text="GPU clock precision (not CPU timer overhead)"
            icon="1"
            highlight
          />
          <BulletPoint
            index={1}
            delay={2 * fps}
            text="Works across async operations"
            icon="2"
          />
          <BulletPoint
            index={2}
            delay={2 * fps}
            text="Standard way to benchmark CUDA kernels"
            icon="3"
            highlight
          />
          <BulletPoint
            index={3}
            delay={2 * fps}
            text="Can time individual stages of a pipeline"
            icon="4"
          />

          {/* Comparison table */}
          <div
            style={{
              marginTop: 28,
              opacity: tableOpacity,
              width: 460,
            }}
          >
            <FadeInText
              text="Timing Methods Compared"
              delay={7 * fps}
              fontSize={18}
              fontWeight={700}
              color={THEME.colors.accentCyan}
              style={{ marginBottom: 14, width: 460 }}
            />

            {/* Table header */}
            <div
              style={{
                display: "flex",
                gap: 0,
                borderBottom: `2px solid rgba(255,255,255,0.1)`,
                paddingBottom: 8,
                marginBottom: 8,
                width: 460,
              }}
            >
              {["Method", "Precision", "Async?", "Overhead"].map(
                (header) => (
                  <div
                    key={header}
                    style={{
                      width: 115,
                      fontSize: 13,
                      fontWeight: 700,
                      color: THEME.colors.textSecondary,
                      fontFamily: fontFamilyHeading,
                    }}
                  >
                    {header}
                  </div>
                )
              )}
            </div>

            {/* Table rows */}
            {comparisonRows.map((row, i) => {
              const rowSpring = spring({
                frame: frame - (7.5 + i * 0.5) * fps,
                fps,
                config: { damping: 200 },
              });
              const rowOpacity = interpolate(rowSpring, [0, 1], [0, 1]);
              const rowX = interpolate(rowSpring, [0, 1], [15, 0]);

              const isRecommended = row.method === "CUDA Events";

              return (
                <div
                  key={row.method}
                  style={{
                    display: "flex",
                    gap: 0,
                    padding: "8px 0",
                    opacity: rowOpacity,
                    transform: `translateX(${rowX}px)`,
                    backgroundColor: isRecommended
                      ? "rgba(118,185,0,0.06)"
                      : "transparent",
                    borderRadius: 4,
                    borderBottom: `1px solid rgba(255,255,255,0.04)`,
                    width: 460,
                  }}
                >
                  <div
                    style={{
                      width: 115,
                      fontSize: 13,
                      fontWeight: 700,
                      color: row.color,
                      fontFamily: fontFamilyCode,
                      paddingLeft: 4,
                    }}
                  >
                    {row.method}
                  </div>
                  <div
                    style={{
                      width: 115,
                      fontSize: 13,
                      color: THEME.colors.textPrimary,
                      fontFamily: fontFamilyCode,
                    }}
                  >
                    {row.precision}
                  </div>
                  <div
                    style={{
                      width: 115,
                      fontSize: 13,
                      color:
                        row.asyncSafe === "Yes"
                          ? THEME.colors.nvidiaGreen
                          : THEME.colors.accentRed,
                      fontFamily: fontFamilyBody,
                      fontWeight: 600,
                    }}
                  >
                    {row.asyncSafe}
                  </div>
                  <div
                    style={{
                      width: 115,
                      fontSize: 13,
                      color:
                        row.overhead === "Low"
                          ? THEME.colors.nvidiaGreen
                          : row.overhead === "High"
                            ? THEME.colors.accentRed
                            : THEME.colors.accentYellow,
                      fontFamily: fontFamilyBody,
                      fontWeight: 600,
                    }}
                  >
                    {row.overhead}
                  </div>
                </div>
              );
            })}

            {/* Recommendation */}
            <div
              style={{
                marginTop: 14,
                display: "flex",
                alignItems: "center",
                gap: 8,
                opacity: interpolate(
                  frame - 10 * fps,
                  [0, 0.5 * fps],
                  [0, 1],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                ),
                width: 460,
              }}
            >
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  backgroundColor: THEME.colors.nvidiaGreen,
                }}
              />
              <span
                style={{
                  fontSize: 14,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                  fontWeight: 700,
                }}
              >
                CUDA Events = best balance of precision, ease, and low overhead
              </span>
            </div>
          </div>
        </div>
      }
    />
  );
};
