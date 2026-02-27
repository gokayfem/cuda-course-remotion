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
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M3S04_WarpDivergence: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const CELL_SIZE = 22;
  const CELL_GAP = 2;
  const THREADS = 32;

  // Phase: after code shows, show execution paths
  const pathPhaseStart = 3.5 * fps;
  const pathSpring = spring({
    frame: frame - pathPhaseStart,
    fps,
    config: { damping: 200 },
  });
  const pathOpacity = interpolate(pathSpring, [0, 1], [0, 1]);

  // Phase 2: mask animation
  const maskPhaseStart = 5 * fps;
  const maskProgress = interpolate(
    frame - maskPhaseStart,
    [0, 1.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const codeExample = `__global__ void kernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (data[idx] > 0.0f) {
      // PATH A: positive values
      data[idx] = sqrtf(data[idx]);
    } else {
      // PATH B: negative values
      data[idx] = 0.0f;
    }
  }
}`;

  const renderThreadMask = (
    label: string,
    activeThreads: boolean[],
    color: string,
    delay: number
  ) => {
    const maskSpring = spring({
      frame: frame - delay,
      fps,
      config: { damping: 200 },
    });
    const opacity = interpolate(maskSpring, [0, 1], [0, 1]);

    return (
      <div style={{ opacity, marginBottom: 10 }}>
        <div
          style={{
            fontSize: 14,
            color,
            fontFamily: fontFamilyBody,
            fontWeight: 700,
            marginBottom: 4,
          }}
        >
          {label}
        </div>
        <div style={{ display: "flex", gap: CELL_GAP, flexWrap: "wrap" }}>
          {activeThreads.map((active, i) => (
            <div
              key={i}
              style={{
                width: CELL_SIZE,
                height: CELL_SIZE,
                borderRadius: 3,
                backgroundColor: active
                  ? `${color}40`
                  : "rgba(255,255,255,0.03)",
                border: `1px solid ${active ? color : "rgba(255,255,255,0.06)"}`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 10,
                color: active ? color : "rgba(255,255,255,0.15)",
                fontFamily: fontFamilyCode,
                fontWeight: active ? 700 : 400,
              }}
            >
              {i}
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Simulate mixed positive/negative data pattern
  const pathAActive = Array.from(
    { length: THREADS },
    (_, i) => i % 3 !== 0
  );
  const pathBActive = pathAActive.map((a) => !a);

  const renderLeft = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <SlideTitle
        title="Warp Divergence"
        subtitle="When threads in a warp take different paths"
      />

      <CodeBlock
        code={codeExample}
        title="divergent_kernel.cu"
        fontSize={14}
        delay={0.5 * fps}
        highlightLines={[4, 7]}
      />

      {/* Thread execution visualization */}
      <div style={{ opacity: pathOpacity }}>
        <FadeInText
          text="Execution within one warp:"
          fontSize={16}
          fontWeight={600}
          delay={pathPhaseStart}
          style={{ marginBottom: 8 }}
        />

        {renderThreadMask(
          "Pass 1 — PATH A (active threads execute sqrt):",
          pathAActive,
          THEME.colors.nvidiaGreen,
          pathPhaseStart + 0.3 * fps
        )}

        {renderThreadMask(
          "Pass 2 — PATH B (active threads set to 0):",
          pathBActive,
          THEME.colors.accentOrange,
          pathPhaseStart + 1 * fps
        )}
      </div>
    </div>
  );

  const renderRight = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <FadeInText
        text="How Divergence Works"
        fontSize={22}
        fontWeight={700}
        delay={1 * fps}
        color={THEME.colors.accentBlue}
      />

      <BulletPoint
        index={0}
        delay={2 * fps}
        text="Both paths execute sequentially"
        subtext="The warp runs PATH A first (masking B threads), then PATH B (masking A threads)."
        highlight
      />
      <BulletPoint
        index={1}
        delay={2 * fps}
        text="Inactive threads are masked off"
        subtext="They still occupy the warp slot but produce no results for that pass."
      />
      <BulletPoint
        index={2}
        delay={2 * fps}
        text="Worst case: 2x slowdown"
        subtext="If/else with 50/50 split means both paths run at full cost."
      />
      <BulletPoint
        index={3}
        delay={2 * fps}
        text="Paths reconverge after branch"
        subtext="All threads resume executing the same instruction after the if/else."
      />

      {/* Performance impact box */}
      <div
        style={{
          marginTop: 12,
          padding: "10px 16px",
          backgroundColor: "rgba(255,82,82,0.10)",
          border: `1px solid ${THEME.colors.accentRed}50`,
          borderRadius: 8,
          opacity: interpolate(
            frame - 5.5 * fps,
            [0, 0.5 * fps],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          ),
        }}
      >
        <div
          style={{
            fontSize: 16,
            color: THEME.colors.accentRed,
            fontFamily: fontFamilyBody,
            fontWeight: 700,
          }}
        >
          Divergent warp executes BOTH paths
        </div>
        <div
          style={{
            fontSize: 14,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
            marginTop: 4,
          }}
        >
          Time = time(PATH A) + time(PATH B), not max of the two.
        </div>
      </div>
    </div>
  );

  return (
    <TwoColumnLayout
      moduleNumber={3}
      left={renderLeft()}
      right={renderRight()}
      leftWidth="52%"
    />
  );
};
