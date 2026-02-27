import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint, FadeInText } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M2S07_SharedMemoryIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const smBoxSpring = spring({
    frame: frame - 2 * fps,
    fps,
    config: { damping: 200 },
  });
  const smBoxOpacity = interpolate(smBoxSpring, [0, 1], [0, 1]);
  const smBoxScale = interpolate(smBoxSpring, [0, 1], [0.9, 1]);

  const threadRows = [0, 1, 2, 3];

  return (
    <SlideLayout variant="gradient" moduleNumber={2} slideNumber={7} totalSlides={18}>
      <SlideTitle
        title="Shared Memory — The On-Chip Scratchpad"
        subtitle="Programmer-managed cache sitting right next to the compute cores"
      />

      <div style={{ display: "flex", gap: 48, flex: 1 }}>
        {/* Left: Properties + Use cases */}
        <div style={{ flex: 1 }}>
          <BulletPoint
            index={0}
            delay={1 * fps}
            text="Blazing fast: ~5 cycle latency"
            subtext="Compared to 400+ cycles for global memory (80x faster)"
            icon="1"
            highlight
          />
          <BulletPoint
            index={1}
            delay={1 * fps}
            text="Limited size: 48-228 KB per SM"
            subtext="Shared by all threads in a block, configurable vs L1 cache"
            icon="2"
          />
          <BulletPoint
            index={2}
            delay={1 * fps}
            text="Programmer-managed"
            subtext="You explicitly load data in, unlike automatic caches"
            icon="3"
          />
          <BulletPoint
            index={3}
            delay={1 * fps}
            text="Block-scoped lifetime"
            subtext="Data exists only while the thread block runs"
            icon="4"
          />

          <FadeInText
            text="Common Use Cases:"
            delay={3.5 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentOrange}
            style={{ marginTop: 16, marginBottom: 8 }}
          />

          <BulletPoint
            index={0}
            delay={4 * fps}
            text="Cache frequently accessed data"
          />
          <BulletPoint
            index={1}
            delay={4 * fps}
            text="Inter-thread communication within a block"
          />
          <BulletPoint
            index={2}
            delay={4 * fps}
            text="Transform uncoalesced global reads into coalesced"
            highlight
          />
        </div>

        {/* Right: Diagram + Code */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 16 }}>
          {/* SM Diagram */}
          <div style={{
            padding: 20,
            backgroundColor: "rgba(255,171,64,0.06)",
            border: `2px solid ${THEME.colors.accentOrange}40`,
            borderRadius: 12,
            opacity: smBoxOpacity,
            transform: `scale(${smBoxScale})`,
          }}>
            <div style={{
              fontSize: 18,
              fontWeight: 700,
              color: THEME.colors.accentOrange,
              fontFamily: fontFamilyBody,
              marginBottom: 12,
              textAlign: "center",
            }}>
              Streaming Multiprocessor (SM)
            </div>

            {/* Shared memory block */}
            <div style={{
              padding: "10px 16px",
              backgroundColor: "rgba(255,171,64,0.15)",
              border: `2px solid ${THEME.colors.accentOrange}`,
              borderRadius: 8,
              marginBottom: 12,
              textAlign: "center",
            }}>
              <span style={{
                fontSize: 16,
                fontWeight: 700,
                color: THEME.colors.accentOrange,
                fontFamily: fontFamilyCode,
              }}>
                __shared__ memory (48-228 KB)
              </span>
            </div>

            {/* Thread groups */}
            <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
              {threadRows.map((row) => {
                const rowDelay = 2.5 * fps + row * 0.2 * fps;
                const rowSpring = spring({
                  frame: frame - rowDelay,
                  fps,
                  config: { damping: 200 },
                });
                return (
                  <div
                    key={row}
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: 4,
                      opacity: interpolate(rowSpring, [0, 1], [0, 1]),
                    }}
                  >
                    {Array.from({ length: 4 }).map((_, t) => (
                      <div
                        key={t}
                        style={{
                          width: 36,
                          height: 20,
                          backgroundColor: `${THEME.colors.accentCyan}20`,
                          border: `1px solid ${THEME.colors.accentCyan}60`,
                          borderRadius: 3,
                          fontSize: 10,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          color: THEME.colors.accentCyan,
                          fontFamily: fontFamilyCode,
                        }}
                      >
                        T{row * 4 + t}
                      </div>
                    ))}
                    <span style={{ fontSize: 10, color: THEME.colors.textMuted, fontFamily: fontFamilyCode }}>
                      Warp {row}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Arrows from shared to threads */}
            <div style={{
              textAlign: "center",
              fontSize: 14,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
              marginTop: 8,
              opacity: interpolate(frame - 3.5 * fps, [0, 0.3 * fps], [0, 1], {
                extrapolateLeft: "clamp", extrapolateRight: "clamp",
              }),
            }}>
              All threads in the block can read/write shared memory
            </div>
          </div>

          {/* Code snippet */}
          <CodeBlock
            delay={4.5 * fps}
            title="shared_basics.cu"
            fontSize={14}
            showLineNumbers={false}
            code={`__global__ void kernel() {
    // Declare shared memory
    __shared__ float cache[256];

    // Load from global -> shared
    cache[threadIdx.x] = global_data[idx];

    // MUST sync before reading!
    __syncthreads();

    // Now all threads can read any
    // element from cache (fast!)
    float val = cache[threadIdx.x + 1];
}`}
            highlightLines={[3, 9]}
          />
        </div>
      </div>
    </SlideLayout>
  );
};
