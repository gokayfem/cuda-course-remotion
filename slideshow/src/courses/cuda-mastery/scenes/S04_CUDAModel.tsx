import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, FadeInText, BulletPoint } from "../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../styles/fonts";

export const S04_CUDAModel: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="gradient" slideNumber={4} totalSlides={18}>
      <SlideTitle
        title="The CUDA Programming Model"
        subtitle="Grid → Blocks → Threads: the execution hierarchy"
      />

      <div style={{ display: "flex", gap: 48, flex: 1 }}>
        {/* Left: Hierarchy diagram */}
        <div style={{ flex: 1, position: "relative" }}>
          {/* Grid box */}
          {(() => {
            const gridDelay = 0.8 * fps;
            const gridSpring = spring({ frame: frame - gridDelay, fps, config: { damping: 200 } });
            const gridOpacity = interpolate(gridSpring, [0, 1], [0, 1]);

            return (
              <div
                style={{
                  position: "relative",
                  border: `2px solid ${THEME.colors.accentPurple}`,
                  borderRadius: 16,
                  padding: 24,
                  backgroundColor: `${THEME.colors.accentPurple}08`,
                  opacity: gridOpacity,
                  height: "100%",
                }}
              >
                <div style={{ fontSize: 20, fontWeight: 700, color: THEME.colors.accentPurple, fontFamily: fontFamilyBody, marginBottom: 16 }}>
                  Grid
                </div>

                {/* Block row */}
                <div style={{ display: "flex", gap: 16 }}>
                  {[0, 1, 2].map((blockIdx) => {
                    const blockDelay = 1.5 * fps + blockIdx * 0.3 * fps;
                    const blockSpring = spring({ frame: frame - blockDelay, fps, config: { damping: 200 } });
                    const blockOpacity = interpolate(blockSpring, [0, 1], [0, 1]);

                    return (
                      <div
                        key={blockIdx}
                        style={{
                          flex: 1,
                          border: `2px solid ${THEME.colors.accentBlue}`,
                          borderRadius: 12,
                          padding: 16,
                          backgroundColor: `${THEME.colors.accentBlue}08`,
                          opacity: blockOpacity,
                        }}
                      >
                        <div style={{ fontSize: 16, fontWeight: 700, color: THEME.colors.accentBlue, fontFamily: fontFamilyBody, marginBottom: 12 }}>
                          Block {blockIdx}
                        </div>

                        {/* Threads within block */}
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 6 }}>
                          {Array.from({ length: 8 }).map((_, tIdx) => {
                            const threadDelay = 2.5 * fps + blockIdx * 0.2 * fps + tIdx * 2;
                            const tSpring = spring({ frame: frame - threadDelay, fps, config: { damping: 200 } });
                            const tOpacity = interpolate(tSpring, [0, 1], [0, 1]);

                            return (
                              <div
                                key={tIdx}
                                style={{
                                  height: 36,
                                  backgroundColor: `${THEME.colors.nvidiaGreen}25`,
                                  border: `1.5px solid ${THEME.colors.nvidiaGreen}`,
                                  borderRadius: 4,
                                  display: "flex",
                                  alignItems: "center",
                                  justifyContent: "center",
                                  fontSize: 11,
                                  color: THEME.colors.nvidiaGreen,
                                  fontFamily: fontFamilyCode,
                                  fontWeight: 700,
                                  opacity: tOpacity,
                                }}
                              >
                                T{tIdx}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Labels */}
                <div style={{ marginTop: 20, display: "flex", gap: 24, justifyContent: "center" }}>
                  {[
                    { color: THEME.colors.accentPurple, label: "Grid" },
                    { color: THEME.colors.accentBlue, label: "Block" },
                    { color: THEME.colors.nvidiaGreen, label: "Thread" },
                  ].map(({ color, label }, i) => {
                    const legendDelay = 4 * fps + i * 0.2 * fps;
                    const legendOpacity = interpolate(
                      frame - legendDelay,
                      [0, 0.3 * fps],
                      [0, 1],
                      { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                    );
                    return (
                      <div key={label} style={{ display: "flex", alignItems: "center", gap: 8, opacity: legendOpacity }}>
                        <div style={{ width: 16, height: 16, backgroundColor: `${color}40`, border: `2px solid ${color}`, borderRadius: 4 }} />
                        <span style={{ fontSize: 16, color, fontFamily: fontFamilyBody, fontWeight: 600 }}>{label}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })()}
        </div>

        {/* Right: Explanation */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text="Key Rules"
            delay={1 * fps}
            fontSize={26}
            fontWeight={700}
            color={THEME.colors.nvidiaGreen}
            style={{ marginBottom: 16 }}
          />

          <BulletPoint
            index={0}
            delay={1.5 * fps}
            text="Grid = entire problem space"
            subtext="You define how many blocks the grid has"
          />
          <BulletPoint
            index={1}
            delay={1.5 * fps}
            text="Block = group of cooperating threads"
            subtext="Can share memory & synchronize within a block"
            highlight
          />
          <BulletPoint
            index={2}
            delay={1.5 * fps}
            text="Thread = single unit of execution"
            subtext="Each thread has a unique ID (threadIdx, blockIdx)"
          />
          <BulletPoint
            index={3}
            delay={1.5 * fps}
            text="Blocks are INDEPENDENT"
            subtext="Can run in any order → GPU scales across hardware"
            highlight
          />

          {/* Important note */}
          <div
            style={{
              marginTop: 24,
              padding: "16px 20px",
              backgroundColor: "rgba(255,82,82,0.08)",
              borderRadius: 8,
              borderLeft: `4px solid ${THEME.colors.accentRed}`,
              opacity: interpolate(
                frame - 5 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <span style={{ fontSize: 18, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody }}>
              Max <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>1024 threads per block</span>.
              Block dimensions must be multiples of <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>32 (warp size)</span>.
            </span>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
