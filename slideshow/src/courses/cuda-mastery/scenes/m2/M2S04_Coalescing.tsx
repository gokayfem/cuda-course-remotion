import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M2S04_Coalescing: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const goodDiagramSpring = spring({
    frame: frame - 1.5 * fps,
    fps,
    config: { damping: 200 },
  });
  const goodOpacity = interpolate(goodDiagramSpring, [0, 1], [0, 1]);

  const badDiagramSpring = spring({
    frame: frame - 3.5 * fps,
    fps,
    config: { damping: 200 },
  });
  const badOpacity = interpolate(badDiagramSpring, [0, 1], [0, 1]);

  const warningOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const cellSize = 22;
  const cellGap = 2;

  const renderMemoryCells = (
    highlightIndices: number[],
    color: string,
    animDelay: number
  ) => {
    return (
      <div style={{ display: "flex", gap: cellGap, flexWrap: "wrap", width: 32 * (cellSize + cellGap) }}>
        {Array.from({ length: 32 }).map((_, i) => {
          const isHighlighted = highlightIndices.includes(i);
          const cellSpring = spring({
            frame: frame - animDelay - i * 0.02 * fps,
            fps,
            config: { damping: 200 },
          });
          const cellScale = interpolate(cellSpring, [0, 1], [0.5, 1]);
          return (
            <div
              key={i}
              style={{
                width: cellSize,
                height: cellSize,
                borderRadius: 3,
                backgroundColor: isHighlighted ? color : "rgba(255,255,255,0.06)",
                border: `1px solid ${isHighlighted ? color : "rgba(255,255,255,0.1)"}`,
                transform: `scale(${cellScale})`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 10,
                color: isHighlighted ? "#000" : THEME.colors.textMuted,
                fontFamily: fontFamilyCode,
                fontWeight: 700,
              }}
            >
              {i}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <SlideLayout variant="gradient" moduleNumber={2} slideNumber={4} totalSlides={18}>
      <SlideTitle
        title="Memory Coalescing — THE Key Optimization"
        subtitle="How threads in a warp access memory determines everything"
      />

      <div style={{ display: "flex", gap: 48, flex: 1 }}>
        {/* Left: Diagrams */}
        <div style={{ flex: 1.2 }}>
          {/* GOOD: Coalesced access */}
          <div style={{ marginBottom: 32, opacity: goodOpacity }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
              <div style={{
                padding: "4px 12px",
                backgroundColor: "rgba(118,185,0,0.15)",
                borderRadius: 6,
                fontSize: 15,
                fontWeight: 700,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
              }}>
                COALESCED
              </div>
              <span style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody }}>
                32 threads access addresses 0-31
              </span>
            </div>

            {/* Thread row */}
            <div style={{ display: "flex", gap: cellGap, marginBottom: 4, width: 32 * (cellSize + cellGap) }}>
              {Array.from({ length: 32 }).map((_, i) => {
                const tSpring = spring({
                  frame: frame - 1.8 * fps - i * 0.02 * fps,
                  fps,
                  config: { damping: 200 },
                });
                return (
                  <div
                    key={i}
                    style={{
                      width: cellSize,
                      height: 14,
                      borderRadius: 2,
                      backgroundColor: `${THEME.colors.accentCyan}60`,
                      opacity: interpolate(tSpring, [0, 1], [0, 1]),
                      fontSize: 7,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: THEME.colors.accentCyan,
                      fontFamily: fontFamilyCode,
                    }}
                  >
                    T{i}
                  </div>
                );
              })}
            </div>

            {/* Memory cells - all consecutive highlighted */}
            {renderMemoryCells(
              Array.from({ length: 32 }, (_, i) => i),
              THEME.colors.nvidiaGreen,
              2 * fps
            )}

            <div style={{
              marginTop: 6,
              fontSize: 15,
              color: THEME.colors.nvidiaGreen,
              fontFamily: fontFamilyBody,
              fontWeight: 700,
            }}>
              = 1 memory transaction (128 bytes)
            </div>
          </div>

          {/* BAD: Strided access */}
          <div style={{ opacity: badOpacity }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
              <div style={{
                padding: "4px 12px",
                backgroundColor: "rgba(255,82,82,0.15)",
                borderRadius: 6,
                fontSize: 15,
                fontWeight: 700,
                color: THEME.colors.accentRed,
                fontFamily: fontFamilyBody,
              }}>
                STRIDED
              </div>
              <span style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody }}>
                32 threads access every 2nd address
              </span>
            </div>

            {/* Strided memory cells - every other cell highlighted */}
            {renderMemoryCells(
              Array.from({ length: 32 }, (_, i) => i).filter((_, i) => i % 2 === 0),
              THEME.colors.accentRed,
              4 * fps
            )}
            {/* Second row for remaining addresses */}
            <div style={{ marginTop: 4 }}>
              {renderMemoryCells(
                Array.from({ length: 32 }, (_, i) => i).filter((_, i) => i % 2 === 1),
                THEME.colors.accentRed,
                4.3 * fps
              )}
            </div>

            <div style={{
              marginTop: 6,
              fontSize: 15,
              color: THEME.colors.accentRed,
              fontFamily: fontFamilyBody,
              fontWeight: 700,
            }}>
              = up to 32 separate transactions (wasted bandwidth!)
            </div>
          </div>
        </div>

        {/* Right: Explanation */}
        <div style={{ flex: 0.8 }}>
          <BulletPoint
            index={0}
            delay={5 * fps}
            text="128-byte cache lines"
            subtext="GPU fetches memory in 128-byte chunks. If all 32 threads land in one chunk: 1 transaction."
          />
          <BulletPoint
            index={1}
            delay={5 * fps}
            text="Warp = 32 threads in lockstep"
            subtext="All 32 threads issue memory requests simultaneously."
          />
          <BulletPoint
            index={2}
            delay={5 * fps}
            text="Consecutive = coalesced"
            subtext="Thread i accesses address i means all 32 fit in 1-4 cache lines."
            highlight
          />
          <BulletPoint
            index={3}
            delay={5 * fps}
            text="Strided = disaster"
            subtext="Thread i accesses address i*stride wastes most of each cache line fetch."
          />
        </div>
      </div>

      {/* Warning box */}
      <div
        style={{
          marginTop: 8,
          padding: "14px 24px",
          backgroundColor: "rgba(255,82,82,0.10)",
          borderRadius: 10,
          border: `2px solid ${THEME.colors.accentRed}60`,
          opacity: warningOpacity,
          textAlign: "center",
        }}
      >
        <span style={{
          fontSize: 22,
          color: THEME.colors.textPrimary,
          fontFamily: fontFamilyBody,
          fontWeight: 700,
        }}>
          Coalescing is the{" "}
          <span style={{ color: THEME.colors.accentRed }}>single most impactful optimization</span>
          {" "}you can make.
        </span>
      </div>
    </SlideLayout>
  );
};
