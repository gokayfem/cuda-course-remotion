import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M2S02_MemoryHierarchy: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const memories = [
    { name: "Registers", latency: "~1 cycle", size: "255 per thread", bw: "~8 TB/s", scope: "Thread", color: THEME.colors.accentRed, barPct: 100 },
    { name: "Shared Memory", latency: "~5 cycles", size: "up to 228 KB/SM", bw: "~4 TB/s", scope: "Block", color: THEME.colors.accentOrange, barPct: 80 },
    { name: "L1 Cache", latency: "~28 cycles", size: "128-256 KB/SM", bw: "~2 TB/s", scope: "SM (auto)", color: THEME.colors.accentYellow, barPct: 50 },
    { name: "L2 Cache", latency: "~200 cycles", size: "6-50 MB", bw: "~1 TB/s", scope: "GPU-wide", color: THEME.colors.accentBlue, barPct: 25 },
    { name: "Global (HBM)", latency: "~400+ cycles", size: "24-80 GB", bw: "1-3 TB/s", scope: "GPU-wide", color: THEME.colors.nvidiaGreen, barPct: 15 },
    { name: "PCIe (CPU↔GPU)", latency: "~10,000 cycles", size: "System RAM", bw: "~32 GB/s", scope: "System", color: THEME.colors.accentPurple, barPct: 1 },
  ];

  return (
    <SlideLayout variant="gradient" moduleNumber={2} slideNumber={2} totalSlides={18}>
      <SlideTitle
        title="GPU Memory Hierarchy — The Full Picture"
        subtitle="Understanding latency, size, and scope of each memory level"
      />

      <div style={{ flex: 1 }}>
        {/* Header row */}
        <div style={{
          display: "grid", gridTemplateColumns: "180px 110px 160px 100px 100px 1fr",
          gap: 8, padding: "10px 16px", backgroundColor: "rgba(255,255,255,0.05)",
          borderRadius: 8, marginBottom: 8,
          opacity: interpolate(frame - 0.5 * fps, [0, 0.3 * fps], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
        }}>
          {["Memory Type", "Latency", "Size", "Bandwidth", "Scope", "Relative Speed"].map(h => (
            <span key={h} style={{ fontSize: 14, fontWeight: 700, color: THEME.colors.textMuted, fontFamily: fontFamilyBody }}>{h}</span>
          ))}
        </div>

        {/* Rows */}
        {memories.map((mem, i) => {
          const rowDelay = 1 * fps + i * 0.3 * fps;
          const rowSpring = spring({ frame: frame - rowDelay, fps, config: { damping: 200 } });
          const rowOpacity = interpolate(rowSpring, [0, 1], [0, 1]);

          const barAnim = interpolate(frame - rowDelay - 0.2 * fps, [0, 0.5 * fps], [0, 1], {
            extrapolateLeft: "clamp", extrapolateRight: "clamp",
          });

          return (
            <div key={mem.name} style={{
              display: "grid", gridTemplateColumns: "180px 110px 160px 100px 100px 1fr",
              gap: 8, padding: "12px 16px", backgroundColor: `${mem.color}08`,
              borderLeft: `3px solid ${mem.color}`, borderRadius: 6, marginBottom: 5, opacity: rowOpacity,
            }}>
              <span style={{ fontSize: 16, fontWeight: 700, color: mem.color, fontFamily: fontFamilyBody }}>{mem.name}</span>
              <span style={{ fontSize: 15, color: THEME.colors.textPrimary, fontFamily: fontFamilyCode }}>{mem.latency}</span>
              <span style={{ fontSize: 15, color: THEME.colors.textSecondary, fontFamily: fontFamilyCode }}>{mem.size}</span>
              <span style={{ fontSize: 15, color: THEME.colors.textSecondary, fontFamily: fontFamilyCode }}>{mem.bw}</span>
              <span style={{ fontSize: 14, color: THEME.colors.textMuted, fontFamily: fontFamilyBody }}>{mem.scope}</span>
              <div style={{ display: "flex", alignItems: "center" }}>
                <div style={{ flex: 1, height: 10, backgroundColor: "rgba(255,255,255,0.05)", borderRadius: 5 }}>
                  <div style={{
                    width: `${mem.barPct * barAnim}%`, height: "100%",
                    backgroundColor: mem.color, borderRadius: 5, minWidth: barAnim > 0 ? 3 : 0,
                  }} />
                </div>
              </div>
            </div>
          );
        })}

        {/* Key insight */}
        <div style={{
          marginTop: 16, padding: "14px 24px", backgroundColor: "rgba(255,82,82,0.08)",
          borderRadius: 10, borderLeft: `4px solid ${THEME.colors.accentRed}`,
          opacity: interpolate(frame - 5 * fps, [0, 0.5 * fps], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
        }}>
          <span style={{ fontSize: 20, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody }}>
            Registers are <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>400x faster</span> than global memory.
            Moving data closer to compute is the <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>single biggest optimization</span> in CUDA.
          </span>
        </div>
      </div>
    </SlideLayout>
  );
};
