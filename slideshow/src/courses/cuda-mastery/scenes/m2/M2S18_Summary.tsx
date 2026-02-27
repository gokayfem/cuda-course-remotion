import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideBackground } from "../../../../components/SlideBackground";
import { fontFamilyHeading, fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M2S18_Summary: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const takeaways = [
    { icon: "1", text: "Memory hierarchy: registers > shared > L1/L2 > global > PCIe", color: THEME.colors.accentRed },
    { icon: "2", text: "Coalescing: adjacent threads -> adjacent addresses = fast", color: THEME.colors.nvidiaGreen },
    { icon: "3", text: "Shared memory: 100x faster than global, programmer-managed", color: THEME.colors.accentOrange },
    { icon: "4", text: "Bank conflicts: 32 banks, fix with +1 padding", color: THEME.colors.accentYellow },
    { icon: "5", text: "SoA > AoS for GPU memory layout", color: THEME.colors.accentBlue },
    { icon: "6", text: "Registers: fastest but limited, affects occupancy", color: THEME.colors.accentCyan },
    { icon: "7", text: "Constant memory: 64KB, broadcast to entire warp", color: THEME.colors.accentPurple },
    { icon: "8", text: "Minimize CPU<->GPU transfers (PCIe bottleneck)", color: THEME.colors.accentPink },
  ];

  const nextOpacity = interpolate(
    frame - 8 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <AbsoluteFill>
      <SlideBackground variant="accent" />

      <div style={{ position: "absolute", inset: 80, display: "flex", flexDirection: "column" }}>
        {/* Title */}
        <div style={{ marginBottom: 32 }}>
          <h1
            style={{
              fontSize: 56,
              fontWeight: 900,
              color: THEME.colors.textPrimary,
              fontFamily: fontFamilyHeading,
              margin: 0,
              opacity: interpolate(
                spring({ frame, fps, config: { damping: 200 } }),
                [0, 1],
                [0, 1]
              ),
            }}
          >
            Module 2: <span style={{ color: THEME.colors.accentOrange }}>Complete</span>
          </h1>
          <div
            style={{
              height: 4,
              width: interpolate(frame, [0, 1 * fps], [0, 300], { extrapolateRight: "clamp" }),
              background: `linear-gradient(90deg, ${THEME.colors.accentOrange}, ${THEME.colors.accentRed})`,
              borderRadius: 2,
              marginTop: 12,
            }}
          />
        </div>

        {/* Key Takeaways - 2x4 grid */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, flex: 1 }}>
          {takeaways.map((item, i) => {
            const itemDelay = 0.8 * fps + i * 0.25 * fps;
            const itemSpring = spring({
              frame: frame - itemDelay,
              fps,
              config: { damping: 200 },
            });
            const itemOpacity = interpolate(itemSpring, [0, 1], [0, 1]);
            const itemX = interpolate(itemSpring, [0, 1], [-20, 0]);

            return (
              <div
                key={i}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 16,
                  padding: "14px 20px",
                  backgroundColor: `${item.color}08`,
                  borderLeft: `4px solid ${item.color}`,
                  borderRadius: 8,
                  opacity: itemOpacity,
                  transform: `translateX(${itemX}px)`,
                }}
              >
                <span
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: 16,
                    backgroundColor: `${item.color}20`,
                    color: item.color,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 16,
                    fontWeight: 700,
                    fontFamily: fontFamilyBody,
                    flexShrink: 0,
                  }}
                >
                  {item.icon}
                </span>
                <span
                  style={{
                    fontSize: 18,
                    color: THEME.colors.textPrimary,
                    fontFamily: fontFamilyBody,
                    fontWeight: 400,
                  }}
                >
                  {item.text}
                </span>
              </div>
            );
          })}
        </div>

        {/* Next Module Teaser */}
        <div
          style={{
            marginTop: 24,
            padding: "24px 32px",
            background: `linear-gradient(135deg, rgba(255,171,64,0.08), rgba(255,82,82,0.08))`,
            borderRadius: 16,
            border: `1px solid rgba(255,171,64,0.2)`,
            opacity: nextOpacity,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div>
            <div style={{ fontSize: 16, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginBottom: 4 }}>
              Coming Next
            </div>
            <div style={{ fontSize: 28, fontWeight: 700, color: THEME.colors.accentOrange, fontFamily: fontFamilyHeading }}>
              Module 3: Thread Synchronization & Execution Model
            </div>
            <div style={{ fontSize: 18, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, marginTop: 4 }}>
              Warps, divergence, atomics — understanding how the GPU really executes your code
            </div>
          </div>
          <div
            style={{
              fontSize: 48,
              color: THEME.colors.accentOrange,
              opacity: 0.5,
            }}
          >
            {"\u2192"}
          </div>
        </div>
      </div>

      {/* Bottom bar */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          height: 6,
          background: `linear-gradient(90deg, ${THEME.colors.accentOrange}, ${THEME.colors.accentRed}, ${THEME.colors.accentPurple})`,
        }}
      />
    </AbsoluteFill>
  );
};
