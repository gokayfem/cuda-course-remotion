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

const WARP_COLORS = [
  THEME.colors.nvidiaGreen,
  THEME.colors.accentBlue,
  THEME.colors.accentOrange,
  THEME.colors.accentPurple,
  THEME.colors.accentCyan,
  THEME.colors.accentPink,
];

const WARP_STATES = ["EXEC", "MEM", "EXEC", "MEM", "EXEC", "READY"] as const;

export const M3S03_WarpScheduling: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const diagramSpring = spring({
    frame: frame - 1 * fps,
    fps,
    config: { damping: 200 },
  });
  const diagramOpacity = interpolate(diagramSpring, [0, 1], [0, 1]);

  const timelineSpring = spring({
    frame: frame - 2.5 * fps,
    fps,
    config: { damping: 200 },
  });
  const timelineOpacity = interpolate(timelineSpring, [0, 1], [0, 1]);

  const insightOpacity = interpolate(
    frame - 6 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Timeline scheduling pattern (which warp runs at each cycle)
  const TIMELINE_SLOTS = 12;
  const schedulePattern = [0, 0, 1, 2, 2, 3, 4, 1, 5, 0, 3, 2];

  // Animated timeline cursor
  const cursorPos =
    frame > 3 * fps
      ? interpolate(
          frame - 3 * fps,
          [0, 4 * fps],
          [0, TIMELINE_SLOTS - 1],
          { extrapolateRight: "clamp" }
        )
      : -1;
  const activeCursorSlot = Math.floor(cursorPos);

  const renderSMDiagram = () => (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 10,
        opacity: diagramOpacity,
      }}
    >
      <div
        style={{
          fontSize: 16,
          color: THEME.colors.textSecondary,
          fontFamily: fontFamilyBody,
          fontWeight: 600,
          marginBottom: 4,
        }}
      >
        Streaming Multiprocessor (SM)
      </div>

      <div
        style={{
          padding: 16,
          backgroundColor: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 10,
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}
      >
        {/* Warp pool */}
        <div
          style={{
            fontSize: 14,
            color: THEME.colors.textMuted,
            fontFamily: fontFamilyBody,
            marginBottom: 2,
          }}
        >
          Resident Warps:
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {Array.from({ length: 6 }).map((_, i) => {
            const warpDelay = 1.2 * fps + i * 0.15 * fps;
            const wSpring = spring({
              frame: frame - warpDelay,
              fps,
              config: { damping: 200 },
            });
            const wScale = interpolate(wSpring, [0, 1], [0.6, 1]);
            const wOpacity = interpolate(wSpring, [0, 1], [0, 1]);

            const isActive = activeCursorSlot >= 0 && schedulePattern[activeCursorSlot] === i;
            const state = isActive ? "EXEC" : WARP_STATES[i];

            return (
              <div
                key={i}
                style={{
                  width: 128,
                  padding: "8px 10px",
                  backgroundColor: isActive
                    ? `${WARP_COLORS[i]}25`
                    : "rgba(255,255,255,0.04)",
                  border: `1.5px solid ${isActive ? WARP_COLORS[i] : `${WARP_COLORS[i]}40`}`,
                  borderRadius: 6,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 4,
                  transform: `scale(${wScale})`,
                  opacity: wOpacity,
                }}
              >
                <span
                  style={{
                    fontSize: 13,
                    color: WARP_COLORS[i],
                    fontFamily: fontFamilyCode,
                    fontWeight: 700,
                  }}
                >
                  Warp {i}
                </span>
                <span
                  style={{
                    fontSize: 11,
                    color: isActive
                      ? THEME.colors.nvidiaGreen
                      : THEME.colors.textMuted,
                    fontFamily: fontFamilyCode,
                    fontWeight: 600,
                  }}
                >
                  {isActive ? "EXECUTING" : state}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );

  const renderTimeline = () => (
    <div
      style={{
        opacity: timelineOpacity,
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      <div
        style={{
          fontSize: 16,
          color: THEME.colors.textSecondary,
          fontFamily: fontFamilyBody,
          fontWeight: 600,
        }}
      >
        Execution Timeline (warp scheduler picks a ready warp each cycle):
      </div>

      <div
        style={{
          display: "flex",
          gap: 3,
          alignItems: "flex-end",
        }}
      >
        {schedulePattern.map((warpIdx, slot) => {
          const slotDelay = 2.5 * fps + slot * 0.08 * fps;
          const slotSpring = spring({
            frame: frame - slotDelay,
            fps,
            config: { damping: 200 },
          });
          const slotOpacity = interpolate(slotSpring, [0, 1], [0, 1]);
          const isCurrent = slot === activeCursorSlot;

          return (
            <div
              key={slot}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 4,
                opacity: slotOpacity,
              }}
            >
              <div
                style={{
                  width: 62,
                  height: 40,
                  borderRadius: 4,
                  backgroundColor: isCurrent
                    ? `${WARP_COLORS[warpIdx]}40`
                    : `${WARP_COLORS[warpIdx]}20`,
                  border: `2px solid ${isCurrent ? WARP_COLORS[warpIdx] : `${WARP_COLORS[warpIdx]}50`}`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 12,
                  color: WARP_COLORS[warpIdx],
                  fontFamily: fontFamilyCode,
                  fontWeight: 700,
                  transform: isCurrent ? "scale(1.1)" : "scale(1)",
                }}
              >
                W{warpIdx}
              </div>
              <span
                style={{
                  fontSize: 10,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyCode,
                }}
              >
                t{slot}
              </span>
            </div>
          );
        })}
      </div>

      {/* Latency hiding explanation */}
      <div
        style={{
          marginTop: 6,
          display: "flex",
          gap: 16,
          alignItems: "center",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <div
            style={{
              width: 14,
              height: 14,
              borderRadius: 3,
              backgroundColor: `${THEME.colors.nvidiaGreen}40`,
              border: `1.5px solid ${THEME.colors.nvidiaGreen}`,
            }}
          />
          <span
            style={{
              fontSize: 13,
              color: THEME.colors.textSecondary,
              fontFamily: fontFamilyBody,
            }}
          >
            Executing
          </span>
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <div
            style={{
              width: 14,
              height: 14,
              borderRadius: 3,
              backgroundColor: "rgba(255,255,255,0.04)",
              border: "1.5px solid rgba(255,255,255,0.15)",
            }}
          />
          <span
            style={{
              fontSize: 13,
              color: THEME.colors.textSecondary,
              fontFamily: fontFamilyBody,
            }}
          >
            Waiting on memory
          </span>
        </div>
      </div>
    </div>
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={3}>
      <SlideTitle
        title="Warp Scheduling & Latency Hiding"
        subtitle="When one warp stalls, another one runs"
      />

      <div style={{ display: "flex", gap: 40, flex: 1 }}>
        {/* Left: SM diagram + timeline */}
        <div
          style={{
            flex: 1.3,
            display: "flex",
            flexDirection: "column",
            gap: 20,
          }}
        >
          {renderSMDiagram()}
          {renderTimeline()}
        </div>

        {/* Right: Bullet points */}
        <div style={{ flex: 0.7 }}>
          <BulletPoint
            index={0}
            delay={3.5 * fps}
            text="Zero-cost context switch"
            subtext="Warp state is always resident on SM. Switching warps takes 0 cycles."
            highlight
          />
          <BulletPoint
            index={1}
            delay={3.5 * fps}
            text="Latency hiding = throughput"
            subtext="Memory latency is 400-800 cycles. Fill that time with other warps."
          />
          <BulletPoint
            index={2}
            delay={3.5 * fps}
            text="More warps = better hiding"
            subtext="Need enough resident warps to keep SM busy during memory stalls."
          />
          <BulletPoint
            index={3}
            delay={3.5 * fps}
            text="Occupancy matters"
            subtext="Ratio of resident warps to max possible. Higher = better latency hiding."
          />
        </div>
      </div>

      {/* Insight box */}
      <div
        style={{
          marginTop: 8,
          padding: "12px 24px",
          backgroundColor: "rgba(118,185,0,0.10)",
          borderRadius: 10,
          border: `2px solid ${THEME.colors.nvidiaGreen}50`,
          opacity: insightOpacity,
          textAlign: "center",
        }}
      >
        <span
          style={{
            fontSize: 20,
            color: THEME.colors.textPrimary,
            fontFamily: fontFamilyBody,
            fontWeight: 700,
          }}
        >
          GPU parallelism is about{" "}
          <span style={{ color: THEME.colors.nvidiaGreen }}>
            hiding latency
          </span>
          , not reducing it.
        </span>
      </div>
    </SlideLayout>
  );
};
