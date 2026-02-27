import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";
import { AbsoluteFill } from "remotion";
import { SlideBackground } from "../../../../components/SlideBackground";
import { fontFamilyHeading } from "../../../../styles/fonts";

export const M3S01_Title: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const barWidth = interpolate(frame, [0, 1 * fps], [0, 400], {
    extrapolateRight: "clamp",
  });

  const titleSpring = spring({
    frame,
    fps,
    config: { damping: 200 },
    delay: 0.3 * fps,
  });
  const titleOpacity = interpolate(titleSpring, [0, 1], [0, 1]);
  const titleY = interpolate(titleSpring, [0, 1], [40, 0]);

  const subtitleOpacity = interpolate(
    frame,
    [1 * fps, 1.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const moduleOpacity = interpolate(
    frame,
    [1.5 * fps, 2 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const WARP_SIZE = 32;
  const THREAD_SIZE = 18;
  const THREAD_GAP = 3;

  return (
    <AbsoluteFill>
      <SlideBackground variant="accent" />

      {/* Warp visualization on right */}
      <div
        style={{
          position: "absolute",
          right: 80,
          top: 180,
          width: 600,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 20,
        }}
      >
        {/* Warp label */}
        <div
          style={{
            fontSize: 18,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyBody,
            fontWeight: 700,
            opacity: interpolate(
              frame - 1.2 * fps,
              [0, 0.5 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          32 Threads = 1 Warp (executing in lockstep)
        </div>

        {/* 32 thread blocks in 4 rows of 8 */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: THREAD_GAP,
            alignItems: "center",
          }}
        >
          {Array.from({ length: 4 }).map((_, row) => (
            <div key={row} style={{ display: "flex", gap: THREAD_GAP }}>
              {Array.from({ length: 8 }).map((_, col) => {
                const idx = row * 8 + col;
                const threadDelay = 1 * fps + idx * 0.03 * fps;
                const threadSpring = spring({
                  frame: frame - threadDelay,
                  fps,
                  config: { damping: 200 },
                });
                const scale = interpolate(threadSpring, [0, 1], [0.3, 1]);
                const opacity = interpolate(threadSpring, [0, 1], [0, 1]);

                // Pulsing animation for lockstep
                const pulsePhase = (frame - 2.5 * fps) / (0.5 * fps);
                const pulse =
                  frame > 2.5 * fps
                    ? interpolate(
                        Math.sin(pulsePhase * Math.PI * 2),
                        [-1, 1],
                        [0.6, 1]
                      )
                    : 1;

                return (
                  <div
                    key={idx}
                    style={{
                      width: THREAD_SIZE * 3,
                      height: THREAD_SIZE * 2,
                      borderRadius: 4,
                      backgroundColor: `${THEME.colors.nvidiaGreen}${Math.round(pulse * 40).toString(16).padStart(2, "0")}`,
                      border: `1.5px solid ${THEME.colors.nvidiaGreen}${Math.round(pulse * 180).toString(16).padStart(2, "0")}`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 11,
                      color: THEME.colors.nvidiaGreen,
                      fontFamily: fontFamilyCode,
                      fontWeight: 700,
                      transform: `scale(${scale})`,
                      opacity,
                    }}
                  >
                    T{idx}
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        {/* Instruction arrow */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            opacity: interpolate(
              frame - 2.5 * fps,
              [0, 0.5 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          <div
            style={{
              width: 40,
              height: 3,
              backgroundColor: THEME.colors.accentCyan,
              borderRadius: 2,
            }}
          />
          <span
            style={{
              fontSize: 16,
              color: THEME.colors.accentCyan,
              fontFamily: fontFamilyCode,
              fontWeight: 600,
            }}
          >
            Same instruction, all 32 threads
          </span>
          <div
            style={{
              width: 40,
              height: 3,
              backgroundColor: THEME.colors.accentCyan,
              borderRadius: 2,
            }}
          />
        </div>
      </div>

      {/* Main content */}
      <div
        style={{
          position: "absolute",
          left: 100,
          top: "50%",
          transform: "translateY(-50%)",
          maxWidth: 800,
        }}
      >
        <div
          style={{
            width: barWidth,
            height: 6,
            backgroundColor: THEME.colors.nvidiaGreen,
            borderRadius: 3,
            marginBottom: 32,
          }}
        />

        <h1
          style={{
            fontSize: 72,
            fontWeight: 900,
            color: THEME.colors.textPrimary,
            fontFamily: fontFamilyHeading,
            margin: 0,
            opacity: titleOpacity,
            transform: `translateY(${titleY}px)`,
            lineHeight: 1.1,
            letterSpacing: "-2px",
          }}
        >
          Thread{" "}
          <span style={{ color: THEME.colors.nvidiaGreen }}>
            Synchronization
          </span>
          <br />& Execution Model
        </h1>

        <p
          style={{
            fontSize: 28,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
            margin: 0,
            marginTop: 24,
            opacity: subtitleOpacity,
            fontWeight: 400,
          }}
        >
          Warps, Barriers, Atomics & Warp Primitives
        </p>

        <div
          style={{
            marginTop: 48,
            display: "flex",
            gap: 16,
            alignItems: "center",
            opacity: moduleOpacity,
          }}
        >
          <div
            style={{
              padding: "12px 28px",
              backgroundColor: "rgba(118,185,0,0.15)",
              border: `2px solid ${THEME.colors.nvidiaGreen}`,
              borderRadius: 30,
              fontSize: 22,
              color: THEME.colors.nvidiaGreen,
              fontFamily: fontFamilyBody,
              fontWeight: 700,
            }}
          >
            Module 3
          </div>
          <span
            style={{
              fontSize: 22,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
            }}
          >
            Mastering GPU thread coordination
          </span>
        </div>
      </div>

      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          height: 6,
          background: `linear-gradient(90deg, ${THEME.colors.nvidiaGreen}, ${THEME.colors.accentCyan}, ${THEME.colors.accentBlue})`,
        }}
      />
    </AbsoluteFill>
  );
};
