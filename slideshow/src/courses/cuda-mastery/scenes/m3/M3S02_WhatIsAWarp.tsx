import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M3S02_WhatIsAWarp: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const WARP_SIZE = 32;
  const CELL_SIZE = 38;
  const CELL_GAP = 3;

  const warpGroupSpring = spring({
    frame: frame - 0.8 * fps,
    fps,
    config: { damping: 200 },
  });
  const warpGroupOpacity = interpolate(warpGroupSpring, [0, 1], [0, 1]);

  const formulaOpacity = interpolate(
    frame - 3 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const simtOpacity = interpolate(
    frame - 4 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const renderWarpDiagram = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* SIMT label */}
      <div
        style={{
          padding: "6px 14px",
          backgroundColor: "rgba(118,185,0,0.12)",
          borderRadius: 8,
          fontSize: 16,
          color: THEME.colors.nvidiaGreen,
          fontFamily: fontFamilyBody,
          fontWeight: 700,
          alignSelf: "flex-start",
          opacity: simtOpacity,
        }}
      >
        SIMT: Single Instruction, Multiple Threads
      </div>

      {/* 32 threads in 4 rows of 8 */}
      <div
        style={{
          opacity: warpGroupOpacity,
          display: "flex",
          flexDirection: "column",
          gap: CELL_GAP,
        }}
      >
        <div
          style={{
            fontSize: 14,
            color: THEME.colors.textMuted,
            fontFamily: fontFamilyBody,
            marginBottom: 4,
          }}
        >
          Warp 0 (threads 0-31):
        </div>
        {Array.from({ length: 4 }).map((_, row) => (
          <div key={row} style={{ display: "flex", gap: CELL_GAP }}>
            {Array.from({ length: 8 }).map((_, col) => {
              const idx = row * 8 + col;
              const cellDelay = 0.8 * fps + idx * 0.02 * fps;
              const cellSpring = spring({
                frame: frame - cellDelay,
                fps,
                config: { damping: 200 },
              });
              const scale = interpolate(cellSpring, [0, 1], [0.5, 1]);

              // Highlight to show lockstep execution
              const cycleFrame = frame - 4.5 * fps;
              const isExecuting =
                cycleFrame > 0 &&
                Math.floor(cycleFrame / (0.5 * fps)) % 2 === 0;

              return (
                <div
                  key={idx}
                  style={{
                    width: CELL_SIZE,
                    height: CELL_SIZE,
                    borderRadius: 4,
                    backgroundColor: isExecuting
                      ? `${THEME.colors.nvidiaGreen}35`
                      : "rgba(255,255,255,0.06)",
                    border: `1.5px solid ${isExecuting ? THEME.colors.nvidiaGreen : "rgba(255,255,255,0.12)"}`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 12,
                    color: isExecuting
                      ? THEME.colors.nvidiaGreen
                      : THEME.colors.textSecondary,
                    fontFamily: fontFamilyCode,
                    fontWeight: 600,
                    transform: `scale(${scale})`,
                    transition: "background-color 0.15s, border-color 0.15s",
                  }}
                >
                  T{idx}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* Warp ID formula */}
      <div
        style={{
          marginTop: 12,
          padding: "10px 16px",
          backgroundColor: "rgba(79,195,247,0.10)",
          border: `1px solid ${THEME.colors.accentBlue}50`,
          borderRadius: 8,
          opacity: formulaOpacity,
        }}
      >
        <div
          style={{
            fontSize: 15,
            color: THEME.colors.accentBlue,
            fontFamily: fontFamilyCode,
            fontWeight: 700,
            marginBottom: 4,
          }}
        >
          Warp ID = threadIdx.x / 32
        </div>
        <div
          style={{
            fontSize: 14,
            color: THEME.colors.textSecondary,
            fontFamily: fontFamilyBody,
          }}
        >
          Lane ID = threadIdx.x % 32 (thread's position within warp)
        </div>
      </div>

      {/* Warp size badge */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          marginTop: 8,
          opacity: formulaOpacity,
        }}
      >
        <div
          style={{
            padding: "6px 14px",
            backgroundColor: "rgba(255,171,64,0.12)",
            border: `1px solid ${THEME.colors.accentOrange}50`,
            borderRadius: 6,
            fontSize: 15,
            color: THEME.colors.accentOrange,
            fontFamily: fontFamilyCode,
            fontWeight: 700,
          }}
        >
          warpSize = 32
        </div>
        <span
          style={{
            fontSize: 14,
            color: THEME.colors.textMuted,
            fontFamily: fontFamilyBody,
          }}
        >
          All NVIDIA GPUs since Kepler
        </span>
      </div>
    </div>
  );

  const renderBullets = () => (
    <div>
      <SlideTitle
        title="What Is a Warp?"
        subtitle="The fundamental execution unit on the GPU"
      />
      <BulletPoint
        index={0}
        delay={1.5 * fps}
        text="Warp = 32 threads, always"
        subtext="The GPU hardware groups threads into warps of 32. This is fixed and immutable."
        highlight
      />
      <BulletPoint
        index={1}
        delay={1.5 * fps}
        text="All 32 execute the same instruction"
        subtext="At any cycle, every thread in a warp runs the identical instruction (SIMT model)."
      />
      <BulletPoint
        index={2}
        delay={1.5 * fps}
        text="SM schedules warps, not threads"
        subtext="The Streaming Multiprocessor picks an entire warp to issue an instruction. Individual threads are never scheduled independently."
      />
      <BulletPoint
        index={3}
        delay={1.5 * fps}
        text="Block of 256 threads = 8 warps"
        subtext="A block is divided into warps in order: threads 0-31 = warp 0, threads 32-63 = warp 1, etc."
      />
    </div>
  );

  return (
    <TwoColumnLayout
      moduleNumber={3}
      left={renderWarpDiagram()}
      right={renderBullets()}
      leftWidth="45%"
    />
  );
};
