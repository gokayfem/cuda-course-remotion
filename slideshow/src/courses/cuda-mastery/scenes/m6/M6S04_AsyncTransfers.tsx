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

const ENGINE_BLOCK_H = 28;
const ENGINE_TRACK_W = 320;

interface EngineBlock {
  label: string;
  width: number;
  color: string;
  x: number;
  delay: number;
}

const copyEngineBlocks: EngineBlock[] = [
  { label: "H2D chunk 0", width: 90, color: THEME.colors.accentBlue, x: 0, delay: 3 },
  { label: "H2D chunk 1", width: 90, color: THEME.colors.accentBlue, x: 100, delay: 4.2 },
  { label: "D2H chunk 0", width: 80, color: THEME.colors.accentOrange, x: 200, delay: 5.5 },
];

const computeEngineBlocks: EngineBlock[] = [
  { label: "Kernel 0", width: 110, color: THEME.colors.nvidiaGreen, x: 50, delay: 3.8 },
  { label: "Kernel 1", width: 110, color: THEME.colors.nvidiaGreen, x: 170, delay: 5 },
];

const codeLines = [
  { text: "// Allocate pinned host memory", color: THEME.colors.syntaxComment },
  { text: "float *h_data;", color: THEME.colors.textCode },
  { text: "cudaMallocHost(&h_data, size);", color: THEME.colors.syntaxFunction },
  { text: "", color: "transparent" },
  { text: "// Async copy (returns immediately)", color: THEME.colors.syntaxComment },
  { text: "cudaMemcpyAsync(d_data, h_data,", color: THEME.colors.syntaxFunction },
  { text: "  size, cudaMemcpyHostToDevice,", color: THEME.colors.textCode },
  { text: "  stream);", color: THEME.colors.textCode },
];

export const M6S04_AsyncTransfers: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Key concept box
  const conceptSpring = spring({
    frame: frame - 0.5 * fps,
    fps,
    config: { damping: 200 },
  });
  const conceptOpacity = interpolate(conceptSpring, [0, 1], [0, 1]);
  const conceptScale = interpolate(conceptSpring, [0, 1], [0.95, 1]);

  // Engines label
  const engineLabelOpacity = interpolate(
    frame - 2 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Parallel running pulse
  const pulse = interpolate(
    frame % (1.5 * fps),
    [0, 0.75 * fps, 1.5 * fps],
    [0.3, 1, 0.3],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Code block
  const codeOpacity = interpolate(
    frame - 8 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={6}
      leftWidth="52%"
      left={
        <div style={{ width: 560 }}>
          <SlideTitle
            title="Async Memory Transfers"
            subtitle="Non-blocking copies with separate engines"
          />

          {/* Key concept box */}
          <div
            style={{
              padding: "12px 20px",
              backgroundColor: "rgba(79,195,247,0.08)",
              borderRadius: 10,
              border: `2px solid ${THEME.colors.accentBlue}40`,
              opacity: conceptOpacity,
              transform: `scale(${conceptScale})`,
              transformOrigin: "left center",
              marginBottom: 24,
              width: 530,
            }}
          >
            <span
              style={{
                fontSize: 16,
                fontWeight: 700,
                color: THEME.colors.accentBlue,
                fontFamily: fontFamilyHeading,
              }}
            >
              cudaMemcpyAsync
            </span>
            <span
              style={{
                fontSize: 16,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
              }}
            >
              {" "}= non-blocking copy (returns immediately to CPU)
            </span>
          </div>

          {/* Two engines diagram */}
          <div style={{ width: 540, opacity: engineLabelOpacity }}>
            {/* Copy Engine */}
            <div style={{ marginBottom: 16, width: 530 }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  marginBottom: 8,
                }}
              >
                <div
                  style={{
                    padding: "4px 14px",
                    backgroundColor: "rgba(79,195,247,0.12)",
                    borderRadius: 6,
                    fontSize: 14,
                    fontWeight: 700,
                    color: THEME.colors.accentBlue,
                    fontFamily: fontFamilyBody,
                  }}
                >
                  Copy Engine (DMA)
                </div>
                <span
                  style={{
                    fontSize: 12,
                    color: THEME.colors.textMuted,
                    fontFamily: fontFamilyBody,
                  }}
                >
                  handles memory transfers
                </span>
              </div>

              <div
                style={{
                  position: "relative",
                  height: ENGINE_BLOCK_H + 8,
                  backgroundColor: "rgba(79,195,247,0.04)",
                  borderRadius: 6,
                  border: `1px solid ${THEME.colors.accentBlue}20`,
                  width: ENGINE_TRACK_W + 40,
                }}
              >
                {copyEngineBlocks.map((block, i) => {
                  const bSpring = spring({
                    frame: frame - block.delay * fps,
                    fps,
                    config: { damping: 150 },
                  });
                  const bOpacity = interpolate(bSpring, [0, 1], [0, 1]);
                  const bX = interpolate(bSpring, [0, 1], [-20, 0]);

                  return (
                    <div
                      key={`copy-${i}`}
                      style={{
                        position: "absolute",
                        left: block.x + 8,
                        top: 4,
                        width: block.width,
                        height: ENGINE_BLOCK_H,
                        backgroundColor: `${block.color}cc`,
                        borderRadius: 5,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 11,
                        fontFamily: fontFamilyCode,
                        fontWeight: 700,
                        color: "#000",
                        opacity: bOpacity,
                        transform: `translateX(${bX}px)`,
                      }}
                    >
                      {block.label}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Parallel indicator */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 8,
                marginBottom: 16,
                opacity: frame > 4 * fps ? pulse : 0,
                width: ENGINE_TRACK_W + 40,
              }}
            >
              <div
                style={{
                  width: 40,
                  height: 2,
                  backgroundColor: THEME.colors.nvidiaGreen,
                }}
              />
              <span
                style={{
                  fontSize: 13,
                  fontWeight: 700,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyBody,
                }}
              >
                PARALLEL
              </span>
              <div
                style={{
                  width: 40,
                  height: 2,
                  backgroundColor: THEME.colors.nvidiaGreen,
                }}
              />
            </div>

            {/* Compute Engine */}
            <div style={{ width: 530 }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  marginBottom: 8,
                }}
              >
                <div
                  style={{
                    padding: "4px 14px",
                    backgroundColor: "rgba(118,185,0,0.12)",
                    borderRadius: 6,
                    fontSize: 14,
                    fontWeight: 700,
                    color: THEME.colors.nvidiaGreen,
                    fontFamily: fontFamilyBody,
                  }}
                >
                  Compute Engine (SMs)
                </div>
                <span
                  style={{
                    fontSize: 12,
                    color: THEME.colors.textMuted,
                    fontFamily: fontFamilyBody,
                  }}
                >
                  handles kernel execution
                </span>
              </div>

              <div
                style={{
                  position: "relative",
                  height: ENGINE_BLOCK_H + 8,
                  backgroundColor: "rgba(118,185,0,0.04)",
                  borderRadius: 6,
                  border: `1px solid ${THEME.colors.nvidiaGreen}20`,
                  width: ENGINE_TRACK_W + 40,
                }}
              >
                {computeEngineBlocks.map((block, i) => {
                  const bSpring = spring({
                    frame: frame - block.delay * fps,
                    fps,
                    config: { damping: 150 },
                  });
                  const bOpacity = interpolate(bSpring, [0, 1], [0, 1]);
                  const bX = interpolate(bSpring, [0, 1], [-20, 0]);

                  return (
                    <div
                      key={`compute-${i}`}
                      style={{
                        position: "absolute",
                        left: block.x + 8,
                        top: 4,
                        width: block.width,
                        height: ENGINE_BLOCK_H,
                        backgroundColor: `${block.color}cc`,
                        borderRadius: 5,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 11,
                        fontFamily: fontFamilyCode,
                        fontWeight: 700,
                        color: "#000",
                        opacity: bOpacity,
                        transform: `translateX(${bX}px)`,
                      }}
                    >
                      {block.label}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Bottom code snippet */}
          <div
            style={{
              marginTop: 20,
              backgroundColor: THEME.colors.bgCode,
              borderRadius: 8,
              padding: "12px 16px",
              border: `1px solid rgba(255,255,255,0.08)`,
              opacity: codeOpacity,
              width: 530,
            }}
          >
            {codeLines.map((line, i) => (
              <div
                key={i}
                style={{
                  fontSize: 13,
                  fontFamily: fontFamilyCode,
                  color: line.color,
                  lineHeight: 1.6,
                  minHeight: line.text === "" ? 8 : "auto",
                  whiteSpace: "nowrap",
                }}
              >
                {line.text}
              </div>
            ))}
          </div>
        </div>
      }
      right={
        <div style={{ paddingTop: 50, width: 440 }}>
          <FadeInText
            text="Requirements & Capabilities"
            delay={2 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 20, width: 440 }}
          />

          <BulletPoint
            index={0}
            delay={3 * fps}
            text="Requires pinned (page-locked) host memory"
            icon="1"
            highlight
          />
          <BulletPoint
            index={1}
            delay={3 * fps}
            text="cudaMallocHost() or cudaHostAlloc()"
            icon="2"
            subtext="Prevents OS from paging out memory"
          />
          <BulletPoint
            index={2}
            delay={3 * fps}
            text="DMA can transfer while SMs compute"
            icon="3"
            highlight
          />
          <BulletPoint
            index={3}
            delay={3 * fps}
            text="Modern GPUs: 2 copy engines (H2D + D2H simultaneously)"
            icon="4"
            subtext="PCIe is full-duplex on most platforms"
          />

          {/* Note box */}
          <div
            style={{
              marginTop: 24,
              padding: "12px 18px",
              backgroundColor: "rgba(179,136,255,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.accentPurple}40`,
              opacity: interpolate(
                frame - 10 * fps,
                [0, 0.5 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
              width: 420,
            }}
          >
            <span
              style={{
                fontSize: 15,
                color: THEME.colors.accentPurple,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              The CPU thread is free to do other work while the GPU handles the async transfer!
            </span>
          </div>
        </div>
      }
    />
  );
};
