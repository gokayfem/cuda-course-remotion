import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

type TimelineBlock = {
  label: string;
  color: string;
  width: number;
  isCallback?: boolean;
};

const timelineBlocks: TimelineBlock[] = [
  { label: "Kernel", color: THEME.colors.nvidiaGreen, width: 140 },
  { label: "Callback fires!", color: THEME.colors.accentOrange, width: 130, isCallback: true },
  { label: "Next kernel", color: THEME.colors.accentBlue, width: 140 },
];

export const M6S12_StreamCallbacks: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const bottomOpacity = interpolate(
    frame - 8 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="gradient"
      moduleNumber={6}
      leftWidth="50%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle title="Stream Callbacks & Graph Capture" />

          {/* Callback timeline diagram */}
          <FadeInText
            text="Callback Timeline"
            delay={0.8 * fps}
            fontSize={18}
            fontWeight={700}
            color={THEME.colors.accentOrange}
            style={{ marginBottom: 14 }}
          />

          {/* Stream label */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              marginBottom: 10,
              opacity: interpolate(
                frame - 1.2 * fps,
                [0, 0.4 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div
              style={{
                fontSize: 14,
                color: THEME.colors.textMuted,
                fontFamily: fontFamilyCode,
                width: 60,
                flexShrink: 0,
              }}
            >
              Stream:
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
              {timelineBlocks.map((block, i) => {
                const blockDelay = 1.5 * fps + i * 0.8 * fps;
                const blockSpring = spring({
                  frame: frame - blockDelay,
                  fps,
                  config: { damping: 200 },
                });
                const blockOpacity = interpolate(blockSpring, [0, 1], [0, 1]);
                const blockScale = interpolate(blockSpring, [0, 1], [0.8, 1]);

                return (
                  <React.Fragment key={block.label}>
                    <div
                      style={{
                        width: block.width,
                        height: 44,
                        backgroundColor: `${block.color}18`,
                        border: `2px solid ${block.color}`,
                        borderRadius: 8,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        opacity: blockOpacity,
                        transform: `scale(${blockScale})`,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 13,
                          fontWeight: 700,
                          color: block.color,
                          fontFamily: fontFamilyBody,
                        }}
                      >
                        {block.label}
                      </span>
                    </div>
                    {i < timelineBlocks.length - 1 && (
                      <div
                        style={{
                          width: 20,
                          height: 2,
                          backgroundColor: THEME.colors.textMuted,
                          opacity: blockOpacity,
                        }}
                      />
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>

          {/* CPU callback indicator */}
          <div
            style={{
              marginLeft: 72,
              marginTop: 6,
              marginBottom: 20,
              opacity: interpolate(
                frame - 3.5 * fps,
                [0, 0.4 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <div
                style={{
                  width: 0,
                  height: 0,
                  borderLeft: "6px solid transparent",
                  borderRight: "6px solid transparent",
                  borderBottom: `8px solid ${THEME.colors.accentOrange}`,
                  marginLeft: 150,
                }}
              />
              <span
                style={{
                  fontSize: 12,
                  color: THEME.colors.accentOrange,
                  fontFamily: fontFamilyCode,
                }}
              >
                CPU function triggered
              </span>
            </div>
          </div>

          <CodeBlock
            delay={4 * fps}
            fontSize={15}
            code={`cudaLaunchHostFunc(stream, myCallback, userData);`}
            showLineNumbers={false}
          />

          {/* Bottom note */}
          <div
            style={{
              marginTop: 24,
              padding: "12px 18px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.nvidiaGreen}30`,
              opacity: bottomOpacity,
              width: 700,
            }}
          >
            <div
              style={{
                fontSize: 14,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                lineHeight: 1.5,
              }}
            >
              CUDA Graphs are essential for inference serving (low-latency repeated execution)
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ width: 500, marginTop: 80 }}>
          <FadeInText
            text="CUDA Graphs"
            delay={5 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 16 }}
          />

          {[
            "Record a stream -> replay as a graph",
            "Eliminates per-launch overhead",
            "10-100x lower launch latency",
          ].map((text, i) => {
            const itemDelay = 5.5 * fps + i * 0.4 * fps;
            const itemSpring = spring({
              frame: frame - itemDelay,
              fps,
              config: { damping: 200 },
            });
            const itemOpacity = interpolate(itemSpring, [0, 1], [0, 1]);

            return (
              <div
                key={text}
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 10,
                  marginBottom: 12,
                  opacity: itemOpacity,
                }}
              >
                <span
                  style={{
                    color: THEME.colors.accentCyan,
                    fontSize: 18,
                    flexShrink: 0,
                    lineHeight: 1.5,
                  }}
                >
                  {">>"}
                </span>
                <span
                  style={{
                    fontSize: 18,
                    color: THEME.colors.textPrimary,
                    fontFamily: fontFamilyBody,
                    lineHeight: 1.5,
                  }}
                >
                  {text}
                </span>
              </div>
            );
          })}

          <div style={{ marginTop: 16 }}>
            <CodeBlock
              delay={7 * fps}
              fontSize={14}
              code={`cudaStreamBeginCapture(stream, ...);
// ... enqueue work ...
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph);
cudaGraphLaunch(instance, stream);`}
              showLineNumbers={false}
              title="graph_capture.cu"
            />
          </div>
        </div>
      }
    />
  );
};
