import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../styles/fonts";

export const S10_ThreadIndexing1D: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const blocks = [
    { id: 0, threads: [0, 1, 2, 3], globalStart: 0, color: THEME.colors.accentBlue },
    { id: 1, threads: [0, 1, 2, 3], globalStart: 4, color: THEME.colors.nvidiaGreen },
    { id: 2, threads: [0, 1, 2, 3], globalStart: 8, color: THEME.colors.accentPurple },
  ];

  return (
    <SlideLayout variant="gradient" slideNumber={10} totalSlides={18}>
      <SlideTitle
        title="1D Thread Indexing — Visualized"
        subtitle="3 blocks of 4 threads each — blockDim.x = 4"
      />

      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 24 }}>
        {/* Visual diagram */}
        <div style={{ display: "flex", gap: 16, justifyContent: "center", marginTop: 20 }}>
          {blocks.map((block, blockIdx) => {
            const blockDelay = 0.8 * fps + blockIdx * 0.5 * fps;

            return (
              <div
                key={block.id}
                style={{
                  border: `2px solid ${block.color}`,
                  borderRadius: 12,
                  padding: 16,
                  backgroundColor: `${block.color}08`,
                  opacity: interpolate(
                    frame - blockDelay,
                    [0, 0.3 * fps],
                    [0, 1],
                    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                  ),
                }}
              >
                <div style={{ fontSize: 16, fontWeight: 700, color: block.color, fontFamily: fontFamilyBody, marginBottom: 10, textAlign: "center" }}>
                  Block {block.id}
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  {block.threads.map((threadId, tIdx) => {
                    const tDelay = blockDelay + 0.3 * fps + tIdx * 3;
                    const tSpring = spring({ frame: frame - tDelay, fps, config: { damping: 200 } });

                    return (
                      <div key={tIdx} style={{ textAlign: "center", opacity: interpolate(tSpring, [0, 1], [0, 1]) }}>
                        <div
                          style={{
                            width: 90,
                            height: 56,
                            backgroundColor: `${block.color}20`,
                            border: `2px solid ${block.color}`,
                            borderRadius: 8,
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            justifyContent: "center",
                          }}
                        >
                          <span style={{ fontSize: 13, color: THEME.colors.textMuted, fontFamily: fontFamilyCode }}>
                            T{threadId}
                          </span>
                          <span style={{ fontSize: 16, color: block.color, fontWeight: 700, fontFamily: fontFamilyCode }}>
                            idx={block.globalStart + tIdx}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        {/* Formula breakdown */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: 40,
            marginTop: 16,
            opacity: interpolate(
              frame - 3 * fps,
              [0, 0.5 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          {/* Example calculation */}
          <div
            style={{
              padding: "20px 32px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 12,
              border: `1px solid rgba(118,185,0,0.2)`,
            }}
          >
            <div style={{ fontSize: 18, fontWeight: 700, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyBody, marginBottom: 12 }}>
              Example: Block 1, Thread 2
            </div>
            <div style={{ fontFamily: fontFamilyCode, fontSize: 20, color: THEME.colors.textPrimary, lineHeight: 2 }}>
              idx = <span style={{ color: THEME.colors.nvidiaGreen }}>blockIdx.x</span> * <span style={{ color: THEME.colors.accentBlue }}>blockDim.x</span> + <span style={{ color: THEME.colors.accentPurple }}>threadIdx.x</span>
            </div>
            <div style={{ fontFamily: fontFamilyCode, fontSize: 20, color: THEME.colors.textPrimary, lineHeight: 2 }}>
              idx = <span style={{ color: THEME.colors.nvidiaGreen }}>1</span> * <span style={{ color: THEME.colors.accentBlue }}>4</span> + <span style={{ color: THEME.colors.accentPurple }}>2</span> = <span style={{ color: THEME.colors.accentOrange, fontWeight: 700 }}>6</span>
            </div>
          </div>
        </div>

        {/* Built-in variables table */}
        <div
          style={{
            display: "flex",
            gap: 24,
            justifyContent: "center",
            marginTop: 12,
            opacity: interpolate(
              frame - 5 * fps,
              [0, 0.5 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          {[
            { var: "threadIdx.x", desc: "Thread index within block", val: "0-3", color: THEME.colors.accentPurple },
            { var: "blockIdx.x", desc: "Which block this thread is in", val: "0-2", color: THEME.colors.nvidiaGreen },
            { var: "blockDim.x", desc: "Threads per block", val: "4", color: THEME.colors.accentBlue },
            { var: "gridDim.x", desc: "Number of blocks", val: "3", color: THEME.colors.accentOrange },
          ].map((item) => (
            <div
              key={item.var}
              style={{
                padding: "12px 20px",
                backgroundColor: `${item.color}10`,
                borderRadius: 8,
                border: `1px solid ${item.color}30`,
                textAlign: "center",
              }}
            >
              <div style={{ fontFamily: fontFamilyCode, fontSize: 16, color: item.color, fontWeight: 700 }}>
                {item.var}
              </div>
              <div style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, marginTop: 4 }}>
                {item.desc}
              </div>
              <div style={{ fontSize: 14, color: THEME.colors.textMuted, fontFamily: fontFamilyCode, marginTop: 2 }}>
                = {item.val}
              </div>
            </div>
          ))}
        </div>
      </div>
    </SlideLayout>
  );
};
