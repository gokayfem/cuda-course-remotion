import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText, BulletPoint } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

export const M2S14_ConstantMemory: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const broadcastOpacity = interpolate(
    frame - 5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const useCaseOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const warpThreads = Array.from({ length: 8 });

  return (
    <SlideLayout variant="gradient" moduleNumber={2} slideNumber={14} totalSlides={18}>
      <SlideTitle
        title="Constant Memory — Broadcast to All Threads"
        subtitle="64KB read-only cache, optimized for uniform access across a warp"
      />

      <div style={{ display: "flex", gap: 36, flex: 1 }}>
        {/* Left: Properties and code */}
        <div style={{ flex: 1 }}>
          <BulletPoint index={0} delay={0.8 * fps} text="64 KB total, read-only from GPU" icon="1" />
          <BulletPoint index={1} delay={0.8 * fps} text="Written by CPU via cudaMemcpyToSymbol" icon="2" />
          <BulletPoint
            index={2}
            delay={0.8 * fps}
            text="Hardware broadcast: 1 read serves entire warp"
            subtext="All 32 threads get the value in a single transaction"
            icon="3"
            highlight
          />
          <BulletPoint index={3} delay={0.8 * fps} text="Cached in dedicated constant cache (per SM)" icon="4" />

          <div style={{ marginTop: 12 }}>
            <CodeBlock
              delay={2.5 * fps}
              title="constant_memory.cu"
              fontSize={16}
              code={`// Declare in global scope (64 KB max)
__constant__ float filter[3][3];

// Host: copy data to constant memory
float h_filter[3][3] = { ... };
cudaMemcpyToSymbol(filter, h_filter,
                   sizeof(h_filter));

// All threads read the SAME filter values
__global__ void conv2d(float *input,
                       float *output,
                       int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;
    for (int fy = 0; fy < 3; fy++)
      for (int fx = 0; fx < 3; fx++)
        sum += input[(y+fy)*W + (x+fx)]
             * filter[fy][fx]; // broadcast!
    output[y * W + x] = sum;
}`}
              highlightLines={[2, 6, 19]}
            />
          </div>
        </div>

        {/* Right: Broadcast diagram and use cases */}
        <div style={{ flex: 0.75, display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Broadcast diagram */}
          <div style={{ opacity: broadcastOpacity }}>
            <FadeInText
              text="Warp Broadcast"
              delay={5 * fps}
              fontSize={20}
              fontWeight={700}
              color={THEME.colors.accentPurple}
              style={{ marginBottom: 12 }}
            />

            <div style={{ position: "relative", height: 150, marginBottom: 8 }}>
              {/* Constant cache box */}
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  left: 60,
                  width: 200,
                  height: 40,
                  backgroundColor: `${THEME.colors.accentPurple}15`,
                  border: `2px solid ${THEME.colors.accentPurple}`,
                  borderRadius: 8,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 15,
                  color: THEME.colors.accentPurple,
                  fontFamily: fontFamilyCode,
                  fontWeight: 700,
                }}
              >
                filter[0][0] = 1.5
              </div>

              {/* Arrow down */}
              <svg style={{ position: "absolute", left: 0, top: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
                <line x1="160" y1="42" x2="160" y2="65" stroke={THEME.colors.accentPurple} strokeWidth="2" />
                <text x="170" y="58" fill={THEME.colors.accentPurple} fontSize="14" fontFamily={fontFamilyBody}>1 read</text>
              </svg>

              {/* Warp threads */}
              <div style={{ position: "absolute", top: 70, left: 0, display: "flex", gap: 6, flexWrap: "wrap", width: 320 }}>
                {warpThreads.map((_, i) => {
                  const threadDelay = 5.5 * fps + i * 0.08 * fps;
                  const threadSpring = spring({ frame: frame - threadDelay, fps, config: { damping: 200 } });
                  const threadOpacity = interpolate(threadSpring, [0, 1], [0, 1]);

                  return (
                    <div
                      key={i}
                      style={{
                        width: 34,
                        height: 28,
                        backgroundColor: `${THEME.colors.nvidiaGreen}20`,
                        border: `1.5px solid ${THEME.colors.nvidiaGreen}`,
                        borderRadius: 4,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 11,
                        color: THEME.colors.nvidiaGreen,
                        fontFamily: fontFamilyCode,
                        fontWeight: 600,
                        opacity: threadOpacity,
                      }}
                    >
                      T{i}
                    </div>
                  );
                })}
              </div>
              <div
                style={{
                  position: "absolute",
                  top: 105,
                  left: 0,
                  width: 320,
                  textAlign: "center",
                  fontSize: 14,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyBody,
                }}
              >
                All 32 threads get 1.5 simultaneously
              </div>
            </div>
          </div>

          {/* Use cases */}
          <div style={{ opacity: useCaseOpacity }}>
            <FadeInText
              text="Perfect Use Cases"
              delay={7 * fps}
              fontSize={20}
              fontWeight={700}
              color={THEME.colors.accentOrange}
              style={{ marginBottom: 10 }}
            />
            {[
              { label: "Convolution filters", desc: "Same kernel applied everywhere" },
              { label: "Bias parameters", desc: "Added to every output element" },
              { label: "Lookup tables", desc: "Activation function approximations" },
              { label: "Hyperparameters", desc: "Learning rate, epsilon, etc." },
            ].map((item, i) => {
              const itemDelay = 7.5 * fps + i * 0.25 * fps;
              const s = spring({ frame: frame - itemDelay, fps, config: { damping: 200 } });
              return (
                <div
                  key={item.label}
                  style={{
                    display: "flex",
                    alignItems: "baseline",
                    gap: 8,
                    marginBottom: 8,
                    opacity: interpolate(s, [0, 1], [0, 1]),
                    transform: `translateX(${interpolate(s, [0, 1], [-15, 0])}px)`,
                  }}
                >
                  <span style={{ color: THEME.colors.accentOrange, fontSize: 15, fontWeight: 700, fontFamily: fontFamilyBody }}>
                    {item.label}
                  </span>
                  <span style={{ color: THEME.colors.textMuted, fontSize: 14, fontFamily: fontFamilyBody }}>
                    — {item.desc}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
