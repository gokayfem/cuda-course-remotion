import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint, FadeInText } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

type FenceLevel = {
  name: string;
  scope: string;
  color: string;
  width: number;
  description: string;
  useCase: string;
};

const FENCE_LEVELS: FenceLevel[] = [
  {
    name: "__threadfence_block()",
    scope: "Block",
    color: THEME.colors.accentBlue,
    width: 280,
    description: "All writes visible to threads in same block",
    useCase: "Shared memory coordination within a block",
  },
  {
    name: "__threadfence()",
    scope: "Device",
    color: THEME.colors.accentOrange,
    width: 480,
    description: "All writes visible to all threads on device",
    useCase: "Global memory flag/counter visible across blocks",
  },
  {
    name: "__threadfence_system()",
    scope: "System",
    color: THEME.colors.accentPurple,
    width: 680,
    description: "All writes visible to host and all devices",
    useCase: "Unified memory, multi-GPU, CPU-GPU signaling",
  },
];

export const M3S07_MemoryFences: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const renderFenceDiagram = () => {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 16,
          marginTop: 8,
          marginBottom: 16,
        }}
      >
        {/* Nested scope visualization */}
        {FENCE_LEVELS.slice()
          .reverse()
          .map((fence, revIdx) => {
            const idx = FENCE_LEVELS.length - 1 - revIdx;
            const fenceDelay = 1.5 * fps + idx * 0.8 * fps;
            const fenceSpring = spring({
              frame: frame - fenceDelay,
              fps,
              config: { damping: 200 },
            });
            const fenceOpacity = interpolate(fenceSpring, [0, 1], [0, 1]);
            const fenceScale = interpolate(fenceSpring, [0, 1], [0.95, 1]);

            return (
              <div
                key={fence.name}
                style={{
                  width: fence.width,
                  padding: "10px 20px",
                  backgroundColor: `${fence.color}08`,
                  border: `2px solid ${fence.color}40`,
                  borderRadius: 12,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 4,
                  opacity: fenceOpacity,
                  transform: `scale(${fenceScale})`,
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                  }}
                >
                  <code
                    style={{
                      fontSize: 15,
                      color: fence.color,
                      fontFamily: fontFamilyCode,
                      fontWeight: 700,
                    }}
                  >
                    {fence.name}
                  </code>
                  <div
                    style={{
                      padding: "2px 8px",
                      backgroundColor: `${fence.color}20`,
                      borderRadius: 4,
                      fontSize: 12,
                      color: fence.color,
                      fontFamily: fontFamilyBody,
                      fontWeight: 600,
                    }}
                  >
                    {fence.scope} scope
                  </div>
                </div>
                <div
                  style={{
                    fontSize: 14,
                    color: THEME.colors.textSecondary,
                    fontFamily: fontFamilyBody,
                    textAlign: "center",
                  }}
                >
                  {fence.description}
                </div>
              </div>
            );
          })}
      </div>
    );
  };

  const renderScopeGrid = () => {
    const gridDelay = 4 * fps;
    const gridSpring = spring({
      frame: frame - gridDelay,
      fps,
      config: { damping: 200 },
    });
    const gridOpacity = interpolate(gridSpring, [0, 1], [0, 1]);

    const scopeItems = [
      {
        icon: "SM",
        label: "Block",
        sublabel: "Threads in same block",
        color: THEME.colors.accentBlue,
      },
      {
        icon: "GPU",
        label: "Device",
        sublabel: "All threads on GPU",
        color: THEME.colors.accentOrange,
      },
      {
        icon: "SYS",
        label: "System",
        sublabel: "GPU + CPU + other GPUs",
        color: THEME.colors.accentPurple,
      },
    ];

    return (
      <div
        style={{
          display: "flex",
          gap: 20,
          justifyContent: "center",
          opacity: gridOpacity,
        }}
      >
        {scopeItems.map((item, i) => {
          const itemDelay = gridDelay + i * 0.2 * fps;
          const itemSpring = spring({
            frame: frame - itemDelay,
            fps,
            config: { damping: 200 },
          });
          const itemScale = interpolate(itemSpring, [0, 1], [0.8, 1]);

          return (
            <div
              key={item.label}
              style={{
                width: 200,
                padding: "12px 16px",
                backgroundColor: `${item.color}08`,
                border: `1.5px solid ${item.color}40`,
                borderRadius: 10,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 6,
                transform: `scale(${itemScale})`,
              }}
            >
              <div
                style={{
                  width: 44,
                  height: 44,
                  borderRadius: 22,
                  backgroundColor: `${item.color}20`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 14,
                  color: item.color,
                  fontFamily: fontFamilyCode,
                  fontWeight: 700,
                }}
              >
                {item.icon}
              </div>
              <div
                style={{
                  fontSize: 16,
                  color: item.color,
                  fontFamily: fontFamilyBody,
                  fontWeight: 700,
                }}
              >
                {item.label}
              </div>
              <div
                style={{
                  fontSize: 13,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  textAlign: "center",
                }}
              >
                {item.sublabel}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <SlideLayout variant="gradient" moduleNumber={3}>
      <SlideTitle
        title="Memory Fences"
        subtitle="Ensuring write visibility across different scopes"
      />

      <div
        style={{
          display: "flex",
          gap: 40,
          flex: 1,
        }}
      >
        {/* Left: Diagram */}
        <div
          style={{
            flex: 1.2,
            display: "flex",
            flexDirection: "column",
            gap: 12,
          }}
        >
          <FadeInText
            text="Fence scope (nested, inner to outer):"
            fontSize={16}
            fontWeight={600}
            delay={1 * fps}
          />
          {renderFenceDiagram()}
          {renderScopeGrid()}
        </div>

        {/* Right: Bullet points */}
        <div style={{ flex: 0.8 }}>
          <FadeInText
            text="When to Use Each"
            fontSize={20}
            fontWeight={700}
            delay={5 * fps}
            color={THEME.colors.accentBlue}
          />

          <div style={{ marginTop: 12 }}>
            <BulletPoint
              index={0}
              delay={5.5 * fps}
              text="Fence != Barrier"
              subtext="Fences guarantee memory visibility order, not execution synchronization."
              highlight
            />
            <BulletPoint
              index={1}
              delay={5.5 * fps}
              text="Block fence for shared mem"
              subtext="Use __threadfence_block() when coordinating via shared memory within a block."
            />
            <BulletPoint
              index={2}
              delay={5.5 * fps}
              text="Device fence for global flags"
              subtext="Use __threadfence() when one block signals another via global memory."
            />
            <BulletPoint
              index={3}
              delay={5.5 * fps}
              text="System fence for CPU-GPU"
              subtext="Use __threadfence_system() for unified memory or mapped host memory."
            />
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
