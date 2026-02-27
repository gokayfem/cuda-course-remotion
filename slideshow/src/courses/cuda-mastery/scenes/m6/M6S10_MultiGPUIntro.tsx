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

const GPUCard: React.FC<{
  label: string;
  x: number;
  delay: number;
}> = ({ label, x, delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const cardSpring = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });
  const opacity = interpolate(cardSpring, [0, 1], [0, 1]);
  const scale = interpolate(cardSpring, [0, 1], [0.85, 1]);

  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: 30,
        width: 200,
        height: 260,
        backgroundColor: `${THEME.colors.accentBlue}10`,
        border: `2px solid ${THEME.colors.accentBlue}`,
        borderRadius: 12,
        opacity,
        transform: `scale(${scale})`,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: 16,
      }}
    >
      <div
        style={{
          fontSize: 18,
          fontWeight: 700,
          color: THEME.colors.accentBlue,
          fontFamily: fontFamilyBody,
          marginBottom: 12,
        }}
      >
        {label}
      </div>
      {/* Memory block */}
      <div
        style={{
          width: 160,
          height: 50,
          backgroundColor: `${THEME.colors.accentPurple}15`,
          border: `1px solid ${THEME.colors.accentPurple}60`,
          borderRadius: 6,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: 10,
        }}
      >
        <span
          style={{
            fontSize: 13,
            color: THEME.colors.accentPurple,
            fontFamily: fontFamilyCode,
          }}
        >
          Global Memory
        </span>
      </div>
      {/* SMs block */}
      <div
        style={{
          width: 160,
          height: 40,
          backgroundColor: `${THEME.colors.nvidiaGreen}15`,
          border: `1px solid ${THEME.colors.nvidiaGreen}60`,
          borderRadius: 6,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: 10,
        }}
      >
        <span
          style={{
            fontSize: 13,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyCode,
          }}
        >
          SMs
        </span>
      </div>
      {/* Copy Engine */}
      <div
        style={{
          width: 160,
          height: 36,
          backgroundColor: `${THEME.colors.accentOrange}15`,
          border: `1px solid ${THEME.colors.accentOrange}60`,
          borderRadius: 6,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <span
          style={{
            fontSize: 12,
            color: THEME.colors.accentOrange,
            fontFamily: fontFamilyCode,
          }}
        >
          Copy Engine
        </span>
      </div>
    </div>
  );
};

export const M6S10_MultiGPUIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const arrowProgress = interpolate(
    frame - 2.5 * fps,
    [0, 1 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const linkLabelOpacity = interpolate(
    frame - 3.5 * fps,
    [0, 0.4 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const bottomOpacity = interpolate(
    frame - 8 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={6}
      leftWidth="50%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle title="Multi-GPU Basics" />

          <div
            style={{
              position: "relative",
              height: 340,
              width: 500,
            }}
          >
            <GPUCard label="GPU 0" x={0} delay={0.8 * fps} />
            <GPUCard label="GPU 1" x={270} delay={1.2 * fps} />

            {/* Connection arrows */}
            <svg
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                width: 500,
                height: 340,
                pointerEvents: "none",
              }}
            >
              {/* Top arrow (GPU 0 -> GPU 1) */}
              <line
                x1={200}
                y1={130}
                x2={200 + 70 * arrowProgress}
                y2={130}
                stroke={THEME.colors.accentCyan}
                strokeWidth={2.5}
              />
              {arrowProgress > 0.9 && (
                <polygon
                  points="270,130 260,124 260,136"
                  fill={THEME.colors.accentCyan}
                  opacity={interpolate(arrowProgress, [0.9, 1], [0, 1])}
                />
              )}
              {/* Bottom arrow (GPU 1 -> GPU 0) */}
              <line
                x1={270}
                y1={180}
                x2={270 - 70 * arrowProgress}
                y2={180}
                stroke={THEME.colors.accentCyan}
                strokeWidth={2.5}
              />
              {arrowProgress > 0.9 && (
                <polygon
                  points="200,180 210,174 210,186"
                  fill={THEME.colors.accentCyan}
                  opacity={interpolate(arrowProgress, [0.9, 1], [0, 1])}
                />
              )}
            </svg>

            {/* Connection label */}
            <div
              style={{
                position: "absolute",
                left: 205,
                top: 140,
                width: 60,
                textAlign: "center",
                opacity: linkLabelOpacity,
              }}
            >
              <div
                style={{
                  fontSize: 12,
                  fontWeight: 700,
                  color: THEME.colors.accentCyan,
                  fontFamily: fontFamilyCode,
                  backgroundColor: `${THEME.colors.bgCard}`,
                  padding: "3px 8px",
                  borderRadius: 4,
                  border: `1px solid ${THEME.colors.accentCyan}40`,
                }}
              >
                PCIe
              </div>
              <div
                style={{
                  fontSize: 11,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyCode,
                  marginTop: 2,
                }}
              >
                NVLink
              </div>
            </div>
          </div>

          {/* Bottom use case box */}
          <div
            style={{
              marginTop: 8,
              padding: "12px 18px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.nvidiaGreen}30`,
              opacity: bottomOpacity,
              width: 500,
            }}
          >
            <div
              style={{
                fontSize: 15,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                lineHeight: 1.5,
              }}
            >
              Use case: model parallelism, data parallelism in distributed training
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ width: 520, marginTop: 80 }}>
          <BulletPoint
            text="cudaSetDevice(id) -- select which GPU to use"
            index={0}
            delay={1.5 * fps}
          />
          <BulletPoint
            text="Each GPU has its own memory space"
            index={1}
            delay={1.5 * fps}
          />
          <BulletPoint
            text="Peer-to-peer (P2P) access for direct GPU-to-GPU transfers"
            index={2}
            delay={1.5 * fps}
            highlight
          />
          <BulletPoint
            text="cudaMemcpyPeer for cross-GPU copies"
            index={3}
            delay={1.5 * fps}
          />
        </div>
      }
    />
  );
};
