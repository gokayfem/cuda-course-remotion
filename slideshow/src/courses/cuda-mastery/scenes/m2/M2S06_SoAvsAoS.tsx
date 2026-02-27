import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

const MemoryBlock: React.FC<{
  label: string;
  color: string;
  opacity: number;
  width?: number;
}> = ({ label, color, opacity, width = 44 }) => (
  <div
    style={{
      width,
      height: 28,
      backgroundColor: `${color}30`,
      border: `1px solid ${color}80`,
      borderRadius: 3,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: 12,
      color,
      fontFamily: fontFamilyCode,
      fontWeight: 700,
      opacity,
      flexShrink: 0,
    }}
  >
    {label}
  </div>
);

export const M2S06_SoAvsAoS: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const fieldColors = {
    x: THEME.colors.accentRed,
    y: THEME.colors.accentBlue,
    z: THEME.colors.nvidiaGreen,
    w: THEME.colors.accentOrange,
  };

  const aosDiagramSpring = spring({
    frame: frame - 2 * fps,
    fps,
    config: { damping: 200 },
  });
  const aosOpacity = interpolate(aosDiagramSpring, [0, 1], [0, 1]);

  const soaDiagramSpring = spring({
    frame: frame - 4 * fps,
    fps,
    config: { damping: 200 },
  });
  const soaOpacity = interpolate(soaDiagramSpring, [0, 1], [0, 1]);

  const verdictOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const aosFields = ["x", "y", "z", "w"] as const;
  const particleCount = 6;

  return (
    <SlideLayout variant="gradient" moduleNumber={2} slideNumber={6} totalSlides={18}>
      <SlideTitle
        title="Struct of Arrays vs Array of Structs"
        subtitle="Data layout determines whether your memory access is coalesced"
      />

      <div style={{ display: "flex", gap: 40, flex: 1 }}>
        {/* Left: AoS */}
        <div style={{ flex: 1 }}>
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            marginBottom: 14,
            opacity: interpolate(frame - 1.2 * fps, [0, 0.3 * fps], [0, 1], {
              extrapolateLeft: "clamp", extrapolateRight: "clamp",
            }),
          }}>
            <div style={{
              padding: "4px 12px",
              backgroundColor: "rgba(255,82,82,0.15)",
              borderRadius: 6,
              fontSize: 15,
              fontWeight: 700,
              color: THEME.colors.accentRed,
              fontFamily: fontFamilyBody,
            }}>
              AoS — Array of Structs
            </div>
          </div>

          <CodeBlock
            delay={1.5 * fps}
            title="aos_layout.cu"
            fontSize={14}
            showLineNumbers={false}
            code={`struct Particle {
    float x, y, z, w;
};
Particle particles[N];

// Thread i reads particles[i].x
// Stride = sizeof(Particle) = 16 bytes!`}
          />

          {/* Memory diagram: interleaved */}
          <div style={{ marginTop: 14, opacity: aosOpacity }}>
            <span style={{ fontSize: 14, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginBottom: 6, display: "block" }}>
              Memory layout (interleaved):
            </span>
            <div style={{ display: "flex", gap: 2, flexWrap: "wrap", maxWidth: 580 }}>
              {Array.from({ length: particleCount }).flatMap((_, p) =>
                aosFields.map((f) => {
                  const cellDelay = 2.2 * fps + (p * 4 + aosFields.indexOf(f)) * 0.04 * fps;
                  const cellOpacity = interpolate(
                    frame - cellDelay,
                    [0, 0.2 * fps],
                    [0, 1],
                    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                  );
                  return (
                    <MemoryBlock
                      key={`${p}-${f}`}
                      label={`${f}${p}`}
                      color={fieldColors[f]}
                      opacity={cellOpacity}
                    />
                  );
                })
              )}
            </div>

            {/* Thread access arrows for .x */}
            <div style={{
              marginTop: 8,
              display: "flex",
              alignItems: "center",
              gap: 4,
              opacity: interpolate(frame - 3.5 * fps, [0, 0.3 * fps], [0, 1], {
                extrapolateLeft: "clamp", extrapolateRight: "clamp",
              }),
            }}>
              <span style={{ fontSize: 14, color: THEME.colors.accentRed, fontFamily: fontFamilyBody }}>
                Threads reading .x:
              </span>
              <span style={{ fontSize: 14, color: THEME.colors.accentRed, fontFamily: fontFamilyCode, fontWeight: 700 }}>
                stride=4 floats apart (BAD)
              </span>
            </div>
          </div>
        </div>

        {/* Right: SoA */}
        <div style={{ flex: 1 }}>
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            marginBottom: 14,
            opacity: interpolate(frame - 3.2 * fps, [0, 0.3 * fps], [0, 1], {
              extrapolateLeft: "clamp", extrapolateRight: "clamp",
            }),
          }}>
            <div style={{
              padding: "4px 12px",
              backgroundColor: "rgba(118,185,0,0.15)",
              borderRadius: 6,
              fontSize: 15,
              fontWeight: 700,
              color: THEME.colors.nvidiaGreen,
              fontFamily: fontFamilyBody,
            }}>
              SoA — Struct of Arrays
            </div>
          </div>

          <CodeBlock
            delay={3.5 * fps}
            title="soa_layout.cu"
            fontSize={14}
            showLineNumbers={false}
            code={`float *x, *y, *z, *w;
cudaMalloc(&x, N * sizeof(float));
cudaMalloc(&y, N * sizeof(float));
// ... etc

// Thread i reads x[i]
// Stride = 1 float = consecutive!`}
          />

          {/* Memory diagram: contiguous */}
          <div style={{ marginTop: 14, opacity: soaOpacity }}>
            <span style={{ fontSize: 14, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginBottom: 6, display: "block" }}>
              Memory layout (contiguous per field):
            </span>
            {aosFields.map((f) => (
              <div key={f} style={{ display: "flex", gap: 2, marginBottom: 3 }}>
                {Array.from({ length: particleCount }).map((_, p) => {
                  const cellDelay = 4.2 * fps + (aosFields.indexOf(f) * particleCount + p) * 0.04 * fps;
                  const cellOpacity = interpolate(
                    frame - cellDelay,
                    [0, 0.2 * fps],
                    [0, 1],
                    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                  );
                  return (
                    <MemoryBlock
                      key={`${f}-${p}`}
                      label={`${f}${p}`}
                      color={fieldColors[f]}
                      opacity={cellOpacity}
                    />
                  );
                })}
              </div>
            ))}

            <div style={{
              marginTop: 8,
              display: "flex",
              alignItems: "center",
              gap: 4,
              opacity: interpolate(frame - 5.5 * fps, [0, 0.3 * fps], [0, 1], {
                extrapolateLeft: "clamp", extrapolateRight: "clamp",
              }),
            }}>
              <span style={{ fontSize: 14, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyBody }}>
                Threads reading x:
              </span>
              <span style={{ fontSize: 14, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyCode, fontWeight: 700 }}>
                consecutive = COALESCED
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Verdict */}
      <div style={{
        marginTop: 12,
        padding: "14px 24px",
        backgroundColor: "rgba(118,185,0,0.08)",
        borderRadius: 10,
        border: `2px solid ${THEME.colors.nvidiaGreen}40`,
        opacity: verdictOpacity,
        textAlign: "center",
      }}>
        <span style={{ fontSize: 20, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody }}>
          Rule of thumb: <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>always prefer SoA on the GPU</span>.
          AoS is natural for CPUs, SoA is natural for GPUs.
        </span>
      </div>
    </SlideLayout>
  );
};
