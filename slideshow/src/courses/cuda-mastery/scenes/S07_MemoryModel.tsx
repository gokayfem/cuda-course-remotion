import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, FadeInText, BulletPoint } from "../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../styles/fonts";

export const S07_MemoryModel: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Arrow animation
  const arrowProgress = interpolate(
    frame - 3 * fps,
    [0, 1 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" slideNumber={7} totalSlides={18}>
      <SlideTitle
        title="The CUDA Memory Model"
        subtitle="CPU and GPU have SEPARATE memory spaces — you must copy data explicitly"
      />

      <div style={{ display: "flex", gap: 40, flex: 1 }}>
        {/* Diagram */}
        <div style={{ flex: 1.2, position: "relative" }}>
          {/* CPU Box */}
          <div
            style={{
              position: "absolute",
              left: 20,
              top: 20,
              width: 260,
              height: 320,
              border: `2px solid ${THEME.colors.accentOrange}`,
              borderRadius: 16,
              backgroundColor: `${THEME.colors.accentOrange}08`,
              opacity: interpolate(
                frame - 0.5 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
              padding: 20,
            }}
          >
            <div style={{ fontSize: 22, fontWeight: 700, color: THEME.colors.accentOrange, fontFamily: fontFamilyBody, marginBottom: 12 }}>
              CPU (Host)
            </div>
            <div style={{ fontSize: 16, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, marginBottom: 20 }}>
              System RAM
            </div>
            {/* Data blocks */}
            {["float *h_a", "float *h_b", "float *h_c"].map((label, i) => (
              <div
                key={label}
                style={{
                  padding: "10px 14px",
                  backgroundColor: `${THEME.colors.accentOrange}15`,
                  border: `1px solid ${THEME.colors.accentOrange}40`,
                  borderRadius: 6,
                  marginBottom: 8,
                  fontSize: 15,
                  color: THEME.colors.accentOrange,
                  fontFamily: fontFamilyCode,
                  opacity: interpolate(
                    frame - (1 + i * 0.2) * fps,
                    [0, 0.3 * fps],
                    [0, 1],
                    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                  ),
                }}
              >
                {label}
              </div>
            ))}
            <div style={{ fontSize: 14, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginTop: 8 }}>
              malloc() / free()
            </div>
          </div>

          {/* Arrow: CPU -> GPU */}
          <svg
            style={{ position: "absolute", left: 280, top: 60, width: 200, height: 100 }}
          >
            <defs>
              <marker id="arrow1" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill={THEME.colors.nvidiaGreen} />
              </marker>
            </defs>
            <line
              x1="0" y1="30" x2={180 * arrowProgress} y2="30"
              stroke={THEME.colors.nvidiaGreen}
              strokeWidth="3"
              markerEnd={arrowProgress > 0.9 ? "url(#arrow1)" : undefined}
            />
            <text x="60" y="20" fill={THEME.colors.nvidiaGreen} fontSize="14" fontFamily={fontFamilyCode}>
              {arrowProgress > 0.3 ? "cudaMemcpy(H→D)" : ""}
            </text>
          </svg>

          {/* Arrow: GPU -> CPU */}
          <svg
            style={{ position: "absolute", left: 280, top: 180, width: 200, height: 100 }}
          >
            <defs>
              <marker id="arrow2" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto-start-reverse">
                <polygon points="0 0, 10 3.5, 0 7" fill={THEME.colors.accentBlue} />
              </marker>
            </defs>
            <line
              x1={180} y1="30" x2={180 - 180 * arrowProgress} y2="30"
              stroke={THEME.colors.accentBlue}
              strokeWidth="3"
              markerEnd={arrowProgress > 0.9 ? "url(#arrow2)" : undefined}
            />
            <text x="50" y="60" fill={THEME.colors.accentBlue} fontSize="14" fontFamily={fontFamilyCode}>
              {arrowProgress > 0.3 ? "cudaMemcpy(D→H)" : ""}
            </text>
          </svg>

          {/* GPU Box */}
          <div
            style={{
              position: "absolute",
              right: 20,
              top: 20,
              width: 260,
              height: 320,
              border: `2px solid ${THEME.colors.nvidiaGreen}`,
              borderRadius: 16,
              backgroundColor: `${THEME.colors.nvidiaGreen}08`,
              opacity: interpolate(
                frame - 0.5 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
              padding: 20,
            }}
          >
            <div style={{ fontSize: 22, fontWeight: 700, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyBody, marginBottom: 12 }}>
              GPU (Device)
            </div>
            <div style={{ fontSize: 16, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody, marginBottom: 20 }}>
              VRAM / HBM
            </div>
            {["float *d_a", "float *d_b", "float *d_c"].map((label, i) => (
              <div
                key={label}
                style={{
                  padding: "10px 14px",
                  backgroundColor: `${THEME.colors.nvidiaGreen}15`,
                  border: `1px solid ${THEME.colors.nvidiaGreen}40`,
                  borderRadius: 6,
                  marginBottom: 8,
                  fontSize: 15,
                  color: THEME.colors.nvidiaGreen,
                  fontFamily: fontFamilyCode,
                  opacity: interpolate(
                    frame - (1 + i * 0.2) * fps,
                    [0, 0.3 * fps],
                    [0, 1],
                    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                  ),
                }}
              >
                {label}
              </div>
            ))}
            <div style={{ fontSize: 14, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginTop: 8 }}>
              cudaMalloc() / cudaFree()
            </div>
          </div>
        </div>

        {/* Right: Workflow */}
        <div style={{ flex: 0.8 }}>
          <FadeInText
            text="The 5-Step Workflow"
            delay={4 * fps}
            fontSize={24}
            fontWeight={700}
            color={THEME.colors.nvidiaGreen}
            style={{ marginBottom: 16 }}
          />

          {[
            { step: "1", text: "cudaMalloc — allocate GPU memory", icon: "1" },
            { step: "2", text: "cudaMemcpy H→D — copy data to GPU", icon: "2" },
            { step: "3", text: "kernel<<<...>>>() — run on GPU", icon: "3" },
            { step: "4", text: "cudaMemcpy D→H — copy results back", icon: "4" },
            { step: "5", text: "cudaFree — release GPU memory", icon: "5" },
          ].map(({ text, icon }, i) => (
            <BulletPoint
              key={i}
              index={i}
              delay={4.5 * fps}
              text={text}
              icon={icon}
              highlight={i === 2}
            />
          ))}

          <div
            style={{
              marginTop: 20,
              padding: "12px 16px",
              backgroundColor: "rgba(255,82,82,0.08)",
              borderRadius: 8,
              borderLeft: `3px solid ${THEME.colors.accentRed}`,
              opacity: interpolate(
                frame - 7 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <span style={{ fontSize: 16, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody }}>
              Key insight: Data transfer over PCIe is <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>10-50x slower</span> than GPU memory bandwidth. Minimize transfers!
            </span>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
