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

export const M2S12_Registers: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const tableRows = [
    { regs: 32, threads: 2048, occupancy: "100%", color: THEME.colors.nvidiaGreen },
    { regs: 64, threads: 1024, occupancy: "50%", color: THEME.colors.accentYellow },
    { regs: 128, threads: 512, occupancy: "25%", color: THEME.colors.accentOrange },
    { regs: 255, threads: 256, occupancy: "12.5%", color: THEME.colors.accentRed },
  ];

  const tableOpacity = interpolate(
    frame - 4 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const tradeoffOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={2} slideNumber={12} totalSlides={18}>
      <SlideTitle
        title="Registers — The Fastest Memory"
        subtitle="~1 cycle latency, private to each thread, compiler-managed"
      />

      <div style={{ display: "flex", gap: 36, flex: 1 }}>
        {/* Left: Properties and code */}
        <div style={{ flex: 1 }}>
          <BulletPoint index={0} delay={0.8 * fps} text="~1 cycle latency (fastest memory on GPU)" icon="1" highlight />
          <BulletPoint index={1} delay={0.8 * fps} text="Private to each thread — not shared" icon="2" />
          <BulletPoint index={2} delay={0.8 * fps} text="255 registers max per thread" icon="3" />
          <BulletPoint index={3} delay={0.8 * fps} text="65,536 registers per SM (shared across threads)" icon="4" />

          <div style={{ marginTop: 16 }}>
            <CodeBlock
              delay={2.5 * fps}
              title="register_usage.cu"
              fontSize={16}
              code={`// These automatically go to registers:
__global__ void kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx];   // local var -> register
    float temp = val * 2.0f; // local var -> register
    float result = temp + 1; // local var -> register
    data[idx] = result;
}
// 3 float registers + 1 int register used`}
              highlightLines={[4, 5, 6]}
            />
          </div>
        </div>

        {/* Right: Occupancy table and tradeoff */}
        <div style={{ flex: 0.85 }}>
          <FadeInText
            text="Register Pressure vs Occupancy"
            delay={3.5 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 16 }}
          />

          {/* Occupancy table */}
          <div style={{ opacity: tableOpacity }}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr 1fr",
                gap: 1,
                backgroundColor: "rgba(255,255,255,0.05)",
                borderRadius: 8,
                overflow: "hidden",
                marginBottom: 8,
              }}
            >
              {["Regs/Thread", "Max Threads/SM", "Occupancy"].map((header) => (
                <div
                  key={header}
                  style={{
                    padding: "10px 14px",
                    backgroundColor: "rgba(255,255,255,0.08)",
                    fontSize: 14,
                    fontWeight: 700,
                    color: THEME.colors.textMuted,
                    fontFamily: fontFamilyBody,
                    textAlign: "center",
                  }}
                >
                  {header}
                </div>
              ))}
            </div>

            {tableRows.map((row, i) => {
              const rowDelay = 4.5 * fps + i * 0.3 * fps;
              const rowSpring = spring({ frame: frame - rowDelay, fps, config: { damping: 200 } });
              const rowOpacity = interpolate(rowSpring, [0, 1], [0, 1]);

              return (
                <div
                  key={row.regs}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr 1fr",
                    gap: 1,
                    marginBottom: 2,
                    opacity: rowOpacity,
                  }}
                >
                  <div style={{ padding: "10px 14px", backgroundColor: `${row.color}10`, textAlign: "center", fontSize: 16, color: row.color, fontFamily: fontFamilyCode, fontWeight: 600, borderRadius: "4px 0 0 4px" }}>
                    {row.regs}
                  </div>
                  <div style={{ padding: "10px 14px", backgroundColor: `${row.color}08`, textAlign: "center", fontSize: 16, color: THEME.colors.textPrimary, fontFamily: fontFamilyCode }}>
                    {row.threads}
                  </div>
                  <div style={{ padding: "10px 14px", backgroundColor: `${row.color}10`, textAlign: "center", fontSize: 16, color: row.color, fontFamily: fontFamilyCode, fontWeight: 700, borderRadius: "0 4px 4px 0" }}>
                    {row.occupancy}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Tradeoff insight */}
          <div
            style={{
              marginTop: 20,
              padding: "14px 20px",
              backgroundColor: "rgba(255,171,64,0.08)",
              borderRadius: 10,
              borderLeft: `4px solid ${THEME.colors.accentOrange}`,
              opacity: tradeoffOpacity,
            }}
          >
            <span style={{ fontSize: 17, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.5 }}>
              <span style={{ color: THEME.colors.accentOrange, fontWeight: 700 }}>Tradeoff:</span>{" "}
              More registers per thread = fewer concurrent threads = less latency hiding.
              Find the sweet spot with <span style={{ fontFamily: fontFamilyCode, color: THEME.colors.accentCyan }}>--ptxas-options=-v</span>
            </span>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
