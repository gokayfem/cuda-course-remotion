import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
} from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../components/AnimatedText";
import { CodeBlock } from "../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../styles/fonts";

export const S14_GPUTiming: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="gradient" slideNumber={14} totalSlides={18}>
      <SlideTitle
        title="GPU Timing & Bandwidth"
        subtitle="Measure what matters — CUDA Events, not CPU timers"
      />

      <div style={{ display: "flex", gap: 40, flex: 1 }}>
        <div style={{ flex: 1 }}>
          <CodeBlock
            delay={0.5 * fps}
            title="CUDA Event timing"
            fontSize={17}
            code={`cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Always warmup first!
my_kernel<<<grid, block>>>(args);
cudaDeviceSynchronize();

// Timed run
cudaEventRecord(start);
my_kernel<<<grid, block>>>(args);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel took %.4f ms\\n", ms);

// Cleanup
cudaEventDestroy(start);
cudaEventDestroy(stop);`}
            highlightLines={[5, 6, 10, 12, 16]}
          />
        </div>

        <div style={{ flex: 0.8 }}>
          <FadeInText
            text="Why Not CPU Timers?"
            delay={3 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentRed}
            style={{ marginBottom: 12 }}
          />
          <FadeInText
            text="GPU operations are async — CPU timers measure the LAUNCH time, not execution time. CUDA Events record timestamps on the GPU itself."
            delay={3.3 * fps}
            fontSize={18}
            color={THEME.colors.textSecondary}
            style={{ marginBottom: 24 }}
          />

          <FadeInText
            text="Bandwidth Calculation"
            delay={5 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.nvidiaGreen}
            style={{ marginBottom: 12 }}
          />

          <div
            style={{
              padding: "16px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              marginBottom: 16,
              opacity: interpolate(
                frame - 5.5 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <div style={{ fontFamily: fontFamilyCode, fontSize: 17, color: THEME.colors.textPrimary, lineHeight: 2 }}>
              total_bytes = (2 reads + 1 write) * N * 4<br />
              BW = total_bytes / time<br />
              <span style={{ color: THEME.colors.nvidiaGreen }}>efficiency = BW / peak_BW</span>
            </div>
          </div>

          {/* Performance bars */}
          <FadeInText
            text="Typical Results (A100)"
            delay={7 * fps}
            fontSize={18}
            fontWeight={600}
            color={THEME.colors.textSecondary}
            style={{ marginBottom: 12 }}
          />

          {[
            { label: "Vector Add", bw: "1200 GB/s", pct: 74, color: THEME.colors.nvidiaGreen },
            { label: "PCIe H→D", bw: "25 GB/s", pct: 1.5, color: THEME.colors.accentOrange },
            { label: "PCIe D→H", bw: "25 GB/s", pct: 1.5, color: THEME.colors.accentRed },
          ].map((item, i) => {
            const barDelay = 7.5 * fps + i * 0.3 * fps;
            const barProgress = interpolate(
              frame - barDelay,
              [0, 0.5 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );
            return (
              <div key={item.label} style={{ marginBottom: 10, opacity: barProgress > 0 ? 1 : 0 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody }}>{item.label}</span>
                  <span style={{ fontSize: 14, color: item.color, fontFamily: fontFamilyCode, fontWeight: 700 }}>{item.bw}</span>
                </div>
                <div style={{ height: 12, backgroundColor: "rgba(255,255,255,0.05)", borderRadius: 6 }}>
                  <div
                    style={{
                      width: `${Math.max(item.pct, 2) * barProgress}%`,
                      height: "100%",
                      backgroundColor: item.color,
                      borderRadius: 6,
                    }}
                  />
                </div>
              </div>
            );
          })}

          <FadeInText
            text="PCIe is 50x slower! Minimize CPU↔GPU transfers."
            delay={9 * fps}
            fontSize={16}
            color={THEME.colors.accentRed}
            fontWeight={700}
            style={{ marginTop: 12 }}
          />
        </div>
      </div>
    </SlideLayout>
  );
};
