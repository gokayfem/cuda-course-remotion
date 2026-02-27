import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, BulletPoint, FadeInText } from "../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../styles/fonts";

export const S02_WhyGPU: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="gradient" slideNumber={2} totalSlides={18}>
      <SlideTitle
        title="Why GPUs for Machine Learning?"
        subtitle="Understanding the hardware that powers modern AI"
      />

      <div style={{ display: "flex", gap: 48, flex: 1 }}>
        {/* Left: The Problem */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text="The Problem"
            delay={0.8 * fps}
            fontSize={28}
            fontWeight={700}
            color={THEME.colors.accentOrange}
            style={{ marginBottom: 16 }}
          />
          <BulletPoint
            index={0}
            delay={1 * fps}
            text="Matrix multiply in GPT-4: trillions of FLOPs per forward pass"
          />
          <BulletPoint
            index={1}
            delay={1 * fps}
            text="Training data: billions of tokens, each needs gradient computation"
          />
          <BulletPoint
            index={2}
            delay={1 * fps}
            text="Single CPU core: ~100 GFLOPS (FP32)"
          />
          <BulletPoint
            index={3}
            delay={1 * fps}
            text="You'd need YEARS to train a single model"
            highlight
          />
        </div>

        {/* Right: The Solution */}
        <div style={{ flex: 1 }}>
          <FadeInText
            text="The GPU Solution"
            delay={3 * fps}
            fontSize={28}
            fontWeight={700}
            color={THEME.colors.nvidiaGreen}
            style={{ marginBottom: 16 }}
          />
          <BulletPoint
            index={0}
            delay={3.2 * fps}
            text="NVIDIA H100: 989 TFLOPS (FP32)"
            icon="⚡"
            highlight
          />
          <BulletPoint
            index={1}
            delay={3.2 * fps}
            text="That's ~10,000x a single CPU core"
            icon="⚡"
          />
          <BulletPoint
            index={2}
            delay={3.2 * fps}
            text="3.35 TB/s memory bandwidth (HBM3)"
            icon="⚡"
          />
          <BulletPoint
            index={3}
            delay={3.2 * fps}
            text="16,896 CUDA cores working in parallel"
            icon="⚡"
          />

          {/* Speed comparison bar */}
          {(() => {
            const barDelay = 5 * fps;
            const barProgress = interpolate(
              frame - barDelay,
              [0, 1 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );
            const barOpacity = interpolate(
              frame - barDelay,
              [0, 0.3 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );

            return (
              <div style={{ marginTop: 32, opacity: barOpacity }}>
                <div
                  style={{
                    fontSize: 16,
                    color: THEME.colors.textMuted,
                    fontFamily: fontFamilyBody,
                    marginBottom: 8,
                  }}
                >
                  Relative Performance (FP32 TFLOPS)
                </div>
                {/* CPU bar */}
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
                  <span style={{ width: 50, fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyCode }}>CPU</span>
                  <div style={{ flex: 1, height: 24, backgroundColor: "rgba(255,255,255,0.05)", borderRadius: 4 }}>
                    <div
                      style={{
                        width: `${barProgress * 1}%`,
                        height: "100%",
                        backgroundColor: THEME.colors.accentOrange,
                        borderRadius: 4,
                        minWidth: barProgress > 0 ? 4 : 0,
                      }}
                    />
                  </div>
                  <span style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyCode, width: 60 }}>0.1</span>
                </div>
                {/* GPU bar */}
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <span style={{ width: 50, fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyCode }}>GPU</span>
                  <div style={{ flex: 1, height: 24, backgroundColor: "rgba(255,255,255,0.05)", borderRadius: 4 }}>
                    <div
                      style={{
                        width: `${barProgress * 100}%`,
                        height: "100%",
                        background: `linear-gradient(90deg, ${THEME.colors.nvidiaGreen}, ${THEME.colors.nvidiaGreenLight})`,
                        borderRadius: 4,
                      }}
                    />
                  </div>
                  <span style={{ fontSize: 14, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyCode, fontWeight: 700, width: 60 }}>989</span>
                </div>
              </div>
            );
          })()}
        </div>
      </div>
    </SlideLayout>
  );
};
