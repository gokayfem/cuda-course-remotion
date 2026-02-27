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

interface CodeLine {
  readonly text: string;
  readonly color: string;
}

const CUDA_CODE: readonly CodeLine[] = [
  { text: "__global__ void add(", color: THEME.colors.syntaxKeyword },
  { text: "    float* x, float* y,", color: THEME.colors.textCode },
  { text: "    float* out, int N) {", color: THEME.colors.textCode },
  { text: "  int i = blockIdx.x *", color: THEME.colors.syntaxFunction },
  { text: "    blockDim.x + threadIdx.x;", color: THEME.colors.syntaxFunction },
  { text: "  if (i < N)", color: THEME.colors.syntaxKeyword },
  { text: "    out[i] = x[i] + y[i];", color: THEME.colors.textCode },
  { text: "}", color: THEME.colors.textCode },
  { text: "// + launch config,", color: THEME.colors.syntaxComment },
  { text: "//   error checking...", color: THEME.colors.syntaxComment },
];

const TRITON_CODE: readonly CodeLine[] = [
  { text: "@triton.jit", color: THEME.colors.syntaxFunction },
  { text: "def add(x_ptr, y_ptr,", color: THEME.colors.syntaxKeyword },
  { text: "        out_ptr, N,", color: THEME.colors.textCode },
  { text: "        BLOCK: tl.constexpr):", color: THEME.colors.syntaxType },
  { text: "  pid = tl.program_id(0)", color: THEME.colors.syntaxFunction },
  { text: "  offs = pid * BLOCK +", color: THEME.colors.textCode },
  { text: "    tl.arange(0, BLOCK)", color: THEME.colors.syntaxFunction },
  { text: "  mask = offs < N", color: THEME.colors.syntaxKeyword },
  { text: "  x = tl.load(", color: THEME.colors.syntaxFunction },
  { text: "    x_ptr + offs, mask=mask)", color: THEME.colors.textCode },
  { text: "  y = tl.load(", color: THEME.colors.syntaxFunction },
  { text: "    y_ptr + offs, mask=mask)", color: THEME.colors.textCode },
  { text: "  tl.store(out_ptr + offs,", color: THEME.colors.syntaxFunction },
  { text: "    x + y, mask=mask)", color: THEME.colors.textCode },
];

interface CodePanelProps {
  readonly title: string;
  readonly titleColor: string;
  readonly lines: readonly CodeLine[];
  readonly delay: number;
  readonly frame: number;
  readonly fps: number;
}

const CodePanel: React.FC<CodePanelProps> = ({ title, titleColor, lines, delay, frame: f, fps: fp }) => {
  const panelSpring = spring({
    frame: f - delay,
    fps: fp,
    config: { damping: 200 },
  });
  const panelOpacity = interpolate(panelSpring, [0, 1], [0, 1]);

  return (
    <div
      style={{
        flex: 1,
        backgroundColor: "#0d1117",
        borderRadius: 8,
        border: "1px solid rgba(255,255,255,0.08)",
        overflow: "hidden",
        opacity: panelOpacity,
      }}
    >
      <div
        style={{
          padding: "6px 12px",
          backgroundColor: "rgba(255,255,255,0.04)",
          borderBottom: "1px solid rgba(255,255,255,0.08)",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <div style={{ display: "flex", gap: 4 }}>
          {["#ff5f57", "#ffbd2e", "#28c840"].map((c) => (
            <div
              key={c}
              style={{
                width: 8,
                height: 8,
                borderRadius: 4,
                backgroundColor: c,
              }}
            />
          ))}
        </div>
        <span
          style={{
            fontSize: 12,
            fontWeight: 700,
            color: titleColor,
            fontFamily: fontFamilyCode,
          }}
        >
          {title}
        </span>
      </div>
      <div style={{ padding: "10px 14px" }}>
        {lines.map((line, i) => {
          const lineDelay = delay + i * 0.04 * fp;
          const lineOpacity = interpolate(
            f - lineDelay,
            [0, 0.15 * fp],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );

          return (
            <div
              key={i}
              style={{
                opacity: lineOpacity,
                fontFamily: fontFamilyCode,
                fontSize: 13,
                lineHeight: 1.5,
                color: line.color,
                whiteSpace: "pre",
              }}
            >
              {line.text}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const M10S08_TritonIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const vsDelay = 2 * fps;
  const vsOpacity = interpolate(
    frame - vsDelay,
    [0, 0.4 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      variant="code"
      moduleNumber={10}
      leftWidth="55%"
      left={
        <div style={{ width: 860 }}>
          <SlideTitle
            title="Triton -- GPU Programming in Python"
            subtitle="Write high-performance GPU kernels without C/C++"
          />

          {/* Side by side code comparison */}
          <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
            <CodePanel
              title="CUDA C (verbose)"
              titleColor={THEME.colors.accentRed}
              lines={CUDA_CODE}
              delay={0.8 * fps}
              frame={frame}
              fps={fps}
            />

            {/* VS divider */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                width: 36,
                flexShrink: 0,
                opacity: vsOpacity,
              }}
            >
              <span
                style={{
                  fontSize: 16,
                  fontWeight: 800,
                  color: THEME.colors.textMuted,
                  fontFamily: fontFamilyBody,
                }}
              >
                vs
              </span>
            </div>

            <CodePanel
              title="Triton Python (concise)"
              titleColor={THEME.colors.nvidiaGreen}
              lines={TRITON_CODE}
              delay={1.5 * fps}
              frame={frame}
              fps={fps}
            />
          </div>
        </div>
      }
      right={
        <div style={{ width: 540, marginTop: 80 }}>
          <BulletPoint
            text="Write GPU kernels in Python"
            index={0}
            delay={4 * fps}
            highlight
            subtext="No CUDA C, no compilation step, rapid iteration"
          />
          <BulletPoint
            text="Block-based (not thread-based) programming"
            index={1}
            delay={4 * fps}
            subtext="tl.arange, tl.load, tl.store operate on blocks of data"
          />
          <BulletPoint
            text='Built-in auto-tuning (@triton.autotune)'
            index={2}
            delay={4 * fps}
            highlight
            subtext="Automatically searches for best BLOCK_SIZE, num_warps, etc."
          />
          <BulletPoint
            text="torch.compile backend uses Triton"
            index={3}
            delay={4 * fps}
            subtext="PyTorch 2.0+ generates Triton kernels automatically"
          />
        </div>
      }
    />
  );
};
