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
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

type CodePanel = {
  title: string;
  language: string;
  lines: string[];
  color: string;
};

const panels: CodePanel[] = [
  {
    title: "CUDA Kernel (.cu)",
    language: "fused_bias_relu.cu",
    color: THEME.colors.nvidiaGreen,
    lines: [
      "__global__ void fused_bias_relu(",
      "    float* x, float* bias, float* out,",
      "    int N, int D) {",
      "  int idx = blockIdx.x * blockDim.x + threadIdx.x;",
      "  if (idx < N * D) {",
      "    float val = x[idx] + bias[idx % D];",
      "    out[idx] = val > 0.f ? val : 0.f;",
      "  }",
      "}",
    ],
  },
  {
    title: "C++ Binding (.cpp)",
    language: "binding.cpp",
    color: THEME.colors.accentBlue,
    lines: [
      "torch::Tensor fused_bias_relu_cuda(",
      "    torch::Tensor x, torch::Tensor bias) {",
      "  auto out = torch::empty_like(x);",
      "  AT_DISPATCH_FLOATING_TYPES(x.type(), \"fused\", ([&] {",
      "    fused_bias_relu<<<blocks, threads>>>(",
      "      x.data_ptr<float>(), bias.data_ptr<float>(),",
      "      out.data_ptr<float>(), N, D);",
      "  }));",
      "  return out;",
      "}",
      "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {",
      "  m.def(\"fused_bias_relu\", &fused_bias_relu_cuda);",
      "}",
    ],
  },
  {
    title: "Python Usage (.py)",
    language: "train.py",
    color: THEME.colors.accentPurple,
    lines: [
      "from torch.utils.cpp_extension import load",
      "",
      "ext = load(name='fused_ops',",
      "           sources=['fused_bias_relu.cu', 'binding.cpp'])",
      "",
      "out = ext.fused_bias_relu(hidden, bias)  # Custom op!",
    ],
  },
];

export const M10S11_PyTorchExtCode: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="code" moduleNumber={10}>
      <SlideTitle title="Building a CUDA Extension" />

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 12,
          flex: 1,
          width: 1776,
        }}
      >
        {panels.map((panel, panelIdx) => {
          const panelDelay = 0.8 * fps + panelIdx * 2.5 * fps;
          const panelSpring = spring({
            frame: frame - panelDelay,
            fps,
            config: { damping: 200 },
          });
          const panelOpacity = interpolate(panelSpring, [0, 1], [0, 1]);
          const panelY = interpolate(panelSpring, [0, 1], [20, 0]);

          return (
            <div
              key={panel.title}
              style={{
                opacity: panelOpacity,
                transform: `translateY(${panelY}px)`,
                backgroundColor: "#0d1117",
                borderRadius: 10,
                border: `1px solid ${panel.color}30`,
                overflow: "hidden",
                boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
              }}
            >
              {/* Panel header */}
              <div
                style={{
                  padding: "6px 16px",
                  backgroundColor: `${panel.color}10`,
                  borderBottom: `1px solid ${panel.color}20`,
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                }}
              >
                <div style={{ display: "flex", gap: 5 }}>
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
                    color: panel.color,
                    fontSize: 13,
                    fontWeight: 700,
                    fontFamily: fontFamilyBody,
                  }}
                >
                  {panel.title}
                </span>
                <span
                  style={{
                    color: THEME.colors.textMuted,
                    fontSize: 12,
                    fontFamily: fontFamilyCode,
                    marginLeft: "auto",
                  }}
                >
                  {panel.language}
                </span>
              </div>

              {/* Code content */}
              <div style={{ padding: "8px 16px" }}>
                {panel.lines.map((line, lineIdx) => {
                  const lineDelay = panelDelay + 0.04 * fps * lineIdx;
                  const lineOpacity = interpolate(
                    frame - lineDelay,
                    [0, 0.15 * fps],
                    [0, 1],
                    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                  );

                  return (
                    <div
                      key={lineIdx}
                      style={{
                        display: "flex",
                        opacity: lineOpacity,
                      }}
                    >
                      <span
                        style={{
                          color: THEME.colors.textMuted,
                          fontSize: 12,
                          fontFamily: fontFamilyCode,
                          width: 28,
                          textAlign: "right",
                          marginRight: 12,
                          flexShrink: 0,
                          userSelect: "none",
                          opacity: 0.5,
                        }}
                      >
                        {lineIdx + 1}
                      </span>
                      <pre
                        style={{
                          margin: 0,
                          fontFamily: fontFamilyCode,
                          fontSize: 14,
                          lineHeight: 1.5,
                          whiteSpace: "pre",
                          color: THEME.colors.textCode,
                        }}
                      >
                        {line}
                      </pre>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      {/* Bottom callout */}
      <div
        style={{
          marginTop: 10,
          padding: "12px 24px",
          backgroundColor: "rgba(118,185,0,0.08)",
          borderRadius: 10,
          border: `1px solid ${THEME.colors.nvidiaGreen}30`,
          width: 1776,
          textAlign: "center",
          opacity: interpolate(
            frame - 9 * fps,
            [0, 0.5 * fps],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          ),
        }}
      >
        <div
          style={{
            fontSize: 18,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyBody,
            fontWeight: 700,
          }}
        >
          Three files {"\u2192"} one fast custom operation
        </div>
      </div>
    </SlideLayout>
  );
};
