import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../components/AnimatedText";
import { CodeBlock } from "../../../components/CodeBlock";
import { fontFamilyBody } from "../../../styles/fonts";

export const S06_FirstKernel: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="code" slideNumber={6} totalSlides={18}>
      <SlideTitle
        title="Your First CUDA Kernel"
        subtitle="Hello World from the GPU — every thread prints its ID"
      />

      <div style={{ display: "flex", gap: 40, flex: 1 }}>
        {/* Code */}
        <div style={{ flex: 1.2 }}>
          <CodeBlock
            delay={0.5 * fps}
            title="01_hello_gpu.cu"
            fontSize={19}
            code={`#include <stdio.h>

// __global__ = runs on GPU, called from CPU
__global__ void hello_kernel() {
    printf("Hello from Block %d, Thread %d\\n",
           blockIdx.x, threadIdx.x);
}

int main() {
    // Launch: 2 blocks x 4 threads = 8 threads
    hello_kernel<<<2, 4>>>();

    // WAIT for GPU to finish!
    cudaDeviceSynchronize();

    return 0;
}`}
            highlightLines={[3, 4, 11, 14]}
          />
        </div>

        {/* Annotations */}
        <div style={{ flex: 0.8 }}>
          <FadeInText
            text="Step by Step"
            delay={2 * fps}
            fontSize={24}
            fontWeight={700}
            color={THEME.colors.nvidiaGreen}
            style={{ marginBottom: 20 }}
          />

          {/* Annotation cards */}
          {[
            {
              line: "Line 3-4",
              text: "__global__ marks this as a GPU kernel. It returns void — kernels can't return values.",
              color: THEME.colors.syntaxKeyword,
            },
            {
              line: "Line 5-6",
              text: "blockIdx.x and threadIdx.x are built-in variables. Every thread knows its own position.",
              color: THEME.colors.syntaxFunction,
            },
            {
              line: "Line 11",
              text: "<<<2, 4>>> = launch config. 2 blocks, 4 threads each. This creates 8 parallel threads.",
              color: THEME.colors.accentOrange,
            },
            {
              line: "Line 14",
              text: "GPU runs async! Without this, main() exits before GPU prints. Always sync!",
              color: THEME.colors.accentRed,
            },
          ].map((note, i) => {
            const noteDelay = 2.5 * fps + i * 0.5 * fps;
            const noteOpacity = interpolate(
              frame - noteDelay,
              [0, 0.3 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );

            return (
              <div
                key={i}
                style={{
                  marginBottom: 16,
                  padding: "12px 16px",
                  backgroundColor: `${note.color}10`,
                  borderLeft: `3px solid ${note.color}`,
                  borderRadius: 6,
                  opacity: noteOpacity,
                }}
              >
                <div
                  style={{
                    fontSize: 14,
                    fontWeight: 700,
                    color: note.color,
                    fontFamily: fontFamilyBody,
                    marginBottom: 4,
                  }}
                >
                  {note.line}
                </div>
                <div
                  style={{
                    fontSize: 16,
                    color: THEME.colors.textSecondary,
                    fontFamily: fontFamilyBody,
                    lineHeight: 1.4,
                  }}
                >
                  {note.text}
                </div>
              </div>
            );
          })}

          {/* Compile command */}
          <div
            style={{
              marginTop: 16,
              padding: "10px 16px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 8,
              opacity: interpolate(
                frame - 5 * fps,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              ),
            }}
          >
            <span style={{ fontSize: 15, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyBody, fontWeight: 600 }}>
              Compile: nvcc -o hello hello_gpu.cu && ./hello
            </span>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
