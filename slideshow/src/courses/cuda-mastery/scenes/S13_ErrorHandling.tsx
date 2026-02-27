import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, FadeInText, BulletPoint } from "../../../components/AnimatedText";
import { CodeBlock } from "../../../components/CodeBlock";

export const S13_ErrorHandling: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="code" slideNumber={13} totalSlides={18}>
      <SlideTitle
        title="Error Handling — Non-Negotiable"
        subtitle="CUDA errors are DEFERRED — if you don't check, bugs are invisible"
      />

      <div style={{ display: "flex", gap: 40, flex: 1 }}>
        <div style={{ flex: 1 }}>
          <CodeBlock
            delay={0.5 * fps}
            title="CUDA_CHECK macro — use this EVERYWHERE"
            fontSize={17}
            code={`// The essential error-checking macro
#define CUDA_CHECK(call)                   \\
  do {                                     \\
    cudaError_t err = call;                \\
    if (err != cudaSuccess) {              \\
      fprintf(stderr,                      \\
        "CUDA error at %s:%d: %s\\n",      \\
        __FILE__, __LINE__,                \\
        cudaGetErrorString(err));           \\
      exit(EXIT_FAILURE);                  \\
    }                                      \\
  } while (0)

// Usage:
CUDA_CHECK(cudaMalloc(&d_a, bytes));
CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes,
           cudaMemcpyHostToDevice));

// After kernel launch:
my_kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());`}
            highlightLines={[1, 2, 15, 20, 21]}
          />
        </div>

        <div style={{ flex: 0.7 }}>
          <FadeInText
            text="Why Errors Are Tricky"
            delay={3 * fps}
            fontSize={24}
            fontWeight={700}
            color={THEME.colors.accentRed}
            style={{ marginBottom: 16 }}
          />

          <BulletPoint
            index={0}
            delay={3.5 * fps}
            text="Kernel launches don't return errors"
            subtext="The GPU just queues work and the CPU moves on"
          />
          <BulletPoint
            index={1}
            delay={3.5 * fps}
            text="Errors surface LATER"
            subtext="At the next sync point or API call"
            highlight
          />
          <BulletPoint
            index={2}
            delay={3.5 * fps}
            text="Two checkpoints needed:"
            subtext="cudaGetLastError() → was the launch valid? cudaDeviceSynchronize() → did execution succeed?"
          />

          <FadeInText
            text="Common Errors"
            delay={6 * fps}
            fontSize={22}
            fontWeight={700}
            color={THEME.colors.accentOrange}
            style={{ marginBottom: 12, marginTop: 24 }}
          />

          <BulletPoint
            index={0}
            delay={6.5 * fps}
            text="Invalid launch config (>1024 threads)"
            icon="!"
          />
          <BulletPoint
            index={1}
            delay={6.5 * fps}
            text="Out of memory (cudaMalloc fail)"
            icon="!"
          />
          <BulletPoint
            index={2}
            delay={6.5 * fps}
            text="Illegal memory access in kernel"
            icon="!"
          />
          <BulletPoint
            index={3}
            delay={6.5 * fps}
            text="Missing bounds check (silent corruption!)"
            icon="!"
            highlight
          />
        </div>
      </div>
    </SlideLayout>
  );
};
