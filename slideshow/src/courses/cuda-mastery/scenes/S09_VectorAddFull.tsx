import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle } from "../../../components/AnimatedText";
import { CodeBlock } from "../../../components/CodeBlock";

export const S09_VectorAddFull: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <SlideLayout variant="code" slideNumber={9} totalSlides={18}>
      <SlideTitle
        title="Vector Add — Complete Host Code"
        subtitle="The full workflow: allocate, copy, launch, copy back, free"
      />

      <div style={{ display: "flex", gap: 32, flex: 1 }}>
        <div style={{ flex: 1 }}>
          <CodeBlock
            delay={0.5 * fps}
            title="02_vector_add.cu — Host code (left half)"
            fontSize={16}
            code={`// Step 1: Allocate HOST memory
float *h_a = (float *)malloc(bytes);
float *h_b = (float *)malloc(bytes);
float *h_c = (float *)malloc(bytes);

// Initialize data
for (int i = 0; i < N; i++) {
    h_a[i] = (float)i;
    h_b[i] = (float)(i * 2);
}

// Step 2: Allocate DEVICE memory
float *d_a, *d_b, *d_c;
cudaMalloc(&d_a, bytes);
cudaMalloc(&d_b, bytes);
cudaMalloc(&d_c, bytes);

// Step 3: Copy HOST -> DEVICE
cudaMemcpy(d_a, h_a, bytes,
           cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, bytes,
           cudaMemcpyHostToDevice);`}
            highlightLines={[1, 12, 18]}
          />
        </div>
        <div style={{ flex: 1 }}>
          <CodeBlock
            delay={1.5 * fps}
            title="02_vector_add.cu — Host code (right half)"
            fontSize={16}
            code={`// Step 4: Launch kernel
int blockSize = 256;
int numBlocks = (N + blockSize - 1)
              / blockSize;
vector_add<<<numBlocks, blockSize>>>(
    d_a, d_b, d_c, N
);

// Check for errors!
cudaGetLastError();

// Step 5: Copy DEVICE -> HOST
cudaMemcpy(h_c, d_c, bytes,
           cudaMemcpyDeviceToHost);

// Step 6: Verify results
for (int i = 0; i < N; i++) {
    assert(h_c[i] == h_a[i] + h_b[i]);
}

// Step 7: Free ALL memory
cudaFree(d_a); cudaFree(d_b);
cudaFree(d_c);
free(h_a); free(h_b); free(h_c);`}
            highlightLines={[1, 5, 9, 12, 21]}
          />
        </div>
      </div>
    </SlideLayout>
  );
};
