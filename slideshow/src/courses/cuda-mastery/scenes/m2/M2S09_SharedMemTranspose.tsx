import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

const TileGrid: React.FC<{
  frame: number;
  fps: number;
  delay: number;
  label: string;
  readDir: "row" | "col";
  writeDir: "row" | "col";
  readColor: string;
  writeColor: string;
}> = ({ frame, fps, delay, label, readDir, writeDir, readColor, writeColor }) => {
  const gridSpring = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });
  const gridOpacity = interpolate(gridSpring, [0, 1], [0, 1]);

  const tileSize = 4;
  const cellSize = 28;
  const gap = 2;

  const readAnim = interpolate(
    frame - delay - 0.5 * fps,
    [0, 0.8 * fps],
    [0, tileSize],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const writeAnim = interpolate(
    frame - delay - 1.5 * fps,
    [0, 0.8 * fps],
    [0, tileSize],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <div style={{ opacity: gridOpacity }}>
      <span style={{ fontSize: 13, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, display: "block", marginBottom: 6 }}>
        {label}
      </span>
      <div style={{ display: "flex", gap: 24 }}>
        {/* Read pattern */}
        <div>
          <span style={{ fontSize: 11, color: readColor, fontFamily: fontFamilyBody, fontWeight: 700, display: "block", marginBottom: 4 }}>
            READ
          </span>
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${tileSize}, ${cellSize}px)`, gap }}>
            {Array.from({ length: tileSize * tileSize }).map((_, idx) => {
              const r = Math.floor(idx / tileSize);
              const c = idx % tileSize;
              const isActive = readDir === "row" ? (c < Math.floor(readAnim) && r === 0) : (r < Math.floor(readAnim) && c === 0);
              return (
                <div
                  key={idx}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    borderRadius: 3,
                    backgroundColor: isActive ? `${readColor}40` : "rgba(255,255,255,0.04)",
                    border: `1px solid ${isActive ? readColor : "rgba(255,255,255,0.1)"}`,
                    fontSize: 9,
                    color: isActive ? readColor : THEME.colors.textMuted,
                    fontFamily: fontFamilyCode,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  {r},{c}
                </div>
              );
            })}
          </div>
        </div>

        {/* Write pattern */}
        <div>
          <span style={{ fontSize: 11, color: writeColor, fontFamily: fontFamilyBody, fontWeight: 700, display: "block", marginBottom: 4 }}>
            WRITE
          </span>
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${tileSize}, ${cellSize}px)`, gap }}>
            {Array.from({ length: tileSize * tileSize }).map((_, idx) => {
              const r = Math.floor(idx / tileSize);
              const c = idx % tileSize;
              const isActive = writeDir === "row" ? (c < Math.floor(writeAnim) && r === 0) : (r < Math.floor(writeAnim) && c === 0);
              return (
                <div
                  key={idx}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    borderRadius: 3,
                    backgroundColor: isActive ? `${writeColor}40` : "rgba(255,255,255,0.04)",
                    border: `1px solid ${isActive ? writeColor : "rgba(255,255,255,0.1)"}`,
                    fontSize: 9,
                    color: isActive ? writeColor : THEME.colors.textMuted,
                    fontFamily: fontFamilyCode,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  {c},{r}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export const M2S09_SharedMemTranspose: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const insightOpacity = interpolate(
    frame - 7.5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={2} slideNumber={9} totalSlides={18}>
      <SlideTitle
        title="Shared Memory — Matrix Transpose"
        subtitle="The classic example: fix uncoalesced writes with a shared memory tile"
      />

      <div style={{ display: "flex", gap: 32, flex: 1 }}>
        {/* Left: Naive transpose */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{
            display: "flex", alignItems: "center", gap: 8,
            opacity: interpolate(frame - 0.8 * fps, [0, 0.3 * fps], [0, 1], {
              extrapolateLeft: "clamp", extrapolateRight: "clamp",
            }),
          }}>
            <div style={{ width: 12, height: 12, borderRadius: 6, backgroundColor: THEME.colors.accentRed }} />
            <span style={{ fontSize: 18, fontWeight: 700, color: THEME.colors.accentRed, fontFamily: fontFamilyBody }}>
              Naive Transpose
            </span>
          </div>

          <CodeBlock
            delay={1 * fps}
            title="transpose_naive.cu"
            fontSize={13}
            code={`__global__ void transpose_naive(
    float *out, const float *in,
    int width, int height
) {
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // Read: row-major (coalesced)
    // Write: col-major (NOT coalesced!)
    out[x * height + y] = in[y * width + x];
}`}
            highlightLines={[8, 9, 10]}
          />

          <TileGrid
            frame={frame}
            fps={fps}
            delay={2.5 * fps}
            label="Read row-major, write column-major:"
            readDir="row"
            writeDir="col"
            readColor={THEME.colors.nvidiaGreen}
            writeColor={THEME.colors.accentRed}
          />

          <div style={{
            padding: "6px 12px",
            backgroundColor: "rgba(255,82,82,0.08)",
            borderRadius: 6,
            borderLeft: `3px solid ${THEME.colors.accentRed}`,
            opacity: interpolate(frame - 4 * fps, [0, 0.3 * fps], [0, 1], {
              extrapolateLeft: "clamp", extrapolateRight: "clamp",
            }),
          }}>
            <span style={{ fontSize: 13, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody }}>
              Writes are <span style={{ color: THEME.colors.accentRed, fontWeight: 700 }}>uncoalesced</span> - threads write to addresses stride apart.
            </span>
          </div>
        </div>

        {/* Right: Shared memory transpose */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{
            display: "flex", alignItems: "center", gap: 8,
            opacity: interpolate(frame - 3.5 * fps, [0, 0.3 * fps], [0, 1], {
              extrapolateLeft: "clamp", extrapolateRight: "clamp",
            }),
          }}>
            <div style={{ width: 12, height: 12, borderRadius: 6, backgroundColor: THEME.colors.nvidiaGreen }} />
            <span style={{ fontSize: 18, fontWeight: 700, color: THEME.colors.nvidiaGreen, fontFamily: fontFamilyBody }}>
              Shared Memory Transpose
            </span>
          </div>

          <CodeBlock
            delay={4 * fps}
            title="transpose_shared.cu"
            fontSize={13}
            code={`__global__ void transpose_shared(
    float *out, const float *in,
    int width, int height
) {
    __shared__ float tile[32][32];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // Coalesced read -> shared tile
    tile[threadIdx.y][threadIdx.x] =
        in[y * width + x];
    __syncthreads();

    // Swap block indices for output
    int ox = blockIdx.y * 32 + threadIdx.x;
    int oy = blockIdx.x * 32 + threadIdx.y;

    // Coalesced write from transposed tile
    out[oy * height + ox] =
        tile[threadIdx.x][threadIdx.y];
}`}
            highlightLines={[5, 10, 12, 19, 20]}
          />

          <TileGrid
            frame={frame}
            fps={fps}
            delay={5.5 * fps}
            label="Both read and write are coalesced:"
            readDir="row"
            writeDir="row"
            readColor={THEME.colors.nvidiaGreen}
            writeColor={THEME.colors.nvidiaGreen}
          />
        </div>
      </div>

      {/* Bottom insight */}
      <div style={{
        marginTop: 8,
        padding: "12px 20px",
        backgroundColor: "rgba(118,185,0,0.08)",
        borderRadius: 10,
        border: `2px solid ${THEME.colors.nvidiaGreen}40`,
        opacity: insightOpacity,
        textAlign: "center",
      }}>
        <span style={{ fontSize: 18, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody }}>
          Pattern: load with coalesced reads into shared tile, then write back with coalesced writes.{" "}
          <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>32x32 tiles</span> are the standard choice.
        </span>
      </div>
    </SlideLayout>
  );
};
