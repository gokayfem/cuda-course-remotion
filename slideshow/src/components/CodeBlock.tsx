import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../styles/theme";
import { fontFamilyCode } from "../styles/fonts";

type HighlightedToken = {
  text: string;
  color: string;
};

const highlightCudaLine = (line: string): HighlightedToken[] => {
  const tokens: HighlightedToken[] = [];
  let remaining = line;

  const patterns: Array<{ regex: RegExp; color: string }> = [
    { regex: /^(\/\/.*$)/, color: THEME.colors.syntaxComment },
    { regex: /^(\/\*[\s\S]*?\*\/)/, color: THEME.colors.syntaxComment },
    { regex: /^("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/, color: THEME.colors.syntaxString },
    {
      regex: /^(\b(?:__global__|__device__|__host__|__shared__|__constant__|__launch_bounds__|void|int|float|double|char|const|return|if|else|for|while|do|switch|case|break|continue|struct|typedef|sizeof|static|extern|unsigned|long|short|size_t|dim3)\b)/,
      color: THEME.colors.syntaxKeyword,
    },
    {
      regex: /^(\b(?:cudaMalloc|cudaFree|cudaMemcpy|cudaMemcpyToSymbol|cudaDeviceSynchronize|cudaGetLastError|cudaEventCreate|cudaEventRecord|cudaEventSynchronize|cudaEventElapsedTime|cudaEventDestroy|cudaGetDeviceProperties|cudaMemcpyHostToDevice|cudaMemcpyDeviceToHost|cudaOccupancyMaxPotentialBlockSize|printf|malloc|free|blockIdx|threadIdx|blockDim|gridDim|fmaxf|fminf|__syncthreads|atomicAdd)\b)/,
      color: THEME.colors.syntaxFunction,
    },
    { regex: /^(\b\d+\.?\d*f?\b)/, color: THEME.colors.syntaxNumber },
    { regex: /^(<<<|>>>)/, color: THEME.colors.accentOrange },
    { regex: /^([+\-*/%=<>!&|^~?:]+)/, color: THEME.colors.syntaxOperator },
    { regex: /^(#\w+)/, color: THEME.colors.syntaxKeyword },
  ];

  while (remaining.length > 0) {
    let matched = false;

    const wsMatch = remaining.match(/^(\s+)/);
    if (wsMatch) {
      tokens.push({ text: wsMatch[1], color: THEME.colors.textCode });
      remaining = remaining.slice(wsMatch[1].length);
      continue;
    }

    for (const { regex, color } of patterns) {
      const match = remaining.match(regex);
      if (match) {
        tokens.push({ text: match[1], color });
        remaining = remaining.slice(match[1].length);
        matched = true;
        break;
      }
    }

    if (!matched) {
      tokens.push({ text: remaining[0], color: THEME.colors.textCode });
      remaining = remaining.slice(1);
    }
  }

  return tokens;
};

export const CodeBlock: React.FC<{
  code: string;
  title?: string;
  delay?: number;
  highlightLines?: number[];
  fontSize?: number;
  showLineNumbers?: boolean;
  animateLines?: boolean;
}> = ({
  code,
  title,
  delay = 0,
  highlightLines = [],
  fontSize = 18,
  showLineNumbers = true,
  animateLines = true,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const lines = code.split("\n");

  const containerSpring = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });

  const containerOpacity = interpolate(containerSpring, [0, 1], [0, 1]);
  const containerScale = interpolate(containerSpring, [0, 1], [0.98, 1]);

  return (
    <div
      style={{
        opacity: containerOpacity,
        transform: `scale(${containerScale})`,
        backgroundColor: "#0d1117",
        borderRadius: 10,
        border: `1px solid rgba(255,255,255,0.08)`,
        overflow: "hidden",
        boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
      }}
    >
      {/* Title bar */}
      {title && (
        <div
          style={{
            padding: "8px 18px",
            backgroundColor: "rgba(255,255,255,0.04)",
            borderBottom: "1px solid rgba(255,255,255,0.08)",
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
        >
          <div style={{ display: "flex", gap: 6 }}>
            {["#ff5f57", "#ffbd2e", "#28c840"].map((c) => (
              <div
                key={c}
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: 5,
                  backgroundColor: c,
                }}
              />
            ))}
          </div>
          <span
            style={{
              color: THEME.colors.textMuted,
              fontSize: 13,
              fontFamily: fontFamilyCode,
              marginLeft: 6,
              letterSpacing: "0.2px",
            }}
          >
            {title}
          </span>
        </div>
      )}

      {/* Code content */}
      <div style={{ padding: "14px 18px", overflowX: "hidden" }}>
        {lines.map((line, i) => {
          const lineDelay = animateLines
            ? delay + 0.04 * fps * i
            : delay;

          const lineOpacity = interpolate(
            frame - lineDelay,
            [0, 0.15 * fps],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );

          const isHighlighted = highlightLines.includes(i + 1);
          const tokens = highlightCudaLine(line);

          return (
            <div
              key={i}
              style={{
                display: "flex",
                opacity: lineOpacity,
                backgroundColor: isHighlighted
                  ? "rgba(118,185,0,0.1)"
                  : "transparent",
                borderLeft: isHighlighted
                  ? `3px solid ${THEME.colors.nvidiaGreen}`
                  : "3px solid transparent",
                paddingLeft: 8,
                marginLeft: -11,
                borderRadius: 2,
              }}
            >
              {showLineNumbers && (
                <span
                  style={{
                    color: THEME.colors.textMuted,
                    fontSize: fontSize - 3,
                    fontFamily: fontFamilyCode,
                    width: 32,
                    textAlign: "right",
                    marginRight: 14,
                    flexShrink: 0,
                    userSelect: "none",
                    opacity: 0.5,
                  }}
                >
                  {i + 1}
                </span>
              )}
              <pre
                style={{
                  margin: 0,
                  fontFamily: fontFamilyCode,
                  fontSize,
                  lineHeight: 1.55,
                  whiteSpace: "pre",
                }}
              >
                {tokens.map((token, j) => (
                  <span key={j} style={{ color: token.color }}>
                    {token.text}
                  </span>
                ))}
              </pre>
            </div>
          );
        })}
      </div>
    </div>
  );
};
