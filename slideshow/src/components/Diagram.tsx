import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../styles/theme";
import { fontFamilyBody, fontFamilyCode } from "../styles/fonts";

export const Box: React.FC<{
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  color?: string;
  delay?: number;
  sublabel?: string;
  fontSize?: number;
}> = ({
  x,
  y,
  width,
  height,
  label,
  color = THEME.colors.accentBlue,
  delay = 0,
  sublabel,
  fontSize = 18,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const s = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });

  const opacity = interpolate(s, [0, 1], [0, 1]);
  const scale = interpolate(s, [0, 1], [0.8, 1]);

  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        width,
        height,
        backgroundColor: `${color}15`,
        border: `2px solid ${color}`,
        borderRadius: 8,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        opacity,
        transform: `scale(${scale})`,
      }}
    >
      <span
        style={{
          color,
          fontSize,
          fontWeight: 700,
          fontFamily: fontFamilyBody,
          textAlign: "center",
        }}
      >
        {label}
      </span>
      {sublabel && (
        <span
          style={{
            color: THEME.colors.textSecondary,
            fontSize: fontSize - 4,
            fontFamily: fontFamilyCode,
            marginTop: 4,
            textAlign: "center",
          }}
        >
          {sublabel}
        </span>
      )}
    </div>
  );
};

export const Arrow: React.FC<{
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color?: string;
  delay?: number;
  label?: string;
  dashed?: boolean;
}> = ({
  x1,
  y1,
  x2,
  y2,
  color = THEME.colors.textSecondary,
  delay = 0,
  label,
  dashed = false,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = interpolate(frame - delay, [0, 0.5 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const currentX2 = x1 + (x2 - x1) * progress;
  const currentY2 = y1 + (y2 - y1) * progress;

  const angle = Math.atan2(y2 - y1, x2 - x1);
  const arrowSize = 10;

  const midX = (x1 + x2) / 2;
  const midY = (y1 + y2) / 2;

  return (
    <svg
      style={{
        position: "absolute",
        left: 0,
        top: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
      }}
    >
      <line
        x1={x1}
        y1={y1}
        x2={currentX2}
        y2={currentY2}
        stroke={color}
        strokeWidth={2}
        strokeDasharray={dashed ? "8 4" : "none"}
      />
      {progress > 0.9 && (
        <polygon
          points={`
            ${x2},${y2}
            ${x2 - arrowSize * Math.cos(angle - 0.4)},${y2 - arrowSize * Math.sin(angle - 0.4)}
            ${x2 - arrowSize * Math.cos(angle + 0.4)},${y2 - arrowSize * Math.sin(angle + 0.4)}
          `}
          fill={color}
          opacity={interpolate(progress, [0.9, 1], [0, 1])}
        />
      )}
      {label && (
        <text
          x={midX}
          y={midY - 10}
          fill={color}
          fontSize={14}
          fontFamily={fontFamilyBody}
          textAnchor="middle"
          opacity={progress}
        >
          {label}
        </text>
      )}
    </svg>
  );
};

export const ThreadGrid: React.FC<{
  rows: number;
  cols: number;
  x: number;
  y: number;
  cellSize?: number;
  color?: string;
  delay?: number;
  label?: string;
  showIndices?: boolean;
  highlightCells?: number[];
}> = ({
  rows,
  cols,
  x,
  y,
  cellSize = 40,
  color = THEME.colors.accentBlue,
  delay = 0,
  label,
  showIndices = false,
  highlightCells = [],
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <div style={{ position: "absolute", left: x, top: y }}>
      {label && (
        <div
          style={{
            color,
            fontSize: 16,
            fontWeight: 700,
            fontFamily: fontFamilyBody,
            marginBottom: 8,
            textAlign: "center",
          }}
        >
          {label}
        </div>
      )}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`,
          gap: 3,
        }}
      >
        {Array.from({ length: rows * cols }).map((_, i) => {
          const cellDelay = delay + i * 1.5;
          const s = spring({
            frame: frame - cellDelay,
            fps,
            config: { damping: 200 },
          });
          const opacity = interpolate(s, [0, 1], [0, 1]);
          const isHighlighted = highlightCells.includes(i);

          return (
            <div
              key={i}
              style={{
                width: cellSize,
                height: cellSize,
                backgroundColor: isHighlighted
                  ? `${THEME.colors.nvidiaGreen}40`
                  : `${color}20`,
                border: `1.5px solid ${isHighlighted ? THEME.colors.nvidiaGreen : color}`,
                borderRadius: 4,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                opacity,
                fontSize: showIndices ? 12 : 10,
                color: isHighlighted
                  ? THEME.colors.nvidiaGreen
                  : THEME.colors.textSecondary,
                fontFamily: fontFamilyCode,
                fontWeight: isHighlighted ? 700 : 400,
              }}
            >
              {showIndices ? `T${i}` : ""}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const ProgressBar: React.FC<{
  progress: number;
  label: string;
  color?: string;
  width?: number;
  x?: number;
  y?: number;
}> = ({
  progress,
  label,
  color = THEME.colors.nvidiaGreen,
  width = 400,
  x = 0,
  y = 0,
}) => {
  return (
    <div style={{ position: "absolute", left: x, top: y }}>
      <div
        style={{
          fontSize: 16,
          color: THEME.colors.textSecondary,
          fontFamily: fontFamilyBody,
          marginBottom: 6,
        }}
      >
        {label}
      </div>
      <div
        style={{
          width,
          height: 12,
          backgroundColor: "rgba(255,255,255,0.1)",
          borderRadius: 6,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${progress * 100}%`,
            height: "100%",
            backgroundColor: color,
            borderRadius: 6,
            transition: "none",
          }}
        />
      </div>
    </div>
  );
};
