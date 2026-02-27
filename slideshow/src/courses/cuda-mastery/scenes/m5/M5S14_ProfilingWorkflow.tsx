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
import { fontFamilyBody } from "../../../../styles/fonts";

type WorkflowStep = {
  label: string;
  description: string;
  color: string;
};

const steps: WorkflowStep[] = [
  { label: "Profile", description: "Run ncu/nsys, identify hotspot kernel", color: THEME.colors.accentBlue },
  { label: "Analyze", description: "Check: memory-bound or compute-bound?", color: THEME.colors.accentPurple },
  { label: "Optimize", description: "Apply targeted optimization", color: THEME.colors.nvidiaGreen },
  { label: "Measure", description: "Re-profile, compare metrics", color: THEME.colors.accentOrange },
  { label: "Iterate", description: "Loop back to Profile", color: THEME.colors.accentCyan },
];

const CENTER_X = 888;
const CENTER_Y = 380;
const RADIUS = 220;

export const M5S14_ProfilingWorkflow: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Calculate positions on a circle
  const stepPositions = steps.map((_, i) => {
    const angle = -Math.PI / 2 + (i * 2 * Math.PI) / steps.length;
    return {
      x: CENTER_X + RADIUS * Math.cos(angle),
      y: CENTER_Y + RADIUS * Math.sin(angle),
    };
  });

  // Center text animation
  const centerOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Active step highlight cycles through
  const cycleStart = 8 * fps;
  const cyclePeriod = 1.5 * fps;
  const activeStep =
    frame > cycleStart
      ? Math.floor((frame - cycleStart) / cyclePeriod) % steps.length
      : -1;

  return (
    <SlideLayout variant="gradient" moduleNumber={5}>
      <SlideTitle
        title="The Optimization Workflow"
        subtitle="A systematic approach to GPU performance tuning"
      />

      <div style={{ position: "relative", flex: 1, width: 1776 }}>
        <svg
          width={1776}
          height={560}
          style={{ position: "absolute", top: 0, left: 0 }}
        >
          {/* Connecting arrows between steps */}
          {steps.map((_, i) => {
            const next = (i + 1) % steps.length;
            const from = stepPositions[i];
            const to = stepPositions[next];

            const arrowDelay = 1.5 * fps + i * 1 * fps;
            const arrowProgress = interpolate(
              frame - arrowDelay,
              [0, 0.5 * fps],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );

            // Shorten the line to not overlap the circles
            const dx = to.x - from.x;
            const dy = to.y - from.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const nx = dx / dist;
            const ny = dy / dist;
            const startX = from.x + nx * 56;
            const startY = from.y + ny * 56;
            const endX = to.x - nx * 56;
            const endY = to.y - ny * 56;

            const angle = Math.atan2(endY - startY, endX - startX);
            const arrowSize = 10;

            return (
              <g key={`arrow-${i}`} opacity={arrowProgress}>
                <line
                  x1={startX}
                  y1={startY}
                  x2={startX + (endX - startX) * arrowProgress}
                  y2={startY + (endY - startY) * arrowProgress}
                  stroke={steps[i].color}
                  strokeWidth={2}
                  strokeDasharray="6,4"
                  opacity={0.5}
                />
                {arrowProgress > 0.9 && (
                  <polygon
                    points={`
                      ${endX},${endY}
                      ${endX - arrowSize * Math.cos(angle - 0.4)},${endY - arrowSize * Math.sin(angle - 0.4)}
                      ${endX - arrowSize * Math.cos(angle + 0.4)},${endY - arrowSize * Math.sin(angle + 0.4)}
                    `}
                    fill={steps[i].color}
                    opacity={0.7}
                  />
                )}
              </g>
            );
          })}
        </svg>

        {/* Step circles */}
        {steps.map((step, i) => {
          const pos = stepPositions[i];
          const stepDelay = 1 * fps + i * 0.8 * fps;
          const stepSpring = spring({
            frame: frame - stepDelay,
            fps,
            config: { damping: 200 },
          });
          const stepOpacity = interpolate(stepSpring, [0, 1], [0, 1]);
          const stepScale = interpolate(stepSpring, [0, 1], [0.5, 1]);

          const isActive = activeStep === i;
          const highlightScale = isActive ? 1.1 : 1;

          return (
            <div
              key={step.label}
              style={{
                position: "absolute",
                left: pos.x - 50,
                top: pos.y - 50,
                width: 100,
                height: 100,
                opacity: stepOpacity,
                transform: `scale(${stepScale * highlightScale})`,
                transition: "transform 0.15s ease",
              }}
            >
              {/* Circle */}
              <div
                style={{
                  width: 100,
                  height: 100,
                  borderRadius: 50,
                  backgroundColor: isActive ? `${step.color}30` : `${step.color}15`,
                  border: `3px solid ${step.color}${isActive ? "" : "80"}`,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  boxShadow: isActive ? `0 0 20px ${step.color}40` : "none",
                }}
              >
                <div
                  style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color: step.color,
                    fontFamily: fontFamilyBody,
                    textAlign: "center",
                    lineHeight: 1.2,
                  }}
                >
                  {i + 1}
                </div>
                <div
                  style={{
                    fontSize: 15,
                    fontWeight: 700,
                    color: THEME.colors.textPrimary,
                    fontFamily: fontFamilyBody,
                    textAlign: "center",
                    lineHeight: 1.2,
                  }}
                >
                  {step.label}
                </div>
              </div>

              {/* Description below circle */}
              <div
                style={{
                  position: "absolute",
                  top: 108,
                  left: -60,
                  width: 220,
                  textAlign: "center",
                  fontSize: 13,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  lineHeight: 1.4,
                }}
              >
                {step.description}
              </div>
            </div>
          );
        })}

        {/* Center message */}
        <div
          style={{
            position: "absolute",
            left: CENTER_X - 130,
            top: CENTER_Y - 30,
            width: 260,
            textAlign: "center",
            opacity: centerOpacity,
          }}
        >
          <div
            style={{
              fontSize: 16,
              fontWeight: 700,
              color: THEME.colors.accentYellow,
              fontFamily: fontFamilyBody,
              lineHeight: 1.5,
            }}
          >
            Never optimize without profiling first!
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
