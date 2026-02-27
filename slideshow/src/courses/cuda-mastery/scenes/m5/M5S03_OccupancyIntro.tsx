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
import { fontFamilyBody, fontFamilyCode, fontFamilyHeading } from "../../../../styles/fonts";

export const M5S03_OccupancyIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Formula animation
  const formulaSpring = spring({
    frame: frame - 1 * fps,
    fps,
    config: { damping: 200 },
  });
  const formulaOpacity = interpolate(formulaSpring, [0, 1], [0, 1]);
  const formulaScale = interpolate(formulaSpring, [0, 1], [0.9, 1]);

  // SM warp slot grid: 48 slots (6 rows x 8 cols)
  const TOTAL_SLOTS = 48;
  const COLS = 8;
  const ROWS = 6;
  const SLOT_SIZE = 34;
  const SLOT_GAP = 4;
  const GRID_WIDTH = COLS * (SLOT_SIZE + SLOT_GAP);

  // Three scenarios animated sequentially
  const scenarios = [
    { active: 8, label: "Low: 8/48", color: THEME.colors.accentRed, delay: 3 * fps },
    { active: 32, label: "Medium: 32/48", color: THEME.colors.accentOrange, delay: 5 * fps },
    { active: 48, label: "High: 48/48", color: THEME.colors.nvidiaGreen, delay: 7 * fps },
  ];

  // Determine which scenario is currently active
  const currentScenarioIndex = frame < scenarios[1].delay
    ? 0
    : frame < scenarios[2].delay
      ? 1
      : 2;

  const currentScenario = scenarios[currentScenarioIndex];

  // Animate slot filling
  const fillProgress = interpolate(
    frame - currentScenario.delay,
    [0, 1 * fps],
    [0, currentScenario.active],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Scenario label opacity
  const scenarioLabelOpacity = interpolate(
    frame - currentScenario.delay,
    [0, 0.3 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Occupancy percentage
  const occupancyPct = Math.round((Math.min(fillProgress, currentScenario.active) / TOTAL_SLOTS) * 100);

  // Bottom tip box
  const tipOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Latency hiding label
  const latencyOpacity = interpolate(
    frame - 8.5 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={5}>
      <SlideTitle
        title="What is Occupancy?"
        subtitle="The ratio of active warps to the maximum supported by an SM"
      />

      {/* Formula */}
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          marginTop: 8,
          marginBottom: 24,
          opacity: formulaOpacity,
          transform: `scale(${formulaScale})`,
          width: 1776,
        }}
      >
        <div
          style={{
            padding: "16px 40px",
            backgroundColor: "rgba(118,185,0,0.08)",
            borderRadius: 12,
            border: `2px solid ${THEME.colors.nvidiaGreen}50`,
            display: "flex",
            alignItems: "center",
            gap: 16,
          }}
        >
          <span
            style={{
              fontSize: 28,
              fontWeight: 700,
              color: THEME.colors.nvidiaGreen,
              fontFamily: fontFamilyHeading,
            }}
          >
            Occupancy
          </span>
          <span
            style={{
              fontSize: 28,
              color: THEME.colors.textPrimary,
              fontFamily: fontFamilyHeading,
              fontWeight: 400,
            }}
          >
            =
          </span>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
            <span
              style={{
                fontSize: 22,
                color: THEME.colors.accentBlue,
                fontFamily: fontFamilyCode,
                fontWeight: 700,
              }}
            >
              Active Warps
            </span>
            <div
              style={{
                width: 200,
                height: 2,
                backgroundColor: THEME.colors.textSecondary,
                margin: "4px 0",
              }}
            />
            <span
              style={{
                fontSize: 22,
                color: THEME.colors.accentPurple,
                fontFamily: fontFamilyCode,
                fontWeight: 700,
              }}
            >
              Max Warps per SM
            </span>
          </div>
        </div>
      </div>

      {/* SM diagram and scenarios */}
      <div
        style={{
          display: "flex",
          gap: 60,
          flex: 1,
          alignItems: "flex-start",
          width: 1776,
        }}
      >
        {/* Warp slot grid */}
        <div style={{ width: 500 }}>
          <div
            style={{
              fontSize: 16,
              color: THEME.colors.textSecondary,
              fontFamily: fontFamilyBody,
              fontWeight: 600,
              marginBottom: 12,
              textAlign: "center",
            }}
          >
            SM Warp Slots (48 total)
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: `repeat(${COLS}, ${SLOT_SIZE}px)`,
              gap: SLOT_GAP,
              justifyContent: "center",
              width: 500,
            }}
          >
            {Array.from({ length: TOTAL_SLOTS }).map((_, i) => {
              const isActive = i < Math.floor(fillProgress);
              const slotColor = isActive ? currentScenario.color : "rgba(255,255,255,0.06)";
              const slotBorder = isActive
                ? `1px solid ${currentScenario.color}80`
                : "1px solid rgba(255,255,255,0.08)";

              return (
                <div
                  key={i}
                  style={{
                    width: SLOT_SIZE,
                    height: SLOT_SIZE,
                    borderRadius: 4,
                    backgroundColor: isActive ? `${currentScenario.color}30` : slotColor,
                    border: slotBorder,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 10,
                    color: isActive ? currentScenario.color : THEME.colors.textMuted,
                    fontFamily: fontFamilyCode,
                    fontWeight: 600,
                  }}
                >
                  {isActive ? "W" : ""}
                </div>
              );
            })}
          </div>

          {/* Occupancy readout */}
          <div
            style={{
              marginTop: 16,
              textAlign: "center",
              opacity: scenarioLabelOpacity,
            }}
          >
            <span
              style={{
                fontSize: 32,
                fontWeight: 800,
                color: currentScenario.color,
                fontFamily: fontFamilyHeading,
              }}
            >
              {occupancyPct}%
            </span>
            <span
              style={{
                fontSize: 18,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                marginLeft: 12,
              }}
            >
              occupancy
            </span>
          </div>
        </div>

        {/* Right side: scenario indicators and info */}
        <div style={{ flex: 1, paddingTop: 10, width: 500 }}>
          {/* Scenario indicator pills */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 14,
              marginBottom: 24,
              width: 500,
            }}
          >
            {scenarios.map((s, i) => {
              const isCurrentOrPast = i <= currentScenarioIndex;
              const isCurrent = i === currentScenarioIndex;
              const pillOpacity = interpolate(
                frame - s.delay,
                [0, 0.3 * fps],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );

              return (
                <div
                  key={s.label}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    opacity: pillOpacity,
                    width: 500,
                  }}
                >
                  <div
                    style={{
                      width: 14,
                      height: 14,
                      borderRadius: 7,
                      backgroundColor: isCurrent ? s.color : `${s.color}40`,
                      border: `2px solid ${s.color}`,
                      flexShrink: 0,
                    }}
                  />
                  <span
                    style={{
                      fontSize: 18,
                      color: isCurrent ? s.color : THEME.colors.textSecondary,
                      fontFamily: fontFamilyCode,
                      fontWeight: isCurrent ? 700 : 400,
                    }}
                  >
                    {s.label}
                  </span>
                  <span
                    style={{
                      fontSize: 14,
                      color: THEME.colors.textMuted,
                      fontFamily: fontFamilyBody,
                    }}
                  >
                    {i === 0 ? "warps" : i === 1 ? "warps" : "warps (full)"}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Latency hiding explanation */}
          <FadeInText
            text="Higher occupancy = more warps to hide latency"
            delay={8.5 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentCyan}
            style={{ marginBottom: 12, width: 500 }}
          />
          <FadeInText
            text="When one warp stalls on memory, the SM switches to another ready warp instantly."
            delay={9 * fps}
            fontSize={16}
            color={THEME.colors.textSecondary}
            style={{ width: 500 }}
          />

          {/* Tip box */}
          <div
            style={{
              marginTop: 24,
              padding: "12px 18px",
              backgroundColor: "rgba(255,171,64,0.08)",
              borderRadius: 10,
              border: `1px solid ${THEME.colors.accentOrange}40`,
              opacity: tipOpacity,
              width: 460,
            }}
          >
            <span
              style={{
                fontSize: 16,
                color: THEME.colors.accentOrange,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
              }}
            >
              But 100% occupancy is not always needed!
            </span>
            <div
              style={{
                fontSize: 14,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                marginTop: 4,
              }}
            >
              Sometimes fewer warps with better ILP or caching wins.
            </div>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
