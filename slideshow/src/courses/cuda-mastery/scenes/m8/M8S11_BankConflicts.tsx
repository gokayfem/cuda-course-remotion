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

const BANK_COUNT = 32;
const GRID_COLS = 8;
const GRID_ROWS = 4;

export const M8S11_BankConflicts: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const showConflict = frame > 2 * fps && frame < 6 * fps;
  const showFix = frame > 6 * fps;

  const fixOpacity = interpolate(
    frame - 6 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const padCodeOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const bottomOpacity = interpolate(
    frame - 10 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const conflictFlash = showConflict
    ? 0.5 + 0.5 * Math.sin(frame * 0.3)
    : 0;

  return (
    <TwoColumnLayout
      variant="dark"
      moduleNumber={8}
      leftWidth="50%"
      left={
        <div style={{ width: 780 }}>
          <SlideTitle title="Shared Memory Bank Conflicts in MatMul" />

          {/* Phase label */}
          <div
            style={{
              fontSize: 16,
              fontWeight: 700,
              color: showFix
                ? THEME.colors.nvidiaGreen
                : showConflict
                  ? THEME.colors.accentRed
                  : THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
              marginBottom: 12,
            }}
          >
            {showFix
              ? "After Fix: Padding eliminates conflicts"
              : showConflict
                ? "Column-major access: 32-way bank conflict!"
                : "32 Shared Memory Banks"}
          </div>

          {/* Bank grid */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: `repeat(${GRID_COLS}, 1fr)`,
              gap: 4,
              width: 640,
              backgroundColor: "rgba(13,17,23,0.6)",
              padding: 16,
              borderRadius: 10,
              border: `1px solid ${
                showConflict
                  ? THEME.colors.accentRed + "40"
                  : showFix
                    ? THEME.colors.nvidiaGreen + "40"
                    : "rgba(255,255,255,0.08)"
              }`,
            }}
          >
            {Array.from({ length: BANK_COUNT }).map((_, i) => {
              const row = Math.floor(i / GRID_COLS);
              const col = i % GRID_COLS;
              const isConflictBank = col === 0;
              const isFixedAccess = showFix;

              const bankDelay = 0.5 * fps + i * 0.03 * fps;
              const bankSpring = spring({
                frame: frame - bankDelay,
                fps,
                config: { damping: 200 },
              });
              const bankOpacity = interpolate(bankSpring, [0, 1], [0, 1]);

              const bgColor =
                showConflict && isConflictBank
                  ? `rgba(255,82,82,${0.15 + conflictFlash * 0.35})`
                  : isFixedAccess
                    ? `rgba(118,185,0,0.12)`
                    : "rgba(255,255,255,0.04)";

              const borderColor =
                showConflict && isConflictBank
                  ? `${THEME.colors.accentRed}${Math.round(40 + conflictFlash * 60).toString(16)}`
                  : isFixedAccess
                    ? `${THEME.colors.nvidiaGreen}40`
                    : "rgba(255,255,255,0.08)";

              return (
                <div
                  key={i}
                  style={{
                    padding: "8px 4px",
                    backgroundColor: bgColor,
                    border: `1px solid ${borderColor}`,
                    borderRadius: 4,
                    textAlign: "center",
                    opacity: bankOpacity,
                  }}
                >
                  <div
                    style={{
                      fontSize: 11,
                      color: THEME.colors.textMuted,
                      fontFamily: fontFamilyCode,
                    }}
                  >
                    B{i}
                  </div>
                  <div
                    style={{
                      fontSize: 10,
                      color:
                        showConflict && isConflictBank
                          ? THEME.colors.accentRed
                          : showFix
                            ? THEME.colors.nvidiaGreen
                            : THEME.colors.textMuted,
                      fontFamily: fontFamilyCode,
                      marginTop: 2,
                    }}
                  >
                    {showConflict && isConflictBank
                      ? `T${row * GRID_COLS}..${row * GRID_COLS + 7}`
                      : showFix
                        ? `T${i}`
                        : ""}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Padding fix code */}
          <div
            style={{
              marginTop: 16,
              padding: "10px 16px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.nvidiaGreen}30`,
              opacity: padCodeOpacity,
              width: 640,
            }}
          >
            <div
              style={{
                fontSize: 15,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyCode,
                fontWeight: 700,
              }}
            >
              float As[TILE][TILE + 1]; // +1 padding
            </div>
            <div
              style={{
                fontSize: 13,
                color: THEME.colors.textSecondary,
                fontFamily: fontFamilyBody,
                marginTop: 4,
              }}
            >
              Extra column shifts each row to a different bank
            </div>
          </div>
        </div>
      }
      right={
        <div style={{ width: 480, marginTop: 80 }}>
          <BulletPoint
            text="32 banks, each 4 bytes wide"
            index={0}
            delay={2 * fps}
          />
          <BulletPoint
            text="Column access in B tile -- 32-way bank conflict!"
            index={1}
            delay={2 * fps}
            highlight
          />
          <BulletPoint
            text="Fix: pad shared memory by 1 column"
            index={2}
            delay={6 * fps}
          />
          <BulletPoint
            text="Alternative: transpose B tile layout"
            index={3}
            delay={6 * fps}
          />

          {/* Bottom callout */}
          <div
            style={{
              marginTop: 32,
              padding: "14px 20px",
              backgroundColor: "rgba(255,171,64,0.08)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.accentOrange}30`,
              opacity: bottomOpacity,
            }}
          >
            <div
              style={{
                fontSize: 16,
                color: THEME.colors.accentOrange,
                fontFamily: fontFamilyBody,
                fontWeight: 700,
                lineHeight: 1.5,
              }}
            >
              Bank conflicts can cost 2-5x performance in matmul inner loop
            </div>
          </div>
        </div>
      }
    />
  );
};
