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

const BankCell: React.FC<{
  bank: number;
  row: number;
  color: string;
  delay: number;
  conflict?: boolean;
}> = ({ bank, row, color, delay, conflict = false }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const s = spring({ frame: frame - delay, fps, config: { damping: 200 } });
  const opacity = interpolate(s, [0, 1], [0, 1]);

  return (
    <div
      style={{
        width: 36,
        height: 28,
        backgroundColor: conflict ? `${THEME.colors.accentRed}30` : `${color}20`,
        border: `1.5px solid ${conflict ? THEME.colors.accentRed : color}`,
        borderRadius: 4,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: 11,
        color: conflict ? THEME.colors.accentRed : color,
        fontFamily: fontFamilyCode,
        fontWeight: 600,
        opacity,
      }}
    >
      B{bank}
    </div>
  );
};

export const M2S11_BankConflictFix: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const perfOpacity = interpolate(
    frame - 8 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const afterDelay = 5 * fps;
  const afterOpacity = interpolate(
    frame - afterDelay,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={2} slideNumber={11} totalSlides={18}>
      <SlideTitle
        title="Fixing Bank Conflicts — The +1 Padding Trick"
        subtitle="A simple declaration change for massive performance gains"
      />

      <div style={{ display: "flex", gap: 32, flex: 1 }}>
        {/* Left: Code comparison */}
        <div style={{ flex: 1 }}>
          <CodeBlock
            delay={0.5 * fps}
            title="bank_conflict.cu"
            fontSize={17}
            code={`// HAS bank conflicts!
__shared__ float tile[32][32];
// Column access: tile[0][col], tile[1][col]...
// Row 0 col 0 -> Bank 0
// Row 1 col 0 -> Bank 0  CONFLICT!

// NO bank conflicts!
__shared__ float tile[32][33]; // +1 padding
// Row 0 col 0 -> Bank 0
// Row 1 col 0 -> Bank 1  No conflict!
// The extra column shifts each row by 1 bank`}
            highlightLines={[2, 8]}
          />

          {/* Performance bar */}
          <div
            style={{
              marginTop: 20,
              padding: "14px 20px",
              backgroundColor: "rgba(118,185,0,0.08)",
              borderRadius: 10,
              borderLeft: `4px solid ${THEME.colors.nvidiaGreen}`,
              opacity: perfOpacity,
            }}
          >
            <div style={{ fontSize: 17, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, marginBottom: 10 }}>
              Performance improvement: <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>10-30% faster</span>
            </div>
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 13, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginBottom: 4 }}>
                  Without padding (32-way conflicts)
                </div>
                <div style={{ height: 10, backgroundColor: "rgba(255,255,255,0.05)", borderRadius: 5 }}>
                  <div style={{ width: "65%", height: "100%", backgroundColor: THEME.colors.accentRed, borderRadius: 5 }} />
                </div>
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 13, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginBottom: 4 }}>
                  With +1 padding (no conflicts)
                </div>
                <div style={{ height: 10, backgroundColor: "rgba(255,255,255,0.05)", borderRadius: 5 }}>
                  <div style={{ width: "95%", height: "100%", backgroundColor: THEME.colors.nvidiaGreen, borderRadius: 5 }} />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Bank assignment diagrams */}
        <div style={{ flex: 0.9, display: "flex", flexDirection: "column", gap: 24 }}>
          {/* Before padding */}
          <div>
            <FadeInText
              text="Before: float tile[32][32]"
              delay={2 * fps}
              fontSize={18}
              fontWeight={700}
              color={THEME.colors.accentRed}
              style={{ marginBottom: 10 }}
            />
            <div style={{ fontSize: 13, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginBottom: 8 }}>
              Column-major access: all rows hit the same bank
            </div>
            {[0, 1, 2, 3].map((row) => (
              <div key={row} style={{ display: "flex", gap: 4, alignItems: "center", marginBottom: 4 }}>
                <span style={{ width: 50, fontSize: 12, color: THEME.colors.textSecondary, fontFamily: fontFamilyCode }}>
                  Row {row}:
                </span>
                {[0, 1, 2, 3, 4].map((col) => (
                  <BankCell
                    key={col}
                    bank={(row * 32 + col) % 32}
                    row={row}
                    color={THEME.colors.accentBlue}
                    delay={2.5 * fps + row * 0.15 * fps}
                    conflict={col === 0}
                  />
                ))}
                <span style={{ fontSize: 11, color: THEME.colors.textMuted, fontFamily: fontFamilyCode }}>...</span>
              </div>
            ))}
            <div
              style={{
                marginTop: 6,
                fontSize: 13,
                color: THEME.colors.accentRed,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                opacity: interpolate(frame - 3.5 * fps, [0, 0.3 * fps], [0, 1], {
                  extrapolateLeft: "clamp", extrapolateRight: "clamp",
                }),
              }}
            >
              Column 0: B0, B0, B0, B0 ... 32-way conflict!
            </div>
          </div>

          {/* After padding */}
          <div style={{ opacity: afterOpacity }}>
            <FadeInText
              text="After: float tile[32][33]"
              delay={afterDelay}
              fontSize={18}
              fontWeight={700}
              color={THEME.colors.nvidiaGreen}
              style={{ marginBottom: 10 }}
            />
            <div style={{ fontSize: 13, color: THEME.colors.textMuted, fontFamily: fontFamilyBody, marginBottom: 8 }}>
              Padding shifts each row by 1 bank
            </div>
            {[0, 1, 2, 3].map((row) => (
              <div key={row} style={{ display: "flex", gap: 4, alignItems: "center", marginBottom: 4 }}>
                <span style={{ width: 50, fontSize: 12, color: THEME.colors.textSecondary, fontFamily: fontFamilyCode }}>
                  Row {row}:
                </span>
                {[0, 1, 2, 3, 4].map((col) => (
                  <BankCell
                    key={col}
                    bank={(row * 33 + col) % 32}
                    row={row}
                    color={THEME.colors.nvidiaGreen}
                    delay={afterDelay + 0.5 * fps + row * 0.15 * fps}
                  />
                ))}
                <span style={{ fontSize: 11, color: THEME.colors.textMuted, fontFamily: fontFamilyCode }}>...</span>
              </div>
            ))}
            <div
              style={{
                marginTop: 6,
                fontSize: 13,
                color: THEME.colors.nvidiaGreen,
                fontFamily: fontFamilyBody,
                fontWeight: 600,
                opacity: interpolate(frame - (afterDelay + 1.5 * fps), [0, 0.3 * fps], [0, 1], {
                  extrapolateLeft: "clamp", extrapolateRight: "clamp",
                }),
              }}
            >
              Column 0: B0, B1, B2, B3 ... no conflicts!
            </div>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};
