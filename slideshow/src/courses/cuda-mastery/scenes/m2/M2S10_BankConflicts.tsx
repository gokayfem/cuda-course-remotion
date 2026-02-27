import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle, BulletPoint, FadeInText } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

const BankDiagram: React.FC<{
  frame: number;
  fps: number;
  delay: number;
  label: string;
  conflictType: string;
  threadToBank: (t: number) => number;
  color: string;
  labelColor: string;
}> = ({ frame, fps, delay, label, conflictType, threadToBank, color, labelColor }) => {
  const diagSpring = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });
  const diagOpacity = interpolate(diagSpring, [0, 1], [0, 1]);

  const bankCount = 16;
  const threadCount = 8;
  const cellW = 32;
  const cellH = 22;

  const bankHits = new Map<number, number>();
  Array.from({ length: threadCount }).forEach((_, t) => {
    const bank = threadToBank(t) % bankCount;
    bankHits.set(bank, (bankHits.get(bank) ?? 0) + 1);
  });

  return (
    <div style={{ opacity: diagOpacity, marginBottom: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <div style={{
          padding: "3px 10px",
          backgroundColor: `${labelColor}15`,
          borderRadius: 4,
          fontSize: 13,
          fontWeight: 700,
          color: labelColor,
          fontFamily: fontFamilyBody,
        }}>
          {conflictType}
        </div>
        <span style={{ fontSize: 13, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody }}>
          {label}
        </span>
      </div>

      {/* Thread row */}
      <div style={{ display: "flex", gap: 2, marginBottom: 4 }}>
        {Array.from({ length: threadCount }).map((_, t) => {
          const tDelay = delay + 0.3 * fps + t * 0.04 * fps;
          const tSpring = spring({
            frame: frame - tDelay,
            fps,
            config: { damping: 200 },
          });
          return (
            <div
              key={t}
              style={{
                width: cellW,
                height: 16,
                borderRadius: 2,
                backgroundColor: `${THEME.colors.accentCyan}40`,
                border: `1px solid ${THEME.colors.accentCyan}60`,
                fontSize: 10,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: THEME.colors.accentCyan,
                fontFamily: fontFamilyCode,
                opacity: interpolate(tSpring, [0, 1], [0, 1]),
              }}
            >
              T{t}
            </div>
          );
        })}
      </div>

      {/* Connection lines (simplified as arrows) */}
      <div style={{ height: 12, display: "flex", gap: 2 }}>
        {Array.from({ length: threadCount }).map((_, t) => {
          const targetBank = threadToBank(t) % bankCount;
          const lineOpacity = interpolate(
            frame - delay - 0.6 * fps - t * 0.04 * fps,
            [0, 0.15 * fps],
            [0, 0.6],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          return (
            <div
              key={t}
              style={{
                width: cellW,
                display: "flex",
                justifyContent: "center",
                opacity: lineOpacity,
              }}
            >
              <div style={{
                fontSize: 10,
                color,
                fontFamily: fontFamilyCode,
              }}>
                |
              </div>
            </div>
          );
        })}
      </div>

      {/* Bank row */}
      <div style={{ display: "flex", gap: 2 }}>
        {Array.from({ length: bankCount }).map((_, b) => {
          const hits = bankHits.get(b) ?? 0;
          const isConflict = hits > 1;
          const bDelay = delay + 0.8 * fps + b * 0.03 * fps;
          const bSpring = spring({
            frame: frame - bDelay,
            fps,
            config: { damping: 200 },
          });
          return (
            <div
              key={b}
              style={{
                width: cellW,
                height: cellH,
                borderRadius: 3,
                backgroundColor: hits > 0
                  ? (isConflict ? `${THEME.colors.accentRed}30` : `${color}25`)
                  : "rgba(255,255,255,0.03)",
                border: `1px solid ${hits > 0
                  ? (isConflict ? `${THEME.colors.accentRed}80` : `${color}60`)
                  : "rgba(255,255,255,0.08)"}`,
                fontSize: 10,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: hits > 0
                  ? (isConflict ? THEME.colors.accentRed : color)
                  : THEME.colors.textMuted,
                fontFamily: fontFamilyCode,
                fontWeight: hits > 0 ? 700 : 400,
                opacity: interpolate(bSpring, [0, 1], [0, 1]),
              }}
            >
              B{b}{hits > 1 ? `(${hits}x)` : ""}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const M2S10_BankConflicts: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const keyOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SlideLayout variant="gradient" moduleNumber={2} slideNumber={10} totalSlides={18}>
      <SlideTitle
        title="Bank Conflicts — The Hidden Performance Killer"
        subtitle="Shared memory is divided into 32 banks; simultaneous access to the same bank serializes"
      />

      <div style={{ display: "flex", gap: 40, flex: 1 }}>
        {/* Left: Diagrams */}
        <div style={{ flex: 1.3 }}>
          {/* No conflict */}
          <BankDiagram
            frame={frame}
            fps={fps}
            delay={1.5 * fps}
            label="Thread i accesses bank i"
            conflictType="No Conflict"
            threadToBank={(t) => t}
            color={THEME.colors.nvidiaGreen}
            labelColor={THEME.colors.nvidiaGreen}
          />

          {/* 2-way conflict */}
          <BankDiagram
            frame={frame}
            fps={fps}
            delay={3 * fps}
            label="Thread i accesses bank i*2 (stride=2)"
            conflictType="2-Way Conflict"
            threadToBank={(t) => t * 2}
            color={THEME.colors.accentOrange}
            labelColor={THEME.colors.accentOrange}
          />

          {/* 8-way conflict */}
          <BankDiagram
            frame={frame}
            fps={fps}
            delay={4.5 * fps}
            label="All threads access bank 0 (stride=32)"
            conflictType="N-Way Conflict"
            threadToBank={() => 0}
            color={THEME.colors.accentRed}
            labelColor={THEME.colors.accentRed}
          />
        </div>

        {/* Right: Explanation */}
        <div style={{ flex: 0.7 }}>
          <FadeInText
            text="How Banks Work"
            delay={1.5 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentOrange}
            style={{ marginBottom: 12 }}
          />

          <BulletPoint
            index={0}
            delay={2 * fps}
            text="32 banks, 4 bytes each"
            subtext="Address i maps to bank (i/4) % 32"
          />
          <BulletPoint
            index={1}
            delay={2 * fps}
            text="Parallel if different banks"
            subtext="32 threads hit 32 banks = full bandwidth"
          />
          <BulletPoint
            index={2}
            delay={2 * fps}
            text="Serialized if same bank"
            subtext="N threads on 1 bank = N-way conflict = N serial accesses"
          />
          <BulletPoint
            index={3}
            delay={2 * fps}
            text="Stride matters!"
            subtext="Stride 1 = no conflict, stride 2 = 2-way, stride 32 = 32-way"
            highlight
          />

          <FadeInText
            text="Common Fix"
            delay={5.5 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentBlue}
            style={{ marginTop: 16, marginBottom: 8 }}
          />

          <BulletPoint
            index={0}
            delay={6 * fps}
            text="Pad shared memory by +1"
            subtext="float tile[32][32+1] shifts each row to a different bank"
            highlight
          />
        </div>
      </div>

      {/* Key takeaway */}
      <div style={{
        marginTop: 8,
        padding: "12px 20px",
        backgroundColor: "rgba(255,171,64,0.08)",
        borderRadius: 10,
        border: `2px solid ${THEME.colors.accentOrange}40`,
        opacity: keyOpacity,
        textAlign: "center",
      }}>
        <span style={{ fontSize: 18, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody }}>
          In the transpose example: use{" "}
          <span style={{ color: THEME.colors.accentOrange, fontFamily: fontFamilyCode, fontWeight: 700 }}>
            tile[32][32+1]
          </span>
          {" "}to avoid 32-way bank conflicts when reading columns. This one trick adds another{" "}
          <span style={{ color: THEME.colors.nvidiaGreen, fontWeight: 700 }}>1.3x speedup</span>.
        </span>
      </div>
    </SlideLayout>
  );
};
