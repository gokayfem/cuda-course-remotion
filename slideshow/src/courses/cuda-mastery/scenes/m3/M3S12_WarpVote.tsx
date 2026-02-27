import React from "react";
import { THEME } from "../../../../styles/theme";
import { TwoColumnLayout } from "../../../../components/SlideLayout";
import { SlideTitle, FadeInText, BulletPoint } from "../../../../components/AnimatedText";
import { CodeBlock } from "../../../../components/CodeBlock";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

const VotingDot: React.FC<{
  index: number;
  votes: boolean;
  delay: number;
  size?: number;
}> = ({ index, votes, delay, size = 22 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const dotSpring = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });
  const opacity = interpolate(dotSpring, [0, 1], [0, 1]);
  const scale = interpolate(dotSpring, [0, 1], [0.5, 1]);

  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: size / 2,
        backgroundColor: votes
          ? `${THEME.colors.nvidiaGreen}40`
          : `${THEME.colors.accentRed}40`,
        border: `2px solid ${votes ? THEME.colors.nvidiaGreen : THEME.colors.accentRed}`,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: 10,
        color: votes ? THEME.colors.nvidiaGreen : THEME.colors.accentRed,
        fontFamily: fontFamilyCode,
        fontWeight: 700,
        opacity,
        transform: `scale(${scale})`,
      }}
    >
      {index}
    </div>
  );
};

export const M3S12_WarpVote: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Vote pattern: threads 0-7,16-23 vote true; 8-15,24-31 vote false
  const votePattern = Array.from({ length: 32 }).map(
    (_, i) => i % 16 < 8
  );

  const ballotResult = votePattern.reduce(
    (mask, v, i) => (v ? mask | (1 << i) : mask),
    0
  );
  const ballotHex = `0x${(ballotResult >>> 0).toString(16).toUpperCase().padStart(8, "0")}`;

  const anyResult = votePattern.some((v) => v);
  const allResult = votePattern.every((v) => v);

  const resultOpacity = interpolate(
    frame - 7 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const useCaseOpacity = interpolate(
    frame - 9 * fps,
    [0, 0.5 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <TwoColumnLayout
      moduleNumber={3}
      leftWidth="48%"
      left={
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
          <SlideTitle
            title="Warp Vote Functions"
            subtitle="Collective boolean operations across the warp"
          />

          <CodeBlock
            delay={0.5 * fps}
            title="warp_vote.cu"
            fontSize={15}
            code={`// Returns bitmask of predicate across warp
unsigned mask = __ballot_sync(
    0xFFFFFFFF, threadIdx.x < threshold
);

// Returns true if ANY thread's predicate is true
bool any = __any_sync(0xFFFFFFFF, predicate);

// Returns true if ALL threads' predicates are true
bool all = __all_sync(0xFFFFFFFF, predicate);

// Count set bits in ballot result
int count = __popc(mask);`}
            highlightLines={[2, 3, 7, 10, 13]}
          />

          <div style={{ marginTop: 16 }}>
            <FadeInText
              text="Practical Example"
              delay={4 * fps}
              fontSize={18}
              fontWeight={700}
              color={THEME.colors.accentOrange}
              style={{ marginBottom: 8 }}
            />

            <CodeBlock
              delay={4.5 * fps}
              title="early_exit.cu"
              fontSize={14}
              showLineNumbers={false}
              code={`// Early exit: skip work if all threads done
while (iteration < maxIter) {
    bool converged = (error < tolerance);
    if (__all_sync(0xFFFFFFFF, converged))
        break;  // entire warp is done!
    // ... continue iterating
}`}
              highlightLines={[4, 5]}
            />
          </div>
        </div>
      }
      right={
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <FadeInText
            text="32 Threads Voting"
            delay={2 * fps}
            fontSize={20}
            fontWeight={700}
            color={THEME.colors.accentBlue}
            style={{ marginBottom: 4 }}
          />

          <FadeInText
            text="Predicate: threadIdx.x % 16 < 8"
            delay={2.5 * fps}
            fontSize={14}
            color={THEME.colors.textSecondary}
            style={{ marginBottom: 4, fontFamily: fontFamilyCode }}
          />

          {/* 32 voting dots in 4x8 grid */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(8, 1fr)",
              gap: 6,
              maxWidth: 240,
            }}
          >
            {votePattern.map((v, i) => (
              <VotingDot
                key={i}
                index={i}
                votes={v}
                delay={3 * fps + i * 1.5}
              />
            ))}
          </div>

          {/* Legend */}
          <div style={{ display: "flex", gap: 20, marginTop: 4 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: 6,
                  backgroundColor: THEME.colors.nvidiaGreen,
                }}
              />
              <span style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody }}>
                true (16)
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: 6,
                  backgroundColor: THEME.colors.accentRed,
                }}
              />
              <span style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyBody }}>
                false (16)
              </span>
            </div>
          </div>

          {/* Results */}
          <div style={{ opacity: resultOpacity, marginTop: 8 }}>
            <FadeInText
              text="Results"
              delay={7 * fps}
              fontSize={18}
              fontWeight={700}
              color={THEME.colors.accentCyan}
              style={{ marginBottom: 8 }}
            />

            {[
              { label: "__ballot_sync:", value: ballotHex, color: THEME.colors.accentPurple },
              { label: "__any_sync:", value: anyResult ? "true" : "false", color: THEME.colors.nvidiaGreen },
              { label: "__all_sync:", value: allResult ? "true" : "false", color: THEME.colors.accentRed },
              { label: "__popc(ballot):", value: "16", color: THEME.colors.accentOrange },
            ].map((item, i) => {
              const itemSpring = spring({
                frame: frame - (7.3 * fps + i * 0.25 * fps),
                fps,
                config: { damping: 200 },
              });
              return (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    marginBottom: 6,
                    opacity: interpolate(itemSpring, [0, 1], [0, 1]),
                    transform: `translateX(${interpolate(itemSpring, [0, 1], [-15, 0])}px)`,
                  }}
                >
                  <span style={{ fontSize: 14, color: THEME.colors.textSecondary, fontFamily: fontFamilyCode, minWidth: 140 }}>
                    {item.label}
                  </span>
                  <span style={{ fontSize: 16, color: item.color, fontFamily: fontFamilyCode, fontWeight: 700 }}>
                    {item.value}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Use cases */}
          <div
            style={{
              marginTop: 12,
              padding: "12px 16px",
              backgroundColor: "rgba(79,195,247,0.06)",
              borderRadius: 8,
              border: `1px solid ${THEME.colors.accentBlue}30`,
              opacity: useCaseOpacity,
            }}
          >
            <div style={{ fontSize: 15, fontWeight: 700, color: THEME.colors.accentBlue, fontFamily: fontFamilyBody, marginBottom: 6 }}>
              Use Cases
            </div>
            {[
              "Early exit when all threads converge",
              "Branch optimization (skip if no thread needs path)",
              "Population count for load balancing",
              "Warp-level consensus decisions",
            ].map((text, i) => (
              <div
                key={i}
                style={{
                  fontSize: 14,
                  color: THEME.colors.textSecondary,
                  fontFamily: fontFamilyBody,
                  lineHeight: 1.6,
                  paddingLeft: 12,
                  borderLeft: `2px solid ${THEME.colors.accentBlue}30`,
                  marginBottom: 3,
                }}
              >
                {text}
              </div>
            ))}
          </div>
        </div>
      }
    />
  );
};
