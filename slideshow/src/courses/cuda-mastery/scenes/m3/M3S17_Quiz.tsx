import React from "react";
import { THEME } from "../../../../styles/theme";
import { SlideLayout } from "../../../../components/SlideLayout";
import { SlideTitle } from "../../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

type QuizQuestion = {
  question: string;
  answer: string;
  delay: number;
  answerDelay: number;
};

const QuizCard: React.FC<QuizQuestion> = ({ question, answer, delay, answerDelay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const qSpring = spring({ frame: frame - delay, fps, config: { damping: 200 } });
  const qOpacity = interpolate(qSpring, [0, 1], [0, 1]);

  const showAnswer = frame > answerDelay;
  const aOpacity = interpolate(
    frame - answerDelay,
    [0, 0.3 * fps],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <div
      style={{
        padding: "12px 18px",
        backgroundColor: "rgba(255,255,255,0.03)",
        borderRadius: 10,
        border: `1px solid ${showAnswer ? THEME.colors.nvidiaGreen + "40" : "rgba(255,255,255,0.08)"}`,
        marginBottom: 10,
        opacity: qOpacity,
      }}
    >
      <div style={{ fontSize: 16, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.4 }}>
        {question}
      </div>
      {showAnswer && (
        <div
          style={{
            marginTop: 8,
            padding: "7px 12px",
            backgroundColor: "rgba(118,185,0,0.08)",
            borderRadius: 6,
            fontSize: 15,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyCode,
            lineHeight: 1.4,
            opacity: aOpacity,
          }}
        >
          {answer}
        </div>
      )}
    </div>
  );
};

export const M3S17_Quiz: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const questions: QuizQuestion[] = [
    {
      question: "Q1: How many threads execute in a warp, and what happens when threads in a warp take different branches?",
      answer: "32 threads per warp. Divergent branches execute serially -- inactive threads are masked off, reducing throughput.",
      delay: 0.5 * fps,
      answerDelay: 2 * fps,
    },
    {
      question: "Q2: What happens if __syncthreads() is inside an if-block that not all threads enter?",
      answer: "Undefined behavior / deadlock. All threads in the block must reach the same __syncthreads() call.",
      delay: 1 * fps,
      answerDelay: 3.5 * fps,
    },
    {
      question: "Q3: What is the difference between atomicAdd on global memory vs shared memory?",
      answer: "Both are atomic, but shared memory atomics are ~100x faster due to on-chip location and lower contention.",
      delay: 1.5 * fps,
      answerDelay: 5 * fps,
    },
    {
      question: "Q4: What does __shfl_down_sync(0xFFFFFFFF, val, 4) do for lane 10?",
      answer: "Lane 10 receives the value from lane 14 (10+4). The 0xFFFFFFFF mask means all 32 lanes participate.",
      delay: 2 * fps,
      answerDelay: 6.5 * fps,
    },
    {
      question: "Q5: What is the purpose of __threadfence() vs __syncthreads()?",
      answer: "__threadfence() ensures memory writes are visible to other threads (no barrier). __syncthreads() is a barrier -- threads wait.",
      delay: 2.5 * fps,
      answerDelay: 8 * fps,
    },
    {
      question: "Q6: How does cooperative_groups improve on raw __syncthreads()?",
      answer: "Type-safe, flexible granularity (warp/tile/block/grid), composable -- can pass groups to functions. Grid-wide sync possible.",
      delay: 3 * fps,
      answerDelay: 9.5 * fps,
    },
  ];

  return (
    <SlideLayout variant="gradient" moduleNumber={3}>
      <SlideTitle
        title="Module 3 Quiz"
        subtitle="Test your understanding -- answers appear automatically"
      />

      <div style={{ display: "flex", gap: 24, flex: 1 }}>
        <div style={{ flex: 1 }}>
          {questions.slice(0, 3).map((q, i) => (
            <QuizCard key={i} {...q} />
          ))}
        </div>
        <div style={{ flex: 1 }}>
          {questions.slice(3).map((q, i) => (
            <QuizCard key={i} {...q} />
          ))}
        </div>
      </div>
    </SlideLayout>
  );
};
