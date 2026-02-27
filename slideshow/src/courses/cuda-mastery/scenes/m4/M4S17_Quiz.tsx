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

export const M4S17_Quiz: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const questions: QuizQuestion[] = [
    {
      question: "Q1: What is the minimum number of steps to reduce 1024 elements?",
      answer: "10 steps -- log2(1024) = 10. Each step halves the number of active elements.",
      delay: 0.5 * fps,
      answerDelay: 2 * fps,
    },
    {
      question: "Q2: Inclusive scan of [1, 2, 3, 4] = ?",
      answer: "[1, 3, 6, 10] -- each element is the sum of all elements up to and including itself.",
      delay: 1 * fps,
      answerDelay: 3.5 * fps,
    },
    {
      question: "Q3: Why is sequential addressing better than interleaved for reduction?",
      answer: "No warp divergence in the early steps. Threads 0..N/2 are active (contiguous), so full warps execute together.",
      delay: 1.5 * fps,
      answerDelay: 5 * fps,
    },
    {
      question: "Q4: How many atomic operations does the shared memory histogram need for the merge step?",
      answer: "num_bins per block. Each block merges its local shared memory histogram into the global result.",
      delay: 2 * fps,
      answerDelay: 6.5 * fps,
    },
    {
      question: "Q5: What role does exclusive scan play in stream compaction?",
      answer: "It computes unique write addresses. If pred[i]=1, then scan[i] is the output index for element i.",
      delay: 2.5 * fps,
      answerDelay: 8 * fps,
    },
    {
      question: "Q6: Can you do a max-reduction without atomics?",
      answer: "Yes! Same tree reduction pattern but replace += with max(). Atomics are only needed for cross-block coordination.",
      delay: 3 * fps,
      answerDelay: 9.5 * fps,
    },
  ];

  return (
    <SlideLayout variant="gradient" moduleNumber={4} slideNumber={17} totalSlides={18}>
      <SlideTitle
        title="Module 4 Quiz"
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
