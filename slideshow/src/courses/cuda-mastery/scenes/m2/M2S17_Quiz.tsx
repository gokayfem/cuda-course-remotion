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
import { fontFamilyBody, fontFamilyCode } from "../../../../styles/fonts";

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
        padding: "14px 20px",
        backgroundColor: "rgba(255,255,255,0.03)",
        borderRadius: 10,
        border: `1px solid ${showAnswer ? THEME.colors.nvidiaGreen + "40" : "rgba(255,255,255,0.08)"}`,
        marginBottom: 10,
        opacity: qOpacity,
      }}
    >
      <div style={{ fontSize: 17, color: THEME.colors.textPrimary, fontFamily: fontFamilyBody, lineHeight: 1.4 }}>
        {question}
      </div>
      {showAnswer && (
        <div
          style={{
            marginTop: 8,
            padding: "8px 12px",
            backgroundColor: "rgba(118,185,0,0.08)",
            borderRadius: 6,
            fontSize: 16,
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

export const M2S17_Quiz: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const questions: QuizQuestion[] = [
    {
      question: "Q1: What happens when adjacent threads access adjacent memory addresses?",
      answer: "Coalesced access -- combined into 1 memory transaction (128 bytes)",
      delay: 0.5 * fps,
      answerDelay: 2 * fps,
    },
    {
      question: "Q2: Shared memory has how many banks?",
      answer: "32 banks (one per warp thread), 4 bytes wide each",
      delay: 1 * fps,
      answerDelay: 3.5 * fps,
    },
    {
      question: "Q3: How do you fix bank conflicts in a 32x32 shared memory tile?",
      answer: "Pad to 32x33 -- the +1 shifts each row's bank alignment",
      delay: 1.5 * fps,
      answerDelay: 5 * fps,
    },
    {
      question: "Q4: What is register spilling?",
      answer: "When variables overflow registers to slow local/global memory (DRAM)",
      delay: 2 * fps,
      answerDelay: 6.5 * fps,
    },
    {
      question: "Q5: When should you use constant memory?",
      answer: "When ALL threads in a warp read the SAME value -- hardware broadcast",
      delay: 2.5 * fps,
      answerDelay: 8 * fps,
    },
    {
      question: "Q6: AoS vs SoA -- which is better for GPU coalescing?",
      answer: "SoA (Structure of Arrays) -- adjacent threads read adjacent addresses",
      delay: 3 * fps,
      answerDelay: 9.5 * fps,
    },
  ];

  return (
    <SlideLayout variant="gradient" moduleNumber={2} slideNumber={17} totalSlides={18}>
      <SlideTitle
        title="Module 2 Quiz"
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
