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
  questionDelay: number;
  answerDelay: number;
};

const QuizCard: React.FC<QuizQuestion> = ({
  question,
  answer,
  questionDelay,
  answerDelay,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const qSpring = spring({
    frame: frame - questionDelay,
    fps,
    config: { damping: 200 },
  });
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
      <div
        style={{
          fontSize: 16,
          color: THEME.colors.textPrimary,
          fontFamily: fontFamilyBody,
          lineHeight: 1.4,
        }}
      >
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

export const M6S17_Quiz: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const questions: QuizQuestion[] = [
    {
      question: "Q1: What is a CUDA stream?",
      answer: "A sequence of GPU operations that execute in order on the device",
      questionDelay: 0.5 * fps,
      answerDelay: 6 * fps,
    },
    {
      question: "Q2: Why must host memory be pinned for async transfers?",
      answer: "DMA engine needs physical addresses, can't handle page faults",
      questionDelay: 1 * fps,
      answerDelay: 12 * fps,
    },
    {
      question: "Q3: Max overlap with 2 copy engines?",
      answer: "Simultaneous H2D + compute + D2H (3-way overlap)",
      questionDelay: 1.5 * fps,
      answerDelay: 18 * fps,
    },
    {
      question: "Q4: cudaStreamSynchronize vs cudaDeviceSynchronize?",
      answer: "Stream sync waits for one stream; device sync waits for ALL streams",
      questionDelay: 2 * fps,
      answerDelay: 24 * fps,
    },
    {
      question: "Q5: What do CUDA events enable between streams?",
      answer: "Cross-stream dependencies without full synchronization",
      questionDelay: 2.5 * fps,
      answerDelay: 30 * fps,
    },
    {
      question: "Q6: Why are CUDA Graphs faster than stream-based launches?",
      answer: "Single launch replays entire graph, eliminating per-op CPU overhead",
      questionDelay: 3 * fps,
      answerDelay: 36 * fps,
    },
  ];

  return (
    <SlideLayout variant="gradient" moduleNumber={6}>
      <SlideTitle
        title="Module 6 Quiz"
        subtitle="Test your understanding -- answers appear automatically"
      />

      <div style={{ display: "flex", gap: 24, flex: 1, width: 1776 }}>
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
