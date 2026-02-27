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

export const M7S17_Quiz: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const questions: QuizQuestion[] = [
    {
      question: "Q1: Why is cuBLAS column-major?",
      answer:
        "Inherited from FORTRAN BLAS. Use transpose trick for row-major.",
      questionDelay: 0.5 * fps,
      answerDelay: 6 * fps,
    },
    {
      question: "Q2: What does cudnnGetConvolutionForwardAlgorithm do?",
      answer:
        "Benchmarks all algorithms and returns the fastest for your specific input shape.",
      questionDelay: 1 * fps,
      answerDelay: 12 * fps,
    },
    {
      question: "Q3: thrust::device_vector vs cudaMalloc?",
      answer:
        "RAII automatic memory management, STL-like interface, no manual free needed.",
      questionDelay: 1.5 * fps,
      answerDelay: 18 * fps,
    },
    {
      question: "Q4: Host vs Device cuRAND?",
      answer:
        "Host: bulk generation to array. Device: generate inline per-thread in kernel.",
      questionDelay: 2 * fps,
      answerDelay: 24 * fps,
    },
    {
      question: "Q5: Why does cuBLAS achieve 95%+ peak?",
      answer:
        "Hand-optimized assembly, Tensor Core utilization, memory access patterns tuned per GPU.",
      questionDelay: 2.5 * fps,
      answerDelay: 30 * fps,
    },
    {
      question: "Q6: When to write custom kernels vs use libraries?",
      answer:
        "Libraries for standard ops (GEMM, conv). Custom for fused/novel operations.",
      questionDelay: 3 * fps,
      answerDelay: 36 * fps,
    },
  ];

  return (
    <SlideLayout variant="gradient" moduleNumber={7}>
      <SlideTitle
        title="Module 7 Quiz"
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
