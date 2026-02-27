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
          fontSize: 15,
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
            fontSize: 14,
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

export const M10S17_Quiz: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const questions: QuizQuestion[] = [
    {
      question: "Q1: What do Tensor Cores compute?",
      answer: "Matrix multiply-accumulate (D = A*B + C) on small matrices in 1 cycle",
      questionDelay: 0.5 * fps,
      answerDelay: 6 * fps,
    },
    {
      question: "Q2: Why does FP16 training need loss scaling?",
      answer: "Small gradients underflow to zero; scaling keeps them in representable range",
      questionDelay: 1 * fps,
      answerDelay: 12 * fps,
    },
    {
      question: "Q3: BF16 vs FP16?",
      answer: "BF16 has same range as FP32 (8 exp bits) but less precision. Better for training stability.",
      questionDelay: 1.5 * fps,
      answerDelay: 18 * fps,
    },
    {
      question: "Q4: What is CUTLASS?",
      answer: "NVIDIA's open-source GEMM template library. Near-cuBLAS with customizable epilogues.",
      questionDelay: 2 * fps,
      answerDelay: 24 * fps,
    },
    {
      question: "Q5: Triton vs CUDA trade-off?",
      answer: "Triton: Python, easier, auto-tune, 80-95% perf. CUDA: harder, full control, 100% perf.",
      questionDelay: 2.5 * fps,
      answerDelay: 30 * fps,
    },
    {
      question: "Q6: Three ways to add custom CUDA ops to PyTorch?",
      answer: "JIT load(), setup.py CUDAExtension, torch.compile custom ops",
      questionDelay: 3 * fps,
      answerDelay: 36 * fps,
    },
  ];

  return (
    <SlideLayout variant="gradient" moduleNumber={10}>
      <SlideTitle
        title="Module 10 Quiz"
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
