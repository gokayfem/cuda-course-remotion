import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { THEME } from "../../../styles/theme";
import { SlideLayout } from "../../../components/SlideLayout";
import { SlideTitle, FadeInText } from "../../../components/AnimatedText";
import { fontFamilyBody, fontFamilyCode } from "../../../styles/fonts";

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

export const S17_Quiz: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const questions: QuizQuestion[] = [
    {
      question: "Q1: What is the global index for blockIdx.x=3, threadIdx.x=7, blockDim.x=32?",
      answer: "idx = 3 * 32 + 7 = 103",
      delay: 0.5 * fps,
      answerDelay: 2 * fps,
    },
    {
      question: "Q2: You launch <<<128, 512>>>. How many total threads?",
      answer: "128 * 512 = 65,536 threads",
      delay: 1 * fps,
      answerDelay: 3 * fps,
    },
    {
      question: "Q3: Array has 1000 elements, blockSize=256. Blocks needed? Wasted threads?",
      answer: "ceil(1000/256) = 4 blocks. Total = 1024. Wasted = 24 threads.",
      delay: 1.5 * fps,
      answerDelay: 4 * fps,
    },
    {
      question: "Q4: Why can't you use CPU timers (clock/chrono) for GPU code?",
      answer: "GPU ops are async — CPU timers measure LAUNCH time, not execution. Use CUDA Events.",
      delay: 2 * fps,
      answerDelay: 5.5 * fps,
    },
    {
      question: "Q5: Vector add reads 2 arrays, writes 1. N=1M, float32. Total memory traffic?",
      answer: "3 * 1,000,000 * 4 bytes = 12 MB",
      delay: 2.5 * fps,
      answerDelay: 7 * fps,
    },
    {
      question: "Q6: Why must blockSize be a multiple of 32?",
      answer: "GPU executes in 32-thread warps. Non-multiple = idle threads in last warp.",
      delay: 3 * fps,
      answerDelay: 8.5 * fps,
    },
  ];

  return (
    <SlideLayout variant="gradient" slideNumber={17} totalSlides={18}>
      <SlideTitle
        title="Module 1 Quiz"
        subtitle="Test your understanding — answers appear automatically"
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
