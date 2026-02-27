import React from "react";
import { AbsoluteFill } from "remotion";
import { THEME } from "../styles/theme";
import { SlideBackground } from "./SlideBackground";
import { fontFamilyBody } from "../styles/fonts";

export const SlideLayout: React.FC<{
  children: React.ReactNode;
  variant?: "dark" | "gradient" | "code" | "accent";
  padding?: number;
  moduleNumber?: number;
  slideNumber?: number;
  totalSlides?: number;
}> = ({
  children,
  variant = "gradient",
  padding,
  moduleNumber = 1,
  slideNumber,
  totalSlides,
}) => {
  const padX = padding ?? 72;
  const padTop = padding ?? 64;
  const padBottom = padding ?? 48;

  return (
    <AbsoluteFill>
      <SlideBackground variant={variant} />

      {/* Content area — asymmetric padding: less top (badge lives there), less bottom */}
      <div
        style={{
          position: "absolute",
          top: padTop,
          left: padX,
          right: padX,
          bottom: padBottom,
          display: "flex",
          flexDirection: "column",
        }}
      >
        {children}
      </div>

      {/* Module badge — top right, larger and clearer */}
      <div
        style={{
          position: "absolute",
          top: 20,
          right: 28,
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <div
          style={{
            padding: "5px 14px",
            backgroundColor: "rgba(118,185,0,0.12)",
            border: `1px solid ${THEME.colors.nvidiaGreen}50`,
            borderRadius: 16,
            fontSize: 13,
            color: THEME.colors.nvidiaGreen,
            fontFamily: fontFamilyBody,
            fontWeight: 600,
            letterSpacing: "0.3px",
          }}
        >
          Module {moduleNumber}
        </div>
        {slideNumber && totalSlides && (
          <div
            style={{
              padding: "5px 12px",
              backgroundColor: "rgba(255,255,255,0.04)",
              borderRadius: 16,
              fontSize: 13,
              color: THEME.colors.textMuted,
              fontFamily: fontFamilyBody,
            }}
          >
            {slideNumber}/{totalSlides}
          </div>
        )}
      </div>

      {/* Bottom accent bar — thicker for visibility */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          height: 5,
          background: `linear-gradient(90deg, ${THEME.colors.nvidiaGreen}, ${THEME.colors.accentBlue}, ${THEME.colors.accentPurple})`,
        }}
      />
    </AbsoluteFill>
  );
};

export const TwoColumnLayout: React.FC<{
  left: React.ReactNode;
  right: React.ReactNode;
  variant?: "dark" | "gradient" | "code" | "accent";
  leftWidth?: string;
  moduleNumber?: number;
  slideNumber?: number;
  totalSlides?: number;
}> = ({
  left,
  right,
  variant = "gradient",
  leftWidth = "50%",
  moduleNumber = 1,
  slideNumber,
  totalSlides,
}) => {
  return (
    <SlideLayout
      variant={variant}
      moduleNumber={moduleNumber}
      slideNumber={slideNumber}
      totalSlides={totalSlides}
    >
      <div
        style={{
          display: "flex",
          gap: 56,
          flex: 1,
        }}
      >
        <div style={{ width: leftWidth, flexShrink: 0 }}>{left}</div>
        <div style={{ flex: 1 }}>{right}</div>
      </div>
    </SlideLayout>
  );
};
