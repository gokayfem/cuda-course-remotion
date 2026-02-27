export const THEME = {
  // NVIDIA-inspired color palette
  colors: {
    bgPrimary: "#0a0a0a",
    bgSecondary: "#111111",
    bgCard: "#1a1a2e",
    bgCode: "#0d1117",

    nvidiaGreen: "#76b900",
    nvidiaGreenLight: "#a3d64e",
    nvidiaGreenDark: "#5a8f00",

    accentBlue: "#4fc3f7",
    accentPurple: "#b388ff",
    accentOrange: "#ffab40",
    accentRed: "#ff5252",
    accentYellow: "#ffd740",
    accentCyan: "#18ffff",
    accentPink: "#ff80ab",

    textPrimary: "#ffffff",
    textSecondary: "#b0b0b0",
    textMuted: "#666666",
    textCode: "#e6e6e6",

    // Syntax highlighting
    syntaxKeyword: "#ff79c6",
    syntaxString: "#f1fa8c",
    syntaxComment: "#6272a4",
    syntaxFunction: "#50fa7b",
    syntaxType: "#8be9fd",
    syntaxNumber: "#bd93f9",
    syntaxOperator: "#ff79c6",
  },

  fonts: {
    heading: "Inter, sans-serif",
    body: "Inter, sans-serif",
    code: "JetBrains Mono, Fira Code, monospace",
  },

  spacing: {
    xs: 8,
    sm: 16,
    md: 24,
    lg: 40,
    xl: 64,
    xxl: 96,
  },
} as const;

export const SLIDE_WIDTH = 1920;
export const SLIDE_HEIGHT = 1080;
export const FPS = 30;

// 10 minutes = 600 seconds = 18,000 frames
export const TOTAL_DURATION_FRAMES = 18000;
