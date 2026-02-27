import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadJetBrainsMono } from "@remotion/google-fonts/JetBrainsMono";

const inter = loadInter("normal", {
  weights: ["400", "600", "700", "900"],
  subsets: ["latin"],
});

const jetbrainsMono = loadJetBrainsMono("normal", {
  weights: ["400", "700"],
  subsets: ["latin"],
});

export const fontFamilyHeading = inter.fontFamily;
export const fontFamilyBody = inter.fontFamily;
export const fontFamilyCode = jetbrainsMono.fontFamily;
