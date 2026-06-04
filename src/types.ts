import type { LucideIcon } from "lucide-react";

export type TokenKind = "han" | "latin" | "number" | "punctuation" | "symbol";

export type TokenInfo = {
  id: number;
  text: string;
  index: number;
  kind: TokenKind;
  bytes: number;
  color: string;
};

export type TransformerStage = {
  id: string;
  title: string;
  short: string;
  detail: string;
  formula: string;
  icon: LucideIcon;
};

export type Prediction = {
  token: string;
  probability: number;
  note: string;
};

export type Example = {
  id: string;
  title: string;
  input: string;
  focus: string;
};
