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

export type ModelProfile = {
  id: "gpt-style" | "llama-style" | "gpt2-small" | "tinystories-15m";
  name: string;
  label: string;
  summary: string;
  tokenizer: string;
  position: string;
  norm: string;
  attention: string;
  mlp: string;
  residual: string;
  inference: string;
  caveat: string;
  blockCount: string;
  kvCache: string;
};

export type StageId =
  | "tokenize"
  | "embedding"
  | "position"
  | "qkv"
  | "attention"
  | "residual"
  | "mlp"
  | "logits"
  | "sampling"
  | "kvcache";

export type TraceSource = {
  type: "reference-fixture" | "hf-export";
  label: string;
  modelId: string;
  command?: string;
  note: string;
};

export type TraceToken = {
  id: number;
  text: string;
  position: number;
  role: "prompt" | "generated";
  bytes: number;
  vectorNorm: number;
  color: string;
};

export type TensorShape = {
  label: string;
  shape: number[];
  dtype: "int64" | "float16" | "bfloat16" | "float32";
};

export type TensorStats = {
  min: number;
  max: number;
  mean: number;
  std: number;
  l2: number;
};

export type TraceStage = {
  id: StageId;
  title: string;
  subtitle: string;
  description: string;
  formula: string;
  shape: string;
};

export type KvCacheTrace = {
  sequenceLength: number;
  keyShape: number[];
  valueShape: number[];
  memoryMB: number;
  update: string;
};

export type TraceLayer = {
  index: number;
  label: string;
  attentionHead: number;
  heads: number;
  kvHeads: number;
  attention: number[][];
  qkScale: number;
  hidden: TensorStats;
  residual: TensorStats;
  mlp: TensorStats;
  kvCache: KvCacheTrace;
  probes: {
    name: string;
    shape: number[];
    sample: number[];
  }[];
};

export type TopLogit = {
  token: string;
  id: number;
  logit: number;
  probability: number;
  note: string;
};

export type DecodeStep = {
  index: number;
  phase: "prefill" | "decode";
  inputPosition: number;
  outputToken: TraceToken;
  selectedTokenId: number;
  entropy: number;
  temperature: number;
  topP: number;
  topLogits: TopLogit[];
  kvCache: KvCacheTrace;
};

export type ArchitectureComparison = {
  axis: string;
  gpt: string;
  llama: string;
};

export type InferenceTrace = {
  id: string;
  title: string;
  prompt: string;
  generatedText: string;
  modelProfileId: ModelProfile["id"];
  family: string;
  parameterScale: string;
  layerCount: number;
  hiddenSize: number;
  vocabSize: number;
  tokenizer: string;
  source: TraceSource;
  tensors: TensorShape[];
  stages: TraceStage[];
  tokens: TraceToken[];
  layers: TraceLayer[];
  decodeSteps: DecodeStep[];
  comparisons: ArchitectureComparison[];
};
