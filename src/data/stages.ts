import {
  Binary,
  BrainCircuit,
  GitBranch,
  Layers,
  Network,
  Sigma,
  SplitSquareHorizontal,
} from "lucide-react";
import type { TransformerStage } from "../types";

export const transformerStages: TransformerStage[] = [
  {
    id: "tokens",
    title: "Token",
    short: "文本被切成可计算的片段。",
    detail:
      "输入文本先被切成 token。token 不是严格的字或词，而是模型词表中的片段。",
    formula: "text -> [t0, t1, ...]",
    icon: Binary,
  },
  {
    id: "embedding",
    title: "Embedding",
    short: "token id 映射为向量。",
    detail:
      "每个 token id 会查表得到一个向量，再和位置信息组合，成为 Transformer 的输入。",
    formula: "x_i = E[token_i] + P[i]",
    icon: Layers,
  },
  {
    id: "attention",
    title: "Attention",
    short: "每个位置选择要关注的上下文。",
    detail:
      "注意力层为当前位置分配一组权重，用来聚合前文或上下文中的信息。",
    formula: "softmax(QK^T / sqrt(d))V",
    icon: Network,
  },
  {
    id: "residual",
    title: "Residual",
    short: "保留原信号并叠加新信息。",
    detail:
      "残差连接让信息在多层网络中稳定传递，LayerNorm 则帮助数值尺度保持可训练。",
    formula: "y = LayerNorm(x + Attention(x))",
    icon: SplitSquareHorizontal,
  },
  {
    id: "mlp",
    title: "MLP",
    short: "对每个位置做非线性变换。",
    detail:
      "MLP 层像逐位置的知识加工器，把注意力聚合后的表示变成更适合预测的特征。",
    formula: "MLP(x) = W2 act(W1x)",
    icon: BrainCircuit,
  },
  {
    id: "logits",
    title: "Logits",
    short: "向词表中每个 token 打分。",
    detail:
      "最后的向量会被投影到词表维度，softmax 后得到下一 token 的概率分布。",
    formula: "p = softmax(Wx)",
    icon: Sigma,
  },
  {
    id: "rag",
    title: "RAG",
    short: "把外部证据放进上下文。",
    detail:
      "RAG 不是模型内部层，而是把检索到的片段拼进上下文，再交给模型生成。",
    formula: "query -> retrieve -> context -> answer",
    icon: GitBranch,
  },
];
