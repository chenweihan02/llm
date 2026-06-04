import {
  Binary,
  BrainCircuit,
  Layers,
  Network,
  Sigma,
  SplitSquareHorizontal,
  SquareStack,
} from "lucide-react";
import type { TransformerStage } from "../types";

export const transformerStages: TransformerStage[] = [
  {
    id: "tokens",
    title: "Token",
    short: "文本被切成词表片段。",
    detail: "推理从 tokenizer 开始。模型实际读取的是 token id 序列，而不是原始字符串。",
    formula: "text -> [t0, t1, ...]",
    icon: Binary,
  },
  {
    id: "prefill",
    title: "Prefill",
    short: "并行处理整段 prompt。",
    detail:
      "prefill 阶段一次性计算 prompt 中所有位置的 hidden states，并为每层建立初始 KV cache。",
    formula: "X_0 = embed(tokens)",
    icon: Layers,
  },
  {
    id: "kvcache",
    title: "KV Cache",
    short: "缓存历史 K/V。",
    detail:
      "decode 时不会重复计算整段前文的 key/value，而是读取每层缓存并只为新 token 追加一列。",
    formula: "cache_l <- concat(cache_l, K_l, V_l)",
    icon: SquareStack,
  },
  {
    id: "attention",
    title: "Causal Attention",
    short: "query 只能看前文。",
    detail:
      "自回归推理使用 causal mask。新 token 的 query 与缓存中的 keys 做匹配，再加权读取 values。",
    formula: "Attn(q_t, K_<=t, V_<=t)",
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
    id: "sampling",
    title: "Sampling",
    short: "从分布中选下一个 token。",
    detail:
      "temperature、top-k、top-p 等策略作用在 logits 或概率分布上，最终决定追加哪个 token。",
    formula: "next = sample(softmax(logits / T))",
    icon: BrainCircuit,
  },
];
