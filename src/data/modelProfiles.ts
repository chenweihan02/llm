import type { ModelProfile } from "../types";

export const modelProfiles: ModelProfile[] = [
  {
    id: "gpt2-small",
    name: "GPT-2 small",
    label: "GPT-2",
    summary:
      "公开 GPT-2 small 是 decoder-only Transformer，12 层、768 hidden、12 heads，可用 Hugging Face 导出 attention、hidden state、logits 和 KV cache 形状。",
    tokenizer:
      "byte-level BPE，常见空格会并入 token，例如 ` Paris` 是一个带前导空格的 token。",
    position:
      "learned absolute position embedding，与 token embedding 相加进入 residual stream。",
    norm: "LayerNorm；公开 GPT-2 block 是 post-LN 风格，现代 GPT-family 常见 pre-norm 变体。",
    attention:
      "Causal multi-head self-attention；推理时 prompt prefill 并行，decode 阶段读取每层 KV cache。",
    mlp: "GELU feed-forward MLP，典型扩展维度约为 4x hidden size。",
    residual:
      "attention 与 MLP 输出逐层写回 residual stream，最后位置用于 next-token logits。",
    inference:
      "适合第一阶段真实 trace：CPU 可以导出小 prompt 的 attentions、hidden states 和 top logits。",
    caveat:
      "GPT-2 是公开小模型，不能代表 GPT-4/闭源 GPT 的真实内部结构。",
    blockCount: "12 decoder blocks",
    kvCache: "每层缓存 K/V，形状约为 [batch, heads, seq, head_dim]。",
  },
  {
    id: "gpt-style",
    name: "GPT-style",
    label: "GPT 系",
    summary:
      "以 decoder-only Transformer 为核心的自回归语言模型。公开 GPT 早期模型常见 learned position 与 LayerNorm；现代闭源 GPT 的精确内部结构不可直接确认。",
    tokenizer:
      "BPE / byte-level BPE 家族。token 边界通常不等同于自然语言词边界。",
    position:
      "GPT-2 这类公开模型使用 learned absolute position；现代 GPT-family 可能采用不同位置方案，具体依赖模型版本。",
    norm: "LayerNorm / pre-norm 变体，细节随模型代际变化。",
    attention:
      "Causal self-attention。推理时新 token 只读取前文，通常利用 KV cache 复用历史 K/V。",
    mlp: "Feed-forward MLP。公开 GPT 早期模型常见 GELU 激活；现代变体可能不同。",
    residual:
      "Residual stream 贯穿所有 block，attention 与 MLP 的输出逐层写回同一主干表示。",
    inference:
      "Prefill 阶段并行处理整段 prompt；decode 阶段每次生成一个 token，并复用 KV cache。",
    caveat:
      "这里展示的是 GPT-style 架构族，不声称复现 GPT-4 或任何闭源模型的真实内部参数。",
    blockCount: "N 个 decoder block",
    kvCache: "缓存每层 attention 的 K/V，降低逐 token 解码的重复计算。",
  },
  {
    id: "llama-style",
    name: "LLaMA-style",
    label: "LLaMA 系",
    summary:
      "公开 LLaMA-family 的典型 decoder-only 架构：RMSNorm、RoPE、SwiGLU，并在较新/较大模型中常见 grouped-query attention。",
    tokenizer:
      "SentencePiece / BPE 系 tokenizer。LLaMA 3 以后 tokenizer 方案和词表规模相比早期 LLaMA 有明显变化。",
    position:
      "RoPE rotary position embedding，把位置信息注入 Q/K 旋转空间，而不是简单相加到 token embedding。",
    norm: "RMSNorm，通常是 pre-norm，计算更轻，和 LayerNorm 的归一化形式不同。",
    attention:
      "Causal self-attention。部分 LLaMA-family 使用 GQA/MQA 思路减少 KV 头数量，改善长上下文推理成本。",
    mlp: "SwiGLU feed-forward，比早期 GELU MLP 多一个门控分支。",
    residual:
      "Residual stream 仍是主干，但 Norm、RoPE 与门控 MLP 的位置让 block 行为和 GPT 早期公开架构不同。",
    inference:
      "同样分为 prefill 与 autoregressive decode；RoPE 位置和 KV cache 长度会直接影响长上下文行为。",
    caveat:
      "这里概括的是 LLaMA-style 公开架构特征，不代表每个 LLaMA 衍生模型都完全一致。",
    blockCount: "N 个 LLaMA decoder block",
    kvCache: "缓存 RoPE 后参与注意力的 K/V；GQA 会让多个 query heads 共享较少的 K/V heads。",
  },
  {
    id: "tinystories-15m",
    name: "TinyStories 15M",
    label: "Stories 15M",
    summary:
      "适合在本地或浏览器实验的小型 decoder-only 语言模型。参数量低，便于导出真实推理 trace，但能力远弱于生产 LLM。",
    tokenizer:
      "取决于具体 checkpoint；通常可通过 Hugging Face tokenizer 直接导出 token id。",
    position:
      "取决于 checkpoint，TinyStories 系列常见 GPT-like 小型 Transformer 配置。",
    norm: "通常是小型 GPT-like block 的归一化配置，以模型 config 为准。",
    attention:
      "Causal self-attention，可导出 attention 权重和 KV cache 形状作为教学 trace。",
    mlp: "小型 feed-forward MLP，适合快速 CPU trace。",
    residual:
      "标准 residual stream，逐层累积 attention 与 MLP 写回。",
    inference:
      "GitHub Pages 不能服务器侧运行 PyTorch；可预先离线导出 trace JSON，或用浏览器 ONNX/WebGPU 模型。",
    caveat:
      "小模型的数值是真实前向值，但语义能力和大模型相差很大。",
    blockCount: "checkpoint dependent",
    kvCache: "缓存形状由模型 config 的 n_layer、n_head、head_dim 决定。",
  },
];
