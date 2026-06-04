# LLM Inference Anatomy

桌面版大模型推理解剖台：用 trace 数据可视化 tokenization、prefill、KV cache、causal attention、residual stream、MLP、logits 与 next-token selection。

## GitHub Repository Description

Interactive LLM inference anatomy trace viewer for decoder-only Transformer internals: tokens, prefill, KV cache, causal attention, residual stream, MLP, logits, and GPT/LLaMA architecture differences.

## 当前阶段

- 已重构为 trace-driven viewer，页面不再在浏览器里临时伪造采样流程。
- 中心区域是可交互 Transformer 架构图，点击模块后右侧 inspector 联动展示公式、shape、stats、attention 来源和 KV cache。
- 内置 trace 是固定结构样例，用来验证页面和数据 schema。
- `scripts/export_trace.py` 用于从 Hugging Face causal LM 导出真实 forward-pass 数值，再替换内置数据。

## 真实数值怎么来

GitHub Pages 只能托管静态文件，不能在服务器侧运行 PyTorch。真实推理值有三条路线：

1. 本地离线导出 trace JSON，再随网站发布。
2. 浏览器 WebGPU/ONNX 跑小模型，并把输出转成同一套 trace schema。
3. 外部后端/API 跑模型，静态网页只负责展示结果。

第一阶段采用第 1 条路线。

```bash
pip install torch transformers
python scripts/export_trace.py --model gpt2 --prompt "The capital of France is" --out public/traces/gpt2-capital.json
```

脚本会导出：

- token ids 与 token 文本
- selected layers 的 attention matrix
- hidden/residual/delta statistics
- top logits 与 probability
- `past_key_values` / KV cache shape

## 本地开发

```bash
npm install
npm run dev
```

访问：

```text
http://127.0.0.1:5173/llm/
```

## 构建

```bash
npm run build
```

## GitHub Pages

Vite 已配置项目页路径：

```ts
base: "/llm/"
```

推送到 `chenweihan02/llm` 后，在 GitHub 仓库启用：

```text
Settings -> Pages -> Build and deployment -> Source -> GitHub Actions
```

发布地址：

```text
https://chenweihan02.github.io/llm/
```

## 准确性边界

这个网站展示模型 forward pass 可观察到的公开张量，不展示或伪造隐藏 chain-of-thought。GPT/LLaMA 对照基于公开 decoder-only Transformer 架构特征，闭源模型的具体内部实现不能从网页中反推。
