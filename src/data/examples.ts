import type { Example } from "../types";

export const examples: Example[] = [
  {
    id: "deduction",
    title: "因果续写",
    input: "如果所有金属受热会膨胀，铁是金属，那么铁受热会",
    focus: "观察 decode 步骤如何只用最后一个位置的 query 读取整段 KV cache。",
  },
  {
    id: "completion",
    title: "英文补全",
    input: "The capital of France is",
    focus: "prefill 处理完整 prompt，decode 只追加下一个 token。",
  },
  {
    id: "code",
    title: "代码续写",
    input: "function fibonacci(n) {",
    focus: "代码 token 的局部结构会改变 attention 的近邻权重和候选分布。",
  },
];

export const defaultInput = examples[0].input;
