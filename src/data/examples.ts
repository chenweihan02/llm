import type { Example } from "../types";

export const examples: Example[] = [
  {
    id: "coref",
    title: "指代关联",
    input: "小明把书放进书包，因为他要去上学",
    focus: "注意“他”如何回看“小明”一类的上下文线索。",
  },
  {
    id: "prediction",
    title: "下一词预测",
    input: "大模型通过注意力机制",
    focus: "观察 temperature 如何改变候选 token 的分布。",
  },
  {
    id: "rag",
    title: "RAG 流程",
    input: "用户问题进入知识库检索后，模型结合证据生成回答",
    focus: "把检索、上下文拼接和生成看成一条状态流。",
  },
];

export const defaultInput = examples[0].input;
