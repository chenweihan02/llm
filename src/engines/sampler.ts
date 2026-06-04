import type { Prediction, TokenInfo } from "../types";
import { stableHash } from "./tokenizer";

type Candidate = {
  token: string;
  logit: number;
  note: string;
};

export function predictNext(
  tokens: TokenInfo[],
  temperature: number,
  topK: number,
): Prediction[] {
  const text = tokens.map((token) => token.text).join("");
  const base = chooseCandidateSet(text);
  const scaled = softmax(base.map((item) => item.logit / temperature));

  return base
    .map((item, index) => ({
      token: item.token,
      probability: scaled[index],
      note: item.note,
    }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, topK);
}

export function samplePrediction(predictions: Prediction[]): string {
  const roll = (stableHash(`${Date.now()}-${Math.random()}`) % 1000) / 1000;
  let cumulative = 0;

  for (const prediction of predictions) {
    cumulative += prediction.probability;
    if (roll <= cumulative) return prediction.token;
  }

  return predictions[0]?.token ?? "";
}

function chooseCandidateSet(text: string): Candidate[] {
  if (text.includes("因为")) {
    return [
      { token: "所以", logit: 3.4, note: "因果连接继续向后展开" },
      { token: "他", logit: 3.0, note: "回指前文主体" },
      { token: "准备", logit: 2.2, note: "补充行动状态" },
      { token: "需要", logit: 1.7, note: "进入目的解释" },
      { token: "今天", logit: 1.1, note: "增加时间信息" },
      { token: "老师", logit: 0.8, note: "扩展场景实体" },
      { token: "。", logit: 0.5, note: "结束当前句子" },
      { token: "并且", logit: 0.4, note: "继续并列叙述" },
    ];
  }

  if (text.includes("注意力") || text.includes("大模型")) {
    return [
      { token: "来", logit: 3.6, note: "常见结构：通过某机制来完成任务" },
      { token: "理解", logit: 2.9, note: "语义任务方向" },
      { token: "聚合", logit: 2.4, note: "attention 的信息汇聚含义" },
      { token: "选择", logit: 1.8, note: "指向权重分配" },
      { token: "上下文", logit: 1.5, note: "继续解释上下文依赖" },
      { token: "信息", logit: 1.2, note: "抽象对象补全" },
      { token: "。", logit: 0.8, note: "结束陈述" },
      { token: "并", logit: 0.5, note: "继续并列说明" },
    ];
  }

  if (text.includes("知识库") || text.includes("检索")) {
    return [
      { token: "，", logit: 3.2, note: "长句中继续补充流程" },
      { token: "并", logit: 2.7, note: "连接后续动作" },
      { token: "再", logit: 2.1, note: "流程中的下一步" },
      { token: "输出", logit: 1.7, note: "进入最终结果" },
      { token: "引用", logit: 1.3, note: "和证据链相关" },
      { token: "答案", logit: 1.1, note: "回答目标" },
      { token: "。", logit: 0.8, note: "结束流程描述" },
      { token: "工具", logit: 0.4, note: "扩展到 Agent 调用" },
    ];
  }

  return [
    { token: "，", logit: 2.7, note: "继续当前句子" },
    { token: "模型", logit: 2.2, note: "主题实体延续" },
    { token: "可以", logit: 1.8, note: "进入能力描述" },
    { token: "通过", logit: 1.5, note: "引出机制" },
    { token: "因此", logit: 1.1, note: "进入推论" },
    { token: "。", logit: 0.9, note: "结束句子" },
  ];
}

function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exp = values.map((value) => Math.exp(value - max));
  const sum = exp.reduce((total, value) => total + value, 0);

  return exp.map((value) => value / sum);
}
