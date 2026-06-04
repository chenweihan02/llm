import type { TokenInfo } from "../types";
import { stableHash } from "./tokenizer";

export function buildAttentionMatrix(
  tokens: TokenInfo[],
  layer: number,
  head: number,
): number[][] {
  return tokens.map((target, targetIndex) => {
    const raw = tokens.map((source, sourceIndex) => {
      if (sourceIndex > targetIndex) return 0;

      const distance = targetIndex - sourceIndex;
      const locality = 1 / (distance + 1.25);
      const semantic = semanticAffinity(target.text, source.text);
      const kind = target.kind === source.kind ? 0.12 : 0;
      const headBias =
        (stableHash(`${target.text}-${source.text}-${layer}-${head}`) % 19) /
        100;

      return locality + semantic + kind + headBias;
    });

    const sum = raw.reduce((total, value) => total + value, 0) || 1;
    return raw.map((value) => Number((value / sum).toFixed(4)));
  });
}

export function topSources(
  tokens: TokenInfo[],
  matrix: number[][],
  targetIndex: number,
) {
  return (matrix[targetIndex] ?? [])
    .map((weight, sourceIndex) => ({
      token: tokens[sourceIndex],
      weight,
    }))
    .filter((item) => item.weight > 0)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 4);
}

function semanticAffinity(target: string, source: string): number {
  const pairs: Array<[string[], string[], number]> = [
    [["他", "她"], ["小明", "用户"], 0.9],
    [["它"], ["模型", "大模型", "知识库"], 0.72],
    [["回答", "生成"], ["问题", "证据", "检索"], 0.48],
    [["上学"], ["小明", "书包"], 0.42],
    [["注意力", "机制"], ["大模型", "模型", "Token"], 0.38],
    [["检索"], ["知识库", "问题"], 0.5],
  ];

  for (const [targets, sources, score] of pairs) {
    if (targets.includes(target) && sources.includes(source)) return score;
  }

  if (target === source) return 0.34;
  if (source.length > 1 && target.includes(source)) return 0.24;
  if (target.length > 1 && source.includes(target)) return 0.18;
  return 0;
}
