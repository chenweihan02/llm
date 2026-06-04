import type { TokenInfo, TokenKind } from "../types";

const phraseHints = [
  "Transformer",
  "Token",
  "Attention",
  "LayerNorm",
  "大模型",
  "注意力",
  "机制",
  "小明",
  "因为",
  "书包",
  "上学",
  "知识库",
  "检索",
  "证据",
  "生成",
  "回答",
  "模型",
  "用户",
  "问题",
  "工具",
  "推理",
  "上下文",
].sort((a, b) => b.length - a.length);

const colorByKind: Record<TokenKind, string> = {
  han: "#157a6e",
  latin: "#be4b2f",
  number: "#8f6a00",
  punctuation: "#5d6470",
  symbol: "#7a5aa6",
};

export function tokenize(input: string): TokenInfo[] {
  const tokens: string[] = [];
  let cursor = 0;

  while (cursor < input.length) {
    const char = input[cursor];

    if (/\s/u.test(char)) {
      cursor += 1;
      continue;
    }

    const phrase = phraseHints.find((candidate) =>
      input.slice(cursor).startsWith(candidate),
    );

    if (phrase) {
      tokens.push(phrase);
      cursor += phrase.length;
      continue;
    }

    const latin = input.slice(cursor).match(/^[A-Za-z][A-Za-z0-9_-]*/u);
    if (latin) {
      tokens.push(latin[0]);
      cursor += latin[0].length;
      continue;
    }

    const number = input.slice(cursor).match(/^\d+(?:\.\d+)?/u);
    if (number) {
      tokens.push(number[0]);
      cursor += number[0].length;
      continue;
    }

    tokens.push(char);
    cursor += 1;
  }

  return tokens.map((text, index) => {
    const kind = getKind(text);

    return {
      id: stableHash(`${text}:${index}`) % 48000 + 1024,
      text,
      index,
      kind,
      bytes: new TextEncoder().encode(text).length,
      color: colorByKind[kind],
    };
  });
}

function getKind(text: string): TokenKind {
  if (/^\p{Script=Han}+$/u.test(text)) return "han";
  if (/^[A-Za-z][A-Za-z0-9_-]*$/u.test(text)) return "latin";
  if (/^\d/u.test(text)) return "number";
  if (/^[，。！？、,.!?;:：；（）()[\]{}"'“”‘’]$/u.test(text)) {
    return "punctuation";
  }
  return "symbol";
}

export function stableHash(value: string): number {
  let hash = 2166136261;
  for (const char of value) {
    hash ^= char.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}
