import type { OperatorTrace, StageId } from "../types";

export const operatorGroupLabels: Record<OperatorTrace["group"], string> = {
  embedding: "Embedding",
  attention: "Attention",
  mlp: "MLP",
  output: "Output",
  cache: "KV Cache",
};

export function clampOperatorIndex(index: number, operators: OperatorTrace[]) {
  if (operators.length === 0) return 0;
  return Math.min(Math.max(index, 0), operators.length - 1);
}

export function formatShape(shape: number[]) {
  return `[${shape.join(", ")}]`;
}

export function formatNumber(value: number) {
  if (Number.isInteger(value) && Math.abs(value) < 1000) return String(value);
  if (Math.abs(value) >= 1000) return value.toExponential(1);
  return value.toFixed(3).replace(/\.?0+$/, "");
}

export function formatTensorSample(values: number[]) {
  return values.map(formatNumber).join(", ");
}

export function getInputSample(
  operators: OperatorTrace[],
  selectedIndex: number,
  tokenIds: number[],
) {
  if (selectedIndex === 0) return tokenIds.slice(0, 8);
  return operators[selectedIndex - 1]?.sample ?? operators[selectedIndex]?.sample ?? [];
}

export function operatorToStage(operator: OperatorTrace): StageId {
  if (operator.id.includes("position") || operator.id.includes("rope")) {
    return "position";
  }
  if (operator.id.includes("q-proj") || operator.id.includes("kv-proj")) {
    return "qkv";
  }
  if (operator.id.includes("cache")) return "kvcache";
  if (operator.id.includes("residual")) return "residual";
  if (operator.group === "embedding") return "embedding";
  if (operator.group === "attention") return "attention";
  if (operator.group === "mlp") return "mlp";
  if (operator.group === "output") return "logits";
  return "attention";
}
