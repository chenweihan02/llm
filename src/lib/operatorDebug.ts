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
  const selected = operators[selectedIndex];
  if (selected?.inputSample) return selected.inputSample;
  if (selectedIndex === 0) return tokenIds.slice(0, 8);
  return (
    operators[selectedIndex - 1]?.outputSample ??
    operators[selectedIndex - 1]?.sample ??
    selected?.sample ??
    []
  );
}

export function getOutputSample(operator: OperatorTrace | undefined) {
  return operator?.outputSample ?? operator?.sample ?? [];
}

export function formatStats(stats: OperatorTrace["outputStats"]) {
  if (!stats) return null;

  return `mean ${formatNumber(stats.mean)} / std ${formatNumber(
    stats.std,
  )} / l2 ${formatNumber(stats.l2)}`;
}

export function operatorSourceLabel(operator: OperatorTrace) {
  if (!operator.source) return "schema";
  if (operator.source.measured) return operator.source.label;
  return `${operator.source.label} / non-measured`;
}

export function operatorToStage(operator: OperatorTrace): StageId {
  if (operator.stageId) return operator.stageId;
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
