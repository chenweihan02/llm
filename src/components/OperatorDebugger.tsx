import { useEffect, useMemo, useState } from "react";
import { ArrowRight } from "lucide-react";
import { MathFormula } from "./MathFormula";
import type { OperatorTrace, TraceLayer } from "../types";

type OperatorDebuggerProps = {
  layer: TraceLayer;
};

const groupLabels: Record<OperatorTrace["group"], string> = {
  embedding: "Embedding",
  attention: "Attention",
  mlp: "MLP",
  output: "Output",
  cache: "KV Cache",
};

export function OperatorDebugger({ layer }: OperatorDebuggerProps) {
  const firstOperator = layer.operators[0];
  const [selectedOperatorId, setSelectedOperatorId] = useState(
    firstOperator?.id ?? "",
  );

  useEffect(() => {
    setSelectedOperatorId(layer.operators[0]?.id ?? "");
  }, [layer.index, layer.operators]);

  const selectedOperator =
    layer.operators.find((operator) => operator.id === selectedOperatorId) ??
    firstOperator;

  const groupedOperators = useMemo(() => {
    return layer.operators.reduce(
      (groups, operator) => {
        groups[operator.group] = [...(groups[operator.group] ?? []), operator];
        return groups;
      },
      {} as Partial<Record<OperatorTrace["group"], OperatorTrace[]>>,
    );
  }, [layer.operators]);

  if (!selectedOperator) return null;

  return (
    <section className="panel operator-debug-panel">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Operator-Level Debug</span>
          <h2>Layer {layer.index} 逐算子运算链</h2>
        </div>
        <span className="metric">{layer.operators.length} ops traced</span>
      </div>

      <div className="operator-debug-layout">
        <div className="operator-rail">
          {(Object.keys(groupLabels) as OperatorTrace["group"][]).map((group) => {
            const operators = groupedOperators[group] ?? [];
            if (operators.length === 0) return null;

            return (
              <div className="operator-group" key={group}>
                <span className="operator-group-title">{groupLabels[group]}</span>
                {operators.map((operator, index) => (
                  <button
                    className={`operator-step ${
                      operator.id === selectedOperator.id ? "active" : ""
                    }`}
                    key={operator.id}
                    onClick={() => setSelectedOperatorId(operator.id)}
                  >
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <strong>{operator.name}</strong>
                    <em>{formatShape(operator.outputShape)}</em>
                  </button>
                ))}
              </div>
            );
          })}
        </div>

        <div className="operator-inspector">
          <span className={`operator-badge group-${selectedOperator.group}`}>
            {groupLabels[selectedOperator.group]}
          </span>
          <h3>{selectedOperator.name}</h3>
          <p>{selectedOperator.description}</p>

          <div className="operator-formula">
            <span>Formula</span>
            <MathFormula
              block
              latex={selectedOperator.latex ?? selectedOperator.expression}
            />
          </div>

          <div className="operator-shape-flow">
            <div>
              <span>Input</span>
              <strong>{formatShape(selectedOperator.inputShape)}</strong>
            </div>
            <b aria-label="flows to">
              <ArrowRight size={16} />
            </b>
            <div>
              <span>Output</span>
              <strong>{formatShape(selectedOperator.outputShape)}</strong>
            </div>
          </div>

          <div className="operator-sample">
            <span>sample values</span>
            <code>[{selectedOperator.sample.map(formatNumber).join(", ")}]</code>
          </div>
        </div>
      </div>
    </section>
  );
}

function formatShape(shape: number[]) {
  return `[${shape.join(", ")}]`;
}

function formatNumber(value: number) {
  if (Math.abs(value) >= 1000) return value.toExponential(1);
  return value.toFixed(3).replace(/\.?0+$/, "");
}
