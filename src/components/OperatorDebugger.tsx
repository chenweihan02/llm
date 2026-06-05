import { useMemo } from "react";
import { ArrowRight } from "lucide-react";
import { MathFormula } from "./MathFormula";
import type { OperatorTrace, TraceLayer, TraceToken } from "../types";
import {
  clampOperatorIndex,
  formatShape,
  formatTensorSample,
  getInputSample,
  operatorGroupLabels,
} from "../lib/operatorDebug";

type OperatorDebuggerProps = {
  layer: TraceLayer;
  tokens: TraceToken[];
  selectedOperatorIndex: number;
  onSelectOperator: (operatorIndex: number) => void;
};

export function OperatorDebugger({
  layer,
  tokens,
  selectedOperatorIndex,
  onSelectOperator,
}: OperatorDebuggerProps) {
  const firstOperator = layer.operators[0];
  const boundedOperatorIndex = clampOperatorIndex(
    selectedOperatorIndex,
    layer.operators,
  );
  const selectedOperator = layer.operators[boundedOperatorIndex] ?? firstOperator;
  const inputSample = getInputSample(
    layer.operators,
    boundedOperatorIndex,
    tokens.map((token) => token.id),
  );

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

  function selectOperator(index: number) {
    onSelectOperator(clampOperatorIndex(index, layer.operators));
  }

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
          {(Object.keys(operatorGroupLabels) as OperatorTrace["group"][]).map((group) => {
            const operators = groupedOperators[group] ?? [];
            if (operators.length === 0) return null;

            return (
              <div className="operator-group" key={group}>
                <span className="operator-group-title">
                  {operatorGroupLabels[group]}
                </span>
                {operators.map((operator) => {
                  const globalIndex = layer.operators.findIndex(
                    (item) => item.id === operator.id,
                  );

                  return (
                    <button
                      className={`operator-step ${
                        operator.id === selectedOperator.id ? "active" : ""
                      }`}
                      key={operator.id}
                      onClick={() => selectOperator(globalIndex)}
                    >
                      <span>{String(globalIndex + 1).padStart(2, "0")}</span>
                      <strong>{operator.name}</strong>
                      <em>{formatShape(operator.outputShape)}</em>
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>

        <div className="operator-inspector">
          <span className={`operator-badge group-${selectedOperator.group}`}>
            {operatorGroupLabels[selectedOperator.group]}
          </span>
          <h3>{selectedOperator.name}</h3>
          <p>{selectedOperator.description}</p>

          <div className="operator-dataflow">
            <div className="tensor-card">
              <span>Input tensor</span>
              <strong>{formatShape(selectedOperator.inputShape)}</strong>
              <code>[{formatTensorSample(inputSample)}]</code>
            </div>
            <b aria-label="flows to" className="dataflow-arrow">
              <ArrowRight size={16} />
            </b>

            <div className="operator-kernel">
              <span>Operator</span>
              <MathFormula
                block
                latex={selectedOperator.latex ?? selectedOperator.expression}
              />
            </div>

            <b aria-label="flows to" className="dataflow-arrow">
              <ArrowRight size={16} />
            </b>
            <div className="tensor-card">
              <span>Output tensor</span>
              <strong>{formatShape(selectedOperator.outputShape)}</strong>
              <code>[{formatTensorSample(selectedOperator.sample)}]</code>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
