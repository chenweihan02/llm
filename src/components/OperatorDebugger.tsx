import { useMemo, type ReactNode } from "react";
import { ArrowRight, Database, RadioTower, Sigma } from "lucide-react";
import { MathFormula } from "./MathFormula";
import type { OperatorTrace, TensorStats, TraceLayer, TraceToken } from "../types";
import {
  clampOperatorIndex,
  formatNumber,
  formatShape,
  formatStats,
  formatTensorSample,
  getInputSample,
  getOutputSample,
  operatorSourceLabel,
  operatorGroupLabels,
  operatorToStage,
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
  const outputSample = getOutputSample(selectedOperator);
  const inputStats = formatStats(selectedOperator?.inputStats);
  const outputStats = formatStats(selectedOperator?.outputStats);
  const sourceLabel = selectedOperator
    ? operatorSourceLabel(selectedOperator)
    : "schema";
  const sourceMeasured = selectedOperator?.source?.measured ?? false;

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
                  const isActive = operator.id === selectedOperator.id;

                  return (
                    <button
                      className={`operator-step ${
                        isActive ? "active" : ""
                      }`}
                      key={operator.id}
                      onClick={() => selectOperator(globalIndex)}
                    >
                      <span>{String(globalIndex + 1).padStart(2, "0")}</span>
                      <strong>{operator.name}</strong>
                      <em>
                        {formatShape(operator.outputShape)} /{" "}
                        {operatorToStage(operator)}
                      </em>
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>

        <div className="operator-inspector">
          <div className="operator-header-row">
            <span className={`operator-badge group-${selectedOperator.group}`}>
              {operatorGroupLabels[selectedOperator.group]}
            </span>
            <span
              className={`operator-source ${
                sourceMeasured ? "measured" : "fixture"
              }`}
              title={selectedOperator.source?.note}
            >
              <RadioTower size={13} />
              {sourceLabel}
            </span>
          </div>
          <div className="operator-title-block">
            <h3>{selectedOperator.name}</h3>
            <p>{selectedOperator.description}</p>
          </div>

          <div className="operator-dataflow">
            <div className="tensor-card">
              <span>Input tensor</span>
              <em>{selectedOperator.inputTensor ?? "operator input"}</em>
              <strong>{formatShape(selectedOperator.inputShape)}</strong>
              <code>[{formatTensorSample(inputSample)}]</code>
              {inputStats ? <small>{inputStats}</small> : null}
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
              {selectedOperator.debugNote ? (
                <small>{selectedOperator.debugNote}</small>
              ) : null}
            </div>

            <b aria-label="flows to" className="dataflow-arrow">
              <ArrowRight size={16} />
            </b>
            <div className="tensor-card">
              <span>Output tensor</span>
              <em>{selectedOperator.outputTensor ?? "operator output"}</em>
              <strong>{formatShape(selectedOperator.outputShape)}</strong>
              <code>[{formatTensorSample(outputSample)}]</code>
              {outputStats ? <small>{outputStats}</small> : null}
            </div>
          </div>

          <div className="operator-watch-grid">
            <WatchList
              icon={<Database size={15} />}
              items={selectedOperator.reads ?? []}
              title="Reads"
            />
            <WatchList
              icon={<ArrowRight size={15} />}
              items={selectedOperator.writes ?? []}
              title="Writes"
            />
            <StatsWatch
              inputStats={selectedOperator.inputStats}
              outputStats={selectedOperator.outputStats}
            />
          </div>
        </div>
      </div>
    </section>
  );
}

function WatchList({
  icon,
  items,
  title,
}: {
  icon: ReactNode;
  items: string[];
  title: string;
}) {
  return (
    <div className="operator-watch-card">
      <span>
        {icon}
        {title}
      </span>
      {items.length > 0 ? (
        items.map((item) => <code key={item}>{item}</code>)
      ) : (
        <em>none recorded</em>
      )}
    </div>
  );
}

function StatsWatch({
  inputStats,
  outputStats,
}: {
  inputStats: TensorStats | undefined;
  outputStats: TensorStats | undefined;
}) {
  return (
    <div className="operator-watch-card stats-watch-card">
      <span>
        <Sigma size={15} />
        Stats delta
      </span>
      {inputStats && outputStats ? (
        <dl>
          <div>
            <dt>std</dt>
            <dd>
              {formatNumber(inputStats.std)} {"->"} {formatNumber(outputStats.std)}
            </dd>
          </div>
          <div>
            <dt>mean</dt>
            <dd>
              {formatNumber(inputStats.mean)} {"->"} {formatNumber(outputStats.mean)}
            </dd>
          </div>
          <div>
            <dt>l2</dt>
            <dd>
              {formatNumber(inputStats.l2)} {"->"} {formatNumber(outputStats.l2)}
            </dd>
          </div>
        </dl>
      ) : (
        <em>stats unavailable</em>
      )}
    </div>
  );
}
