import { BarChart3 } from "lucide-react";
import type { DecodeStep } from "../types";

type PredictionPanelProps = {
  decodeSteps: DecodeStep[];
  selectedDecodeIndex: number;
  onSelectDecodeStep: (index: number) => void;
};

export function PredictionPanel({
  decodeSteps,
  selectedDecodeIndex,
  onSelectDecodeStep,
}: PredictionPanelProps) {
  const step = decodeSteps[selectedDecodeIndex] ?? decodeSteps[0];

  return (
    <section className="panel prediction-panel">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Logits Probe</span>
          <h2>下一 token 分布</h2>
        </div>
        <span className="metric">
          T={step.temperature} / top-p {step.topP}
        </span>
      </div>

      <div className="decode-tabs">
        {decodeSteps.map((item) => (
          <button
            className={item.index === selectedDecodeIndex ? "active" : ""}
            key={item.index}
            onClick={() => onSelectDecodeStep(item.index)}
          >
            <BarChart3 size={14} />
            step {item.index}: {item.outputToken.text}
          </button>
        ))}
      </div>

      <div className="prediction-list">
        {step.topLogits.map((candidate) => (
          <div className="prediction-row" key={`${candidate.id}-${candidate.token}`}>
            <div className="prediction-label">
              <strong>{candidate.token}</strong>
              <span>id {candidate.id} · logit {candidate.logit.toFixed(2)}</span>
            </div>
            <div className="probability-track">
              <span
                className="probability-fill"
                style={{
                  width: `${Math.max(3, candidate.probability * 100)}%`,
                }}
              />
            </div>
            <span className="probability-value">
              {(candidate.probability * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>

      <div className="logit-footer">
        <span>selected</span>
        <strong>{step.outputToken.text}</strong>
        <em>{step.topLogits[0]?.note}</em>
      </div>
    </section>
  );
}
