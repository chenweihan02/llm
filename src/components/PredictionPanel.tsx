import { Play, RotateCcw } from "lucide-react";
import type { Prediction } from "../types";

type PredictionPanelProps = {
  predictions: Prediction[];
  generatedTokens: string[];
  onGenerate: () => void;
  onClear: () => void;
};

export function PredictionPanel({
  predictions,
  generatedTokens,
  onGenerate,
  onClear,
}: PredictionPanelProps) {
  return (
    <section className="panel prediction-panel">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Step 04</span>
          <h2>下一词预测</h2>
        </div>
        <div className="button-row">
          <button className="icon-button" onClick={onGenerate} title="生成一步">
            <Play size={17} />
          </button>
          <button className="icon-button" onClick={onClear} title="清空生成">
            <RotateCcw size={17} />
          </button>
        </div>
      </div>

      <div className="prediction-list">
        {predictions.map((prediction) => (
          <div className="prediction-row" key={prediction.token}>
            <div className="prediction-label">
              <strong>{prediction.token}</strong>
              <span>{prediction.note}</span>
            </div>
            <div className="probability-track">
              <span
                className="probability-fill"
                style={{ width: `${Math.max(4, prediction.probability * 100)}%` }}
              />
            </div>
            <span className="probability-value">
              {Math.round(prediction.probability * 100)}%
            </span>
          </div>
        ))}
      </div>

      <div className="generated-strip">
        <span className="generated-label">Generated</span>
        {generatedTokens.length === 0 ? (
          <span className="empty-generated">等待采样</span>
        ) : (
          generatedTokens.map((token, index) => (
            <span className="generated-token" key={`${token}-${index}`}>
              {token}
            </span>
          ))
        )}
      </div>
    </section>
  );
}
