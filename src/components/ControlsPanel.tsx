import { Database, FileJson, GitBranch, Layers3 } from "lucide-react";
import type { InferenceTrace } from "../types";

type ControlsPanelProps = {
  traces: InferenceTrace[];
  selectedTraceId: string;
  selectedTrace: InferenceTrace;
  selectedLayerIndex: number;
  selectedDecodeIndex: number;
  onTraceSelect: (traceId: string) => void;
  onLayerChange: (layerIndex: number) => void;
  onDecodeStepChange: (stepIndex: number) => void;
};

export function ControlsPanel({
  traces,
  selectedTraceId,
  selectedTrace,
  selectedLayerIndex,
  selectedDecodeIndex,
  onTraceSelect,
  onLayerChange,
  onDecodeStepChange,
}: ControlsPanelProps) {
  return (
    <aside className="panel controls-panel compact-controls-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Trace Console</span>
          <h2>轨迹控制</h2>
        </div>
        <FileJson size={19} />
      </div>

      <div className="compact-field">
        <label htmlFor="trace-select">
          <Database size={15} />
          <span>Trace</span>
        </label>
        <select
          id="trace-select"
          value={selectedTraceId}
          onChange={(event) => onTraceSelect(event.currentTarget.value)}
        >
          {traces.map((trace) => (
            <option key={trace.id} value={trace.id}>
              {trace.title}
            </option>
          ))}
        </select>
        <div className="trace-metadata">
          <strong>{selectedTrace.family}</strong>
          <span>{selectedTrace.parameterScale}</span>
        </div>
      </div>

      <div className="control-block compact-control-block">
        <div className="control-title">
          <Layers3 size={16} />
          <span>Layer probe</span>
        </div>
        <div className="layer-buttons">
          {selectedTrace.layers.map((layer) => (
            <button
              className={`layer-button ${
                layer.index === selectedLayerIndex ? "active" : ""
              }`}
              key={layer.index}
              onClick={() => onLayerChange(layer.index)}
              title={layer.label}
            >
              L{layer.index}
              <span>{layer.label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="control-block compact-control-block">
        <div className="control-title">
          <GitBranch size={16} />
          <span>Decode step</span>
        </div>
        <div className="decode-list">
          {selectedTrace.decodeSteps.map((step) => (
            <button
              className={`decode-button ${
                step.index === selectedDecodeIndex ? "active" : ""
              }`}
              key={step.index}
              onClick={() => onDecodeStepChange(step.index)}
              title={`${step.phase}: ${step.outputToken.text} / pos ${step.inputPosition} -> id ${step.selectedTokenId}`}
            >
              <span>{step.phase}</span>
              <strong>{step.outputToken.text}</strong>
              <em>pos {step.inputPosition} {"->"} id {step.selectedTokenId}</em>
            </button>
          ))}
        </div>
      </div>

      <details className="trace-context-details">
        <summary>Prompt / source</summary>
        <div className="prompt-block compact-prompt-block">
          <span className="field-label">Prompt</span>
          <p>{selectedTrace.prompt}</p>
          <span className="generated-preview">
            output: {selectedTrace.generatedText}
          </span>
        </div>
        <div className="source-card compact-source-card">
          <span>{selectedTrace.source.label}</span>
          <p>{selectedTrace.source.note}</p>
        </div>
      </details>
    </aside>
  );
}

type LegacyRangeControlProps = {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
};

export function RangeControl({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: LegacyRangeControlProps) {
  return (
    <label className="range-control">
      <span>
        {label}
        <strong>{value}</strong>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}
