import { Database, FileJson, GitBranch, Layers3, Network } from "lucide-react";
import type { InferenceTrace, ModelProfile } from "../types";

type ControlsPanelProps = {
  traces: InferenceTrace[];
  selectedTraceId: string;
  selectedTrace: InferenceTrace;
  selectedLayerIndex: number;
  selectedDecodeIndex: number;
  modelProfiles: ModelProfile[];
  selectedModel: ModelProfile;
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
  modelProfiles,
  selectedModel,
  onTraceSelect,
  onLayerChange,
  onDecodeStepChange,
}: ControlsPanelProps) {
  return (
    <aside className="panel controls-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Trace Console</span>
          <h2>推理轨迹</h2>
        </div>
        <FileJson size={19} />
      </div>

      <div className="control-block">
        <div className="control-title">
          <Database size={16} />
          <span>Trace 样例</span>
        </div>
        <div className="trace-list">
          {traces.map((trace) => (
            <button
              className={`trace-button ${
                trace.id === selectedTraceId ? "active" : ""
              }`}
              key={trace.id}
              onClick={() => onTraceSelect(trace.id)}
            >
              <strong>{trace.title}</strong>
              <span>{trace.family} · {trace.parameterScale}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="prompt-block">
        <span className="field-label">Prompt</span>
        <p>{selectedTrace.prompt}</p>
        <span className="generated-preview">
          output: {selectedTrace.generatedText}
        </span>
      </div>

      <div className="control-block">
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
            >
              L{layer.index}
              <span>{layer.label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="control-block">
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
            >
              <span>{step.phase}</span>
              <strong>{step.outputToken.text}</strong>
              <em>pos {step.inputPosition} {"->"} id {step.selectedTokenId}</em>
            </button>
          ))}
        </div>
      </div>

      <div className="model-selector">
        <div className="control-title">
          <Network size={16} />
          <span>模型族对照</span>
        </div>
        <div className="selected-model-card">
          <strong>{selectedModel.label}</strong>
          <span>{selectedModel.name}</span>
          <p>{selectedModel.summary}</p>
        </div>
        <div className="compact-model-list">
          {modelProfiles.map((profile) => (
            <span
              className={profile.id === selectedModel.id ? "active" : ""}
              key={profile.id}
            >
              {profile.label}
            </span>
          ))}
        </div>
      </div>

      <div className="source-card">
        <span>{selectedTrace.source.label}</span>
        <p>{selectedTrace.source.note}</p>
      </div>
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
