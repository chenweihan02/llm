import { Database, FileJson, FileUp, GitBranch, Layers3 } from "lucide-react";
import type { InferenceTrace, TraceLayer } from "../types";
import { OperatorPlaybackBar } from "./OperatorPlaybackBar";

type ControlsPanelProps = {
  traces: InferenceTrace[];
  selectedTraceId: string;
  selectedTrace: InferenceTrace;
  selectedLayer: TraceLayer;
  selectedLayerIndex: number;
  selectedDecodeIndex: number;
  selectedOperatorIndex: number;
  isOperatorPlaying: boolean;
  onTraceSelect: (traceId: string) => void;
  onLayerChange: (layerIndex: number) => void;
  onDecodeStepChange: (stepIndex: number) => void;
  onSelectOperator: (operatorIndex: number) => void;
  onPlayingChange: (isPlaying: boolean) => void;
  onTraceImport: (trace: unknown) => void;
  traceImportError: string;
};

export function ControlsPanel({
  traces,
  selectedTraceId,
  selectedTrace,
  selectedLayer,
  selectedLayerIndex,
  selectedDecodeIndex,
  selectedOperatorIndex,
  isOperatorPlaying,
  onTraceSelect,
  onLayerChange,
  onDecodeStepChange,
  onSelectOperator,
  onPlayingChange,
  onTraceImport,
  traceImportError,
}: ControlsPanelProps) {
  async function handleTraceFile(file: File | undefined) {
    if (!file) return;
    const text = await file.text();
    onTraceImport(JSON.parse(text));
  }

  return (
    <aside className="panel controls-panel compact-controls-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">LLM Inference</span>
          <h2>大模型推理可视化</h2>
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
        <label className="trace-import-button">
          <FileUp size={14} />
          <span>导入真实 trace JSON</span>
          <input
            accept="application/json,.json"
            onChange={(event) => {
              handleTraceFile(event.currentTarget.files?.[0]).catch((error) =>
                onTraceImport({
                  error: error instanceof Error ? error.message : "JSON import failed",
                }),
              );
              event.currentTarget.value = "";
            }}
            type="file"
          />
        </label>
        {traceImportError ? (
          <p className="trace-import-error">{traceImportError}</p>
        ) : null}
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

      <OperatorPlaybackBar
        layer={selectedLayer}
        selectedOperatorIndex={selectedOperatorIndex}
        isPlaying={isOperatorPlaying}
        onSelectOperator={onSelectOperator}
        onPlayingChange={onPlayingChange}
        variant="sidebar"
      />

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
