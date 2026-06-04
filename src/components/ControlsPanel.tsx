import { FlaskConical, RotateCcw, SlidersHorizontal } from "lucide-react";
import type { Example } from "../types";

type ControlsPanelProps = {
  examples: Example[];
  inputText: string;
  temperature: number;
  topK: number;
  layer: number;
  head: number;
  selectedExampleId: string;
  onInputChange: (value: string) => void;
  onExampleSelect: (example: Example) => void;
  onTemperatureChange: (value: number) => void;
  onTopKChange: (value: number) => void;
  onLayerChange: (value: number) => void;
  onHeadChange: (value: number) => void;
  onReset: () => void;
};

export function ControlsPanel({
  examples,
  inputText,
  temperature,
  topK,
  layer,
  head,
  selectedExampleId,
  onInputChange,
  onExampleSelect,
  onTemperatureChange,
  onTopKChange,
  onLayerChange,
  onHeadChange,
  onReset,
}: ControlsPanelProps) {
  return (
    <aside className="panel controls-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Input Bench</span>
          <h2>实验输入</h2>
        </div>
        <button className="icon-button" onClick={onReset} title="重置输入">
          <RotateCcw size={18} />
        </button>
      </div>

      <div className="example-list">
        {examples.map((example) => (
          <button
            className={`example-button ${
              example.id === selectedExampleId ? "active" : ""
            }`}
            key={example.id}
            onClick={() => onExampleSelect(example)}
          >
            <FlaskConical size={16} />
            <span>{example.title}</span>
          </button>
        ))}
      </div>

      <label className="field-label" htmlFor="prompt">
        Prompt
      </label>
      <textarea
        id="prompt"
        value={inputText}
        onChange={(event) => onInputChange(event.target.value)}
        spellCheck={false}
      />

      <div className="slider-stack">
        <div className="slider-heading">
          <SlidersHorizontal size={16} />
          <span>采样与层设置</span>
        </div>

        <RangeControl
          label="Temperature"
          min={0.2}
          max={1.6}
          step={0.1}
          value={temperature}
          onChange={onTemperatureChange}
        />
        <RangeControl
          label="Top-k"
          min={3}
          max={8}
          step={1}
          value={topK}
          onChange={onTopKChange}
        />
        <RangeControl
          label="Layer"
          min={1}
          max={6}
          step={1}
          value={layer}
          onChange={onLayerChange}
        />
        <RangeControl
          label="Head"
          min={1}
          max={4}
          step={1}
          value={head}
          onChange={onHeadChange}
        />
      </div>
    </aside>
  );
}

type RangeControlProps = {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
};

function RangeControl({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: RangeControlProps) {
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
