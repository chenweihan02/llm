import { type CSSProperties } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RotateCcw,
} from "lucide-react";
import type { TraceLayer } from "../types";
import {
  clampOperatorIndex,
  formatShape,
  operatorGroupLabels,
} from "../lib/operatorDebug";

type OperatorPlaybackBarProps = {
  layer: TraceLayer;
  selectedOperatorIndex: number;
  isPlaying: boolean;
  onSelectOperator: (operatorIndex: number) => void;
  onPlayingChange: (isPlaying: boolean) => void;
  variant?: "default" | "sidebar";
};

export function OperatorPlaybackBar({
  layer,
  selectedOperatorIndex,
  isPlaying,
  onSelectOperator,
  onPlayingChange,
  variant = "default",
}: OperatorPlaybackBarProps) {
  const boundedIndex = clampOperatorIndex(selectedOperatorIndex, layer.operators);
  const selectedOperator = layer.operators[boundedIndex];

  if (!selectedOperator) return null;

  const progress =
    layer.operators.length > 1
      ? (boundedIndex / (layer.operators.length - 1)) * 100
      : 100;

  function selectOperator(index: number) {
    onSelectOperator(clampOperatorIndex(index, layer.operators));
  }

  return (
    <div className={`architecture-playback playback-${variant}`}>
      <div className="architecture-playback-main">
        <div className="playback-controls">
          <button
            className="icon-button debug-control"
            onClick={() => selectOperator(0)}
            title="Reset"
          >
            <RotateCcw size={16} />
          </button>
          <button
            className="icon-button debug-control"
            onClick={() => selectOperator(boundedIndex - 1)}
            title="Previous operator"
          >
            <ChevronLeft size={18} />
          </button>
          <button
            className="icon-button debug-control play-control"
            onClick={() => onPlayingChange(!isPlaying)}
            title={isPlaying ? "Pause" : "Play"}
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          </button>
          <button
            className="icon-button debug-control"
            onClick={() => selectOperator(boundedIndex + 1)}
            title="Next operator"
          >
            <ChevronRight size={18} />
          </button>
        </div>

        <div className="debug-progress-wrap architecture-progress-wrap">
          <div className="debug-progress-meta">
            <span>
              op {String(boundedIndex + 1).padStart(2, "0")} /{" "}
              {String(layer.operators.length).padStart(2, "0")}
            </span>
            <strong>{selectedOperator.name}</strong>
          </div>
          <input
            aria-label="operator debug progress"
            className="debug-progress"
            max={Math.max(0, layer.operators.length - 1)}
            min={0}
            onChange={(event) => selectOperator(Number(event.currentTarget.value))}
            style={{ "--progress": `${progress}%` } as CSSProperties}
            type="range"
            value={boundedIndex}
          />
        </div>

        <div className="playhead-current">
          <span className={`operator-badge group-${selectedOperator.group}`}>
            {operatorGroupLabels[selectedOperator.group]}
          </span>
          <strong>{formatShape(selectedOperator.outputShape)}</strong>
        </div>
      </div>

      <div className="architecture-operator-strip" aria-label="operator timeline">
        {layer.operators.map((operator, index) => (
          <button
            className={`operator-dot group-${operator.group} ${
              index === boundedIndex ? "active" : ""
            }`}
            key={operator.id}
            onClick={() => selectOperator(index)}
            title={`${index + 1}. ${operator.name}`}
          >
            <span>{index + 1}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
