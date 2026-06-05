import type {
  DecodeStep,
  InferenceTrace,
  StageId,
  TraceLayer,
  TraceToken,
} from "../types";
import { MathFormula } from "./MathFormula";
import { OperatorPlaybackBar } from "./OperatorPlaybackBar";

type ArchitectureDiagramProps = {
  trace: InferenceTrace;
  layer: TraceLayer;
  decodeStep: DecodeStep;
  selectedToken: TraceToken;
  activeStageId: StageId;
  selectedOperatorIndex: number;
  isOperatorPlaying: boolean;
  onSelectStage: (stageId: StageId) => void;
  onSelectOperator: (operatorIndex: number) => void;
  onPlayingChange: (isPlaying: boolean) => void;
};

type NodeTone = "token" | "norm" | "attention" | "mlp" | "output" | "cache";

type NodeSpec = {
  id: StageId;
  label: string;
  detail: string;
  x: number;
  y: number;
  w: number;
  h: number;
  tone: NodeTone;
};

export function ArchitectureDiagram({
  trace,
  layer,
  decodeStep,
  selectedToken,
  activeStageId,
  selectedOperatorIndex,
  isOperatorPlaying,
  onSelectStage,
  onSelectOperator,
  onPlayingChange,
}: ArchitectureDiagramProps) {
  const stageById = new Map(trace.stages.map((stage) => [stage.id, stage]));
  const isLlama = trace.modelProfileId === "llama-style";
  const normLabel = isLlama ? "RMSNorm" : "LayerNorm";
  const positionLabel = isLlama ? "RoPE in Q/K" : "Position Encoding";
  const mlpLabel = isLlama ? "SwiGLU Feed Forward" : "GELU Feed Forward";
  const attentionLabel = "Masked Multi-Head Attention";
  const activeStage = stageById.get(activeStageId) ?? trace.stages[0];

  const nodes: NodeSpec[] = [
    {
      id: "tokenize",
      label: "Token IDs",
      detail: `[1, ${trace.tokens.length}]`,
      x: 460,
      y: 792,
      w: 200,
      h: 46,
      tone: "token",
    },
    {
      id: "embedding",
      label: "Input Embedding",
      detail: `[1, seq, ${trace.hiddenSize}]`,
      x: 420,
      y: 715,
      w: 280,
      h: 58,
      tone: "token",
    },
    {
      id: "position",
      label: positionLabel,
      detail: stageById.get("position")?.subtitle ?? "position",
      x: 160,
      y: 717,
      w: 210,
      h: 54,
      tone: "norm",
    },
    {
      id: "residual",
      label: `Add & ${normLabel}`,
      detail: "pre-attention norm",
      x: 445,
      y: 620,
      w: 230,
      h: 56,
      tone: "norm",
    },
    {
      id: "attention",
      label: attentionLabel,
      detail: `Q ${layer.heads} heads / KV ${layer.kvHeads}`,
      x: 405,
      y: 505,
      w: 310,
      h: 76,
      tone: "attention",
    },
    {
      id: "residual",
      label: `Add & ${normLabel}`,
      detail: "attention writeback",
      x: 445,
      y: 410,
      w: 230,
      h: 56,
      tone: "norm",
    },
    {
      id: "mlp",
      label: mlpLabel,
      detail: stageById.get("mlp")?.subtitle ?? "feed-forward",
      x: 425,
      y: 306,
      w: 270,
      h: 66,
      tone: "mlp",
    },
    {
      id: "residual",
      label: `Add & ${normLabel}`,
      detail: "mlp writeback",
      x: 445,
      y: 210,
      w: 230,
      h: 56,
      tone: "norm",
    },
    {
      id: "logits",
      label: "Linear / LM Head",
      detail: `[1, ${trace.vocabSize.toLocaleString()}]`,
      x: 425,
      y: 118,
      w: 270,
      h: 58,
      tone: "output",
    },
    {
      id: "sampling",
      label: "Softmax",
      detail: "output probabilities",
      x: 485,
      y: 42,
      w: 150,
      h: 48,
      tone: "output",
    },
    {
      id: "qkv",
      label: "Q/K/V projections",
      detail: formatShape(layer.kvCache.keyShape),
      x: 780,
      y: 514,
      w: 205,
      h: 64,
      tone: "cache",
    },
    {
      id: "kvcache",
      label: "KV Cache",
      detail: `${decodeStep.kvCache.memoryMB.toFixed(2)} MB / layer`,
      x: 785,
      y: 405,
      w: 195,
      h: 58,
      tone: "cache",
    },
  ];

  return (
    <section className="panel architecture-panel">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Transformer Anatomy</span>
          <h2>Decoder-only Transformer 架构</h2>
        </div>
        <span className="metric">
          L{layer.index} / {trace.layerCount} layers / hidden {trace.hiddenSize}
        </span>
      </div>

      <div className="diagram-workbench">
        <OperatorPlaybackBar
          layer={layer}
          selectedOperatorIndex={selectedOperatorIndex}
          isPlaying={isOperatorPlaying}
          onSelectOperator={onSelectOperator}
          onPlayingChange={onPlayingChange}
        />

        <svg
          className="transformer-schematic transformer-schematic-tall"
          viewBox="0 0 1120 860"
          role="img"
          aria-label={`${trace.title} detailed decoder-only transformer diagram, selected token ${selectedToken.text}`}
        >
          <defs>
            <marker
              id="arrow-head"
              markerHeight="5"
              markerWidth="5"
              orient="auto"
              refX="4.5"
              refY="2.5"
            >
              <path d="M 0 0 L 5 2.5 L 0 5 z" />
            </marker>
          </defs>

          <rect className="diagram-bg" x="40" y="20" width="1040" height="820" rx="14" />

          <g className="decoder-stack">
            <rect x="350" y="190" width="420" height="510" rx="16" />
            <text x="374" y="222">decoder block repeated N x</text>
            <text x="374" y="246">expanded operator trace: Layer {layer.index}</text>
          </g>

          <text className="nx-label" x="785" y="501">N x</text>

          <path className="flow-line main-flow" d="M560 792 L560 773" />
          <path className="flow-line main-flow" d="M560 715 L560 676" />
          <path className="flow-line main-flow" d="M560 620 L560 581" />
          <path className="flow-line main-flow" d="M560 505 L560 466" />
          <path className="flow-line main-flow" d="M560 410 L560 372" />
          <path className="flow-line main-flow" d="M560 306 L560 266" />
          <path className="flow-line main-flow" d="M560 210 L560 176" />
          <path className="flow-line main-flow" d="M560 118 L560 90" />

          <path className="flow-line side-flow" d="M370 744 L420 744" />
          <path className="flow-line cache-flow" d="M675 648 C750 631 755 554 780 546" />
          <path className="flow-line cache-flow" d="M780 546 L715 546" />
          <path className="flow-line cache-flow" d="M882 514 L882 463" />
          <path className="flow-line cache-flow reverse-flow" d="M785 434 C748 458 750 526 715 536" />

          <path className="residual-loop" d="M445 648 C280 648 280 438 445 438" />
          <path className="residual-loop" d="M675 438 C820 438 820 238 675 238" />
          <path className="residual-loop residual-loop-soft" d="M695 339 C790 339 790 238 675 238" />

          {nodes.map((node, index) => (
            <DiagramNode
              key={`${node.id}-${node.label}-${index}`}
              node={node}
              selected={node.id === activeStageId}
              onSelect={onSelectStage}
            />
          ))}

        </svg>

        <div className="diagram-readout">
          <span className="arch-kicker">Selected module</span>
          <h3>{activeStage.title}</h3>
          <p>{activeStage.description}</p>
          <div className="stage-formula">
            <MathFormula block latex={activeStage.latex ?? activeStage.formula} />
          </div>
          <div className="readout-grid">
            <div>
              <span>Tensor shape</span>
              <strong>{activeStage.shape}</strong>
            </div>
            <div>
              <span>Layer 0 operators</span>
              <strong>{layer.operators.length} traced ops</strong>
            </div>
            <div>
              <span>KV cache</span>
              <strong>K {formatShape(decodeStep.kvCache.keyShape)}</strong>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

type DiagramNodeProps = {
  node: NodeSpec;
  selected: boolean;
  onSelect: (stageId: StageId) => void;
};

function DiagramNode({ node, selected, onSelect }: DiagramNodeProps) {
  const titleY = node.y + Math.round(node.h * 0.42);
  const detailY = node.y + Math.round(node.h * 0.82);

  return (
    <g
      className={`diagram-node tone-${node.tone} ${selected ? "selected" : ""}`}
      onClick={() => onSelect(node.id)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          onSelect(node.id);
        }
      }}
      role="button"
      tabIndex={0}
    >
      {selected ? (
        <rect
          className="node-focus-ring"
          x={node.x - 7}
          y={node.y - 7}
          width={node.w + 14}
          height={node.h + 14}
          rx="13"
        />
      ) : null}
      <rect className="node-body" x={node.x} y={node.y} width={node.w} height={node.h} rx="8" />
      <text className="node-title" x={node.x + node.w / 2} y={titleY}>
        {node.label}
      </text>
      <text className="node-detail" x={node.x + node.w / 2} y={detailY}>
        {node.detail}
      </text>
    </g>
  );
}

function formatShape(shape: number[]) {
  return `[${shape.join(", ")}]`;
}
