import type {
  DecodeStep,
  InferenceTrace,
  StageId,
  TraceLayer,
  TraceToken,
} from "../types";
import { MathFormula } from "./MathFormula";

type ArchitectureDiagramProps = {
  trace: InferenceTrace;
  layer: TraceLayer;
  decodeStep: DecodeStep;
  selectedToken: TraceToken;
  activeStageId: StageId;
  onSelectStage: (stageId: StageId) => void;
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
  onSelectStage,
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
      x: 442,
      y: 864,
      w: 230,
      h: 58,
      tone: "token",
    },
    {
      id: "embedding",
      label: "Input Embedding",
      detail: `[1, seq, ${trace.hiddenSize}]`,
      x: 404,
      y: 774,
      w: 306,
      h: 66,
      tone: "token",
    },
    {
      id: "position",
      label: positionLabel,
      detail: stageById.get("position")?.subtitle ?? "position",
      x: 124,
      y: 784,
      w: 196,
      h: 56,
      tone: "norm",
    },
    {
      id: "residual",
      label: `Add & ${normLabel}`,
      detail: "pre-attention norm",
      x: 428,
      y: 668,
      w: 258,
      h: 48,
      tone: "norm",
    },
    {
      id: "attention",
      label: attentionLabel,
      detail: `Q ${layer.heads} heads / KV ${layer.kvHeads}`,
      x: 396,
      y: 552,
      w: 322,
      h: 88,
      tone: "attention",
    },
    {
      id: "residual",
      label: `Add & ${normLabel}`,
      detail: "attention writeback",
      x: 428,
      y: 466,
      w: 258,
      h: 48,
      tone: "norm",
    },
    {
      id: "mlp",
      label: mlpLabel,
      detail: stageById.get("mlp")?.subtitle ?? "feed-forward",
      x: 396,
      y: 340,
      w: 322,
      h: 92,
      tone: "mlp",
    },
    {
      id: "residual",
      label: `Add & ${normLabel}`,
      detail: "mlp writeback",
      x: 428,
      y: 252,
      w: 258,
      h: 48,
      tone: "norm",
    },
    {
      id: "logits",
      label: "Linear / LM Head",
      detail: `[1, ${trace.vocabSize.toLocaleString()}]`,
      x: 428,
      y: 142,
      w: 258,
      h: 56,
      tone: "output",
    },
    {
      id: "sampling",
      label: "Softmax",
      detail: "output probabilities",
      x: 452,
      y: 60,
      w: 210,
      h: 48,
      tone: "output",
    },
    {
      id: "qkv",
      label: "Q/K/V projections",
      detail: formatShape(layer.kvCache.keyShape),
      x: 784,
      y: 552,
      w: 238,
      h: 88,
      tone: "cache",
    },
    {
      id: "kvcache",
      label: "KV Cache",
      detail: `${decodeStep.kvCache.memoryMB.toFixed(2)} MB / layer`,
      x: 784,
      y: 438,
      w: 238,
      h: 82,
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
        <svg
          className="transformer-schematic transformer-schematic-tall"
          viewBox="0 0 1128 960"
          role="img"
          aria-label={`${trace.title} detailed decoder-only transformer diagram`}
        >
          <defs>
            <marker
              id="arrow-head"
              markerHeight="8"
              markerWidth="8"
              orient="auto"
              refX="7"
              refY="4"
            >
              <path d="M 0 0 L 8 4 L 0 8 z" />
            </marker>
          </defs>

          <rect className="diagram-bg" x="20" y="22" width="1088" height="914" rx="18" />
          <text className="diagram-title" x="48" y="62">
            {trace.title}
          </text>
          <text className="diagram-subtitle" x="48" y="88">
            Layer 0 operator debug · selected token {selectedToken.text} · decode output {decodeStep.outputToken.text}
          </text>

          <g className="decoder-stack">
            <rect x="350" y="222" width="414" height="514" rx="22" />
            <text x="372" y="252">decoder block repeated N x</text>
            <text x="372" y="276">expanded: Layer {layer.index}</text>
          </g>

          <text className="nx-label" x="780" y="492">N x</text>
          <text className="prob-label" x="560" y="38">Output Probabilities</text>

          <path className="flow-line main-flow" d="M557 864 L557 840" />
          <path className="flow-line main-flow" d="M557 774 L557 716" />
          <path className="flow-line main-flow" d="M557 668 L557 640" />
          <path className="flow-line main-flow" d="M557 552 L557 514" />
          <path className="flow-line main-flow" d="M557 466 L557 432" />
          <path className="flow-line main-flow" d="M557 340 L557 300" />
          <path className="flow-line main-flow" d="M557 252 L557 198" />
          <path className="flow-line main-flow" d="M557 142 L557 108" />

          <path className="flow-line side-flow" d="M320 812 C354 812 374 812 404 812" />
          <path className="flow-line cache-flow" d="M718 596 L784 596" />
          <path className="flow-line cache-flow" d="M903 552 L903 520" />
          <path className="flow-line cache-flow reverse-flow" d="M784 480 C744 492 734 542 718 596" />

          <path className="residual-loop" d="M428 692 C244 692 244 490 428 490" />
          <path className="residual-loop" d="M686 490 C870 490 870 276 686 276" />
          <path className="residual-loop" d="M718 384 C760 384 764 276 686 276" />

          {nodes.map((node, index) => (
            <DiagramNode
              key={`${node.id}-${node.label}-${index}`}
              node={node}
              selected={node.id === activeStageId}
              onSelect={onSelectStage}
            />
          ))}

          <g className="operator-chain-mini">
            <text x="58" y="390">Layer {layer.index} op chain</text>
            {layer.operators.slice(2, 10).map((operator, index) => (
              <g key={operator.id} transform={`translate(58, ${410 + index * 38})`}>
                <circle r="5" cx="0" cy="0" />
                <text x="14" y="5">{operator.name}</text>
              </g>
            ))}
          </g>
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
      <rect x={node.x} y={node.y} width={node.w} height={node.h} rx="9" />
      <text className="node-title" x={node.x + node.w / 2} y={node.y + 25}>
        {node.label}
      </text>
      <text className="node-detail" x={node.x + node.w / 2} y={node.y + node.h - 16}>
        {node.detail}
      </text>
    </g>
  );
}

function formatShape(shape: number[]) {
  return `[${shape.join(", ")}]`;
}
