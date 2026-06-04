import type {
  DecodeStep,
  InferenceTrace,
  StageId,
  TraceLayer,
  TraceToken,
} from "../types";

type ArchitectureDiagramProps = {
  trace: InferenceTrace;
  layer: TraceLayer;
  decodeStep: DecodeStep;
  selectedToken: TraceToken;
  activeStageId: StageId;
  onSelectStage: (stageId: StageId) => void;
};

type NodeSpec = {
  id: StageId;
  label: string;
  detail: string;
  x: number;
  y: number;
  w: number;
  h: number;
  tone: "token" | "compute" | "attention" | "memory" | "output";
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
  const positionLabel =
    trace.modelProfileId === "llama-style" ? "RoPE on Q/K" : "Position add";
  const normLabel =
    trace.modelProfileId === "llama-style" ? "RMSNorm + residual" : "LayerNorm / residual";
  const mlpLabel =
    trace.modelProfileId === "llama-style" ? "SwiGLU MLP" : "GELU MLP";

  const nodes: NodeSpec[] = [
    {
      id: "tokenize",
      label: "Token ids",
      detail: stageById.get("tokenize")?.shape ?? "[batch, seq]",
      x: 54,
      y: 586,
      w: 168,
      h: 64,
      tone: "token",
    },
    {
      id: "embedding",
      label: "Embedding",
      detail: `${trace.hiddenSize} dims`,
      x: 286,
      y: 586,
      w: 178,
      h: 64,
      tone: "compute",
    },
    {
      id: "position",
      label: positionLabel,
      detail: stageById.get("position")?.subtitle ?? "position",
      x: 528,
      y: 586,
      w: 190,
      h: 64,
      tone: "compute",
    },
    {
      id: "qkv",
      label: "Q K V projections",
      detail: `${layer.heads} Q heads / ${layer.kvHeads} KV heads`,
      x: 274,
      y: 418,
      w: 220,
      h: 70,
      tone: "compute",
    },
    {
      id: "attention",
      label: "Masked attention",
      detail: `head ${layer.attentionHead}, scale ${layer.qkScale}`,
      x: 550,
      y: 396,
      w: 250,
      h: 96,
      tone: "attention",
    },
    {
      id: "residual",
      label: normLabel,
      detail: "write attention/MLP output",
      x: 552,
      y: 302,
      w: 246,
      h: 60,
      tone: "compute",
    },
    {
      id: "mlp",
      label: mlpLabel,
      detail: stageById.get("mlp")?.subtitle ?? "feed-forward",
      x: 552,
      y: 204,
      w: 246,
      h: 70,
      tone: "compute",
    },
    {
      id: "logits",
      label: "LM head",
      detail: `${trace.vocabSize.toLocaleString()} logits`,
      x: 510,
      y: 78,
      w: 206,
      h: 62,
      tone: "output",
    },
    {
      id: "sampling",
      label: "Selection",
      detail: `entropy ${decodeStep.entropy.toFixed(2)}`,
      x: 786,
      y: 78,
      w: 188,
      h: 62,
      tone: "output",
    },
    {
      id: "kvcache",
      label: "KV cache",
      detail: `seq ${decodeStep.kvCache.sequenceLength}`,
      x: 874,
      y: 366,
      w: 210,
      h: 126,
      tone: "memory",
    },
  ];

  const activeStage = stageById.get(activeStageId) ?? trace.stages[0];
  const promptTokens = trace.tokens.filter((token) => token.role === "prompt").length;

  return (
    <section className="panel architecture-panel">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Transformer Anatomy</span>
          <h2>Decoder-only 推理路径</h2>
        </div>
        <span className="metric">
          {trace.layerCount}L / d={trace.hiddenSize} / {trace.tokenizer}
        </span>
      </div>

      <div className="diagram-workbench">
        <svg
          className="transformer-schematic"
          viewBox="0 0 1128 700"
          role="img"
          aria-label={`${trace.title} transformer inference diagram`}
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
            <linearGradient id="residual-gradient" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stopColor="#2f6f73" />
              <stop offset="100%" stopColor="#a94b43" />
            </linearGradient>
          </defs>

          <rect className="diagram-bg" x="18" y="24" width="1092" height="652" rx="18" />
          <text className="diagram-title" x="44" y="64">
            {trace.title}
          </text>
          <text className="diagram-subtitle" x="44" y="90">
            prefill: {promptTokens} prompt tokens · decode: token {decodeStep.outputToken.text}
          </text>

          <g className="decoder-frame">
            <rect x="232" y="174" width="632" height="352" rx="20" />
            <text x="258" y="208">decoder block x {trace.layerCount}</text>
            <text x="258" y="232">selected layer: L{layer.index} · {layer.label}</text>
          </g>

          <path className="flow-line main-flow" d="M222 618 L286 618" />
          <path className="flow-line main-flow" d="M464 618 L528 618" />
          <path className="flow-line main-flow" d="M624 586 C624 536 386 536 386 488" />
          <path className="flow-line main-flow" d="M494 452 L550 452" />
          <path className="flow-line main-flow" d="M675 396 L675 362" />
          <path className="flow-line main-flow" d="M675 302 L675 274" />
          <path className="flow-line main-flow" d="M675 204 L675 140" />
          <path className="flow-line main-flow" d="M716 110 L786 110" />
          <path className="flow-line cache-flow" d="M800 438 L874 438" />
          <path className="flow-line cache-flow reverse-flow" d="M874 470 C834 514 762 510 718 492" />

          <path
            className="residual-rail"
            d="M182 548 C180 308 292 238 552 332"
          />
          <path
            className="residual-rail"
            d="M798 332 C940 304 936 168 716 110"
          />
          <path className="decode-loop" d="M974 110 C1052 150 1062 578 718 618" />

          {trace.tokens.map((token, index) => (
            <g
              className={`mini-token ${
                token.position === selectedToken.position ? "selected" : ""
              }`}
              key={`${token.id}-${token.position}`}
              transform={`translate(${46 + index * 58}, 650)`}
            >
              <rect width="48" height="24" rx="6" style={{ fill: token.color }} />
              <text x="24" y="16">
                {token.position}
              </text>
            </g>
          ))}

          {nodes.map((node) => (
            <DiagramNode
              key={node.id}
              node={node}
              selected={node.id === activeStageId}
              onSelect={onSelectStage}
            />
          ))}

          <g className="kv-shape">
            <text x="898" y="426">K {formatShape(decodeStep.kvCache.keyShape)}</text>
            <text x="898" y="454">V {formatShape(decodeStep.kvCache.valueShape)}</text>
            <text x="898" y="482">{decodeStep.kvCache.memoryMB.toFixed(2)} MB / layer</text>
          </g>
        </svg>

        <div className="diagram-readout">
          <span className="arch-kicker">Selected module</span>
          <h3>{activeStage.title}</h3>
          <p>{activeStage.description}</p>
          <code>{activeStage.formula}</code>
          <div className="readout-grid">
            <div>
              <span>Tensor shape</span>
              <strong>{activeStage.shape}</strong>
            </div>
            <div>
              <span>Selected token</span>
              <strong>{selectedToken.text} · pos {selectedToken.position}</strong>
            </div>
            <div>
              <span>Decode output</span>
              <strong>{decodeStep.outputToken.text} · id {decodeStep.selectedTokenId}</strong>
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
      <rect x={node.x} y={node.y} width={node.w} height={node.h} rx="10" />
      <text className="node-title" x={node.x + 18} y={node.y + 28}>
        {node.label}
      </text>
      <text className="node-detail" x={node.x + 18} y={node.y + node.h - 18}>
        {node.detail}
      </text>
    </g>
  );
}

function formatShape(shape: number[]) {
  return `[${shape.join(", ")}]`;
}
