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
  selectedOperatorIndex: number;
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
  selectedOperatorIndex,
  onSelectStage,
}: ArchitectureDiagramProps) {
  const stageById = new Map(trace.stages.map((stage) => [stage.id, stage]));
  const isLlama = trace.modelProfileId === "llama-style";
  const normLabel = isLlama ? "RMSNorm" : "LayerNorm";
  const positionLabel = isLlama ? "RoPE in Q/K" : "Position Encoding";
  const mlpLabel = isLlama ? "SwiGLU Feed Forward" : "GELU Feed Forward";
  const attentionLabel = "Masked Multi-Head Attention";
  const activeStage = stageById.get(activeStageId) ?? trace.stages[0];
  const selectedOperator = layer.operators[selectedOperatorIndex];
  const selectedOperatorId = selectedOperator?.id ?? "";

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
      id: "kvcache",
      label: "KV Cache",
      detail: `${formatShape(layer.kvCache.keyShape)} / ${decodeStep.kvCache.memoryMB.toFixed(2)} MB`,
      x: 805,
      y: 242,
      w: 230,
      h: 48,
      tone: "cache",
    },
  ];

  return (
    <section className="panel architecture-panel">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Transformer Flow</span>
          <h2>Decoder-only Transformer 架构</h2>
        </div>
        <span className="metric">
          L{layer.index} / {trace.layerCount} layers / hidden {trace.hiddenSize}
        </span>
      </div>

      <div className="diagram-workbench">
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
          <path className="flow-line cache-flow" d="M715 542 C748 504 744 324 812 274" />
          <path className="flow-line cache-flow reverse-flow" d="M912 288 C790 318 744 486 715 536" />

          <g className={`residual-network ${activeStageId === "residual" ? "residual-active" : ""}`}>
            <path className="residual-bypass" d="M560 696 H386 Q378 696 378 688 V494 Q378 486 386 486 H548" />
            <path className="residual-bypass" d="M560 392 H744 Q752 392 752 384 V294 Q752 286 744 286 H572" />
          </g>

          <path className="detail-leader" d="M715 525 L760 344" />
          <path className="detail-leader" d="M715 572 L760 624" />
          <path className="detail-leader" d="M425 326 L345 280" />
          <path className="detail-leader" d="M425 358 L345 444" />

          <MlpDetailPanel
            activeStageId={activeStageId}
            isLlama={isLlama}
            onSelectStage={onSelectStage}
            selectedOperatorId={selectedOperatorId}
          />

          <AttentionDetailPanel
            activeStageId={activeStageId}
            cacheShape={formatShape(layer.kvCache.keyShape)}
            cacheSize={`${decodeStep.kvCache.memoryMB.toFixed(2)} MB`}
            isLlama={isLlama}
            onSelectStage={onSelectStage}
            selectedOperatorId={selectedOperatorId}
          />

          {nodes.map((node, index) => (
            <DiagramNode
              key={`${node.id}-${node.label}-${index}`}
              node={node}
              selected={node.id === activeStageId}
              onSelect={onSelectStage}
            />
          ))}

          <g className={`residual-network residual-ports ${activeStageId === "residual" ? "residual-active" : ""}`}>
            <ResidualBranchPort x={560} y={696} />
            <ResidualMergePort x={560} y={486} />
            <ResidualBranchPort x={560} y={392} />
            <ResidualMergePort x={560} y={286} />
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

function ResidualBranchPort({ x, y }: { x: number; y: number }) {
  return <circle className="residual-branch-port" cx={x} cy={y} r="5.5" />;
}

function ResidualMergePort({ x, y }: { x: number; y: number }) {
  return (
    <g className="residual-merge-port">
      <circle cx={x} cy={y} r="11" />
      <path d={`M${x - 5} ${y} L${x + 5} ${y}`} />
      <path d={`M${x} ${y - 5} L${x} ${y + 5}`} />
    </g>
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

type DetailStepSpec = {
  label: string;
  detail: string;
  x: number;
  y: number;
  w: number;
  h: number;
  stageId: StageId;
  operatorMatches: string[];
};

type DetailPanelProps = {
  activeStageId: StageId;
  isLlama: boolean;
  onSelectStage: (stageId: StageId) => void;
  selectedOperatorId: string;
};

function AttentionDetailPanel({
  activeStageId,
  cacheShape,
  cacheSize,
  isLlama,
  onSelectStage,
  selectedOperatorId,
}: DetailPanelProps & { cacheShape: string; cacheSize: string }) {
  const steps: DetailStepSpec[] = [
    {
      label: isLlama ? "RMSNorm" : "LayerNorm",
      detail: "pre-attn",
      x: 790,
      y: 360,
      w: 100,
      h: 42,
      stageId: "residual",
      operatorMatches: ["rms-attn", "ln-attn"],
    },
    {
      label: isLlama ? "Q + K/V" : "Q/K/V",
      detail: isLlama ? "GQA proj" : "linear proj",
      x: 925,
      y: 360,
      w: 106,
      h: 42,
      stageId: "qkv",
      operatorMatches: ["q-proj", "kv-proj", "qkv-proj"],
    },
    {
      label: isLlama ? "RoPE" : "KV write",
      detail: isLlama ? "rotate Q/K" : cacheSize,
      x: 925,
      y: 420,
      w: 106,
      h: 42,
      stageId: isLlama ? "position" : "kvcache",
      operatorMatches: isLlama ? ["rope"] : ["kv-write"],
    },
    {
      label: "Score",
      detail: "scaled matmul",
      x: 790,
      y: 440,
      w: 100,
      h: 42,
      stageId: "attention",
      operatorMatches: ["qk-score"],
    },
    {
      label: "Mask",
      detail: "causal",
      x: 925,
      y: 480,
      w: 106,
      h: 42,
      stageId: "attention",
      operatorMatches: ["causal-mask"],
    },
    {
      label: "Softmax",
      detail: "weights",
      x: 790,
      y: 520,
      w: 100,
      h: 42,
      stageId: "attention",
      operatorMatches: ["attn-softmax"],
    },
    {
      label: "Value read",
      detail: "A x V",
      x: 925,
      y: 540,
      w: 106,
      h: 42,
      stageId: "attention",
      operatorMatches: ["value-read"],
    },
    {
      label: "Output proj",
      detail: "W_o writeback",
      x: 855,
      y: 616,
      w: 118,
      h: 42,
      stageId: "attention",
      operatorMatches: ["attn-out", "residual-attn"],
    },
  ];

  return (
    <g className="detail-panel attention-detail-panel">
      <rect className="detail-panel-frame" x="760" y="300" width="300" height="376" rx="14" />
      <text className="detail-panel-title" x="785" y="326">
        Attention internals
      </text>
      <g className="kv-cache-badge">
        <rect x="912" y="310" width="126" height="34" rx="8" />
        <text className="kv-cache-badge-title" x="925" y="324">
          KV Cache
        </text>
        <text className="kv-cache-badge-shape" x="925" y="339">
          {cacheShape}
        </text>
      </g>
      <path className="detail-flow" d="M890 381 L925 381" />
      <path className="detail-flow" d="M978 402 L978 420" />
      <path className="detail-flow" d="M925 441 L890 461" />
      <path className="detail-flow" d="M890 461 L925 501" />
      <path className="detail-flow" d="M925 501 L890 541" />
      <path className="detail-flow" d="M890 541 L925 561" />
      <path className="detail-flow" d="M978 582 L914 616" />
      {steps.map((step) => (
        <DetailStepNode
          activeStageId={activeStageId}
          key={`${step.label}-${step.x}-${step.y}`}
          onSelectStage={onSelectStage}
          selectedOperatorId={selectedOperatorId}
          step={step}
        />
      ))}
    </g>
  );
}

function MlpDetailPanel({
  activeStageId,
  isLlama,
  onSelectStage,
  selectedOperatorId,
}: DetailPanelProps) {
  const steps: DetailStepSpec[] = [
    {
      label: isLlama ? "RMSNorm" : "LayerNorm",
      detail: "pre-mlp",
      x: 92,
      y: 284,
      w: 98,
      h: 40,
      stageId: "residual",
      operatorMatches: ["rms-mlp", "ln-mlp"],
    },
    {
      label: isLlama ? "Gate / Up" : "Up proj",
      detail: isLlama ? "two paths" : "d -> 4d",
      x: 218,
      y: 284,
      w: 104,
      h: 40,
      stageId: "mlp",
      operatorMatches: ["swiglu", "mlp-up"],
    },
    {
      label: isLlama ? "SiLU x Up" : "GELU",
      detail: isLlama ? "multiply" : "nonlinear",
      x: 92,
      y: 352,
      w: 98,
      h: 40,
      stageId: "mlp",
      operatorMatches: ["swiglu", "mlp-up"],
    },
    {
      label: "Down proj",
      detail: "back to d",
      x: 218,
      y: 352,
      w: 104,
      h: 40,
      stageId: "mlp",
      operatorMatches: ["down-proj", "mlp-down"],
    },
    {
      label: "Residual add",
      detail: "writeback",
      x: 154,
      y: 424,
      w: 108,
      h: 40,
      stageId: "residual",
      operatorMatches: ["residual-mlp"],
    },
  ];

  return (
    <g className="detail-panel mlp-detail-panel">
      <rect className="detail-panel-frame" x="58" y="238" width="286" height="242" rx="14" />
      <text className="detail-panel-title" x="84" y="264">
        MLP internals
      </text>
      <path className="detail-flow" d="M190 304 L218 304" />
      <path className="detail-flow" d="M270 324 L270 352" />
      <path className="detail-flow" d="M218 372 L190 372" />
      <path className="detail-flow" d="M141 392 L180 424" />
      <path className="detail-flow" d="M270 392 L235 424" />
      {steps.map((step) => (
        <DetailStepNode
          activeStageId={activeStageId}
          key={`${step.label}-${step.x}-${step.y}`}
          onSelectStage={onSelectStage}
          selectedOperatorId={selectedOperatorId}
          step={step}
        />
      ))}
    </g>
  );
}

type DetailStepNodeProps = {
  activeStageId: StageId;
  onSelectStage: (stageId: StageId) => void;
  selectedOperatorId: string;
  step: DetailStepSpec;
};

function DetailStepNode({
  activeStageId,
  onSelectStage,
  selectedOperatorId,
  step,
}: DetailStepNodeProps) {
  const operatorActive = step.operatorMatches.some((match) =>
    selectedOperatorId.includes(match),
  );
  const stageRelated = step.stageId === activeStageId;
  const labelY = step.y + Math.round(step.h * 0.44);
  const detailY = step.y + Math.round(step.h * 0.78);

  return (
    <g
      className={`detail-node ${stageRelated ? "stage-related" : ""} ${
        operatorActive ? "operator-active" : ""
      }`}
      onClick={() => onSelectStage(step.stageId)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          onSelectStage(step.stageId);
        }
      }}
      role="button"
      tabIndex={0}
    >
      <rect x={step.x} y={step.y} width={step.w} height={step.h} rx="7" />
      <text className="detail-node-label" x={step.x + step.w / 2} y={labelY}>
        {step.label}
      </text>
      <text className="detail-node-detail" x={step.x + step.w / 2} y={detailY}>
        {step.detail}
      </text>
    </g>
  );
}
