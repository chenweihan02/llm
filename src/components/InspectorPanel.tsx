import {
  AlertTriangle,
  Boxes,
  Database,
  FunctionSquare,
  Info,
  Sigma,
  Workflow,
} from "lucide-react";
import type {
  DecodeStep,
  InferenceTrace,
  ModelProfile,
  TensorStats,
  TraceLayer,
  TraceStage,
  TraceToken,
} from "../types";
import { MathFormula } from "./MathFormula";

type Source = {
  token: TraceToken;
  weight: number;
};

type InspectorPanelProps = {
  trace: InferenceTrace;
  token: TraceToken | undefined;
  stage: TraceStage;
  layer: TraceLayer;
  decodeStep: DecodeStep;
  sources: Source[];
  profile: ModelProfile;
};

export function InspectorPanel({
  trace,
  token,
  stage,
  layer,
  decodeStep,
  sources,
  profile,
}: InspectorPanelProps) {
  return (
    <aside className="panel inspector-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Tensor Inspector</span>
          <h2>当前剖面</h2>
        </div>
        <Info size={19} />
      </div>

      <div className="inspector-section token-inspector">
        <h3>
          <Boxes size={17} />
          Token
        </h3>
        {token ? (
          <dl>
            <div>
              <dt>片段</dt>
              <dd>{token.text}</dd>
            </div>
            <div>
              <dt>ID</dt>
              <dd>{token.id}</dd>
            </div>
            <div>
              <dt>位置</dt>
              <dd>{token.position}</dd>
            </div>
            <div>
              <dt>向量范数</dt>
              <dd>{token.vectorNorm.toFixed(2)}</dd>
            </div>
          </dl>
        ) : (
          <p>暂无 token。</p>
        )}
      </div>

      <div className="inspector-section stage-inspector">
        <h3>
          <FunctionSquare size={17} />
          {stage.title}
        </h3>
        <p>{stage.description}</p>
        <div className="stage-formula">
          <MathFormula block latex={stage.latex ?? stage.formula} />
        </div>
        <span className="shape-line">{stage.shape}</span>
      </div>

      <div className="inspector-section">
        <h3>
          <Sigma size={17} />
          Layer L{layer.index} stats
        </h3>
        <StatsGrid hidden={layer.hidden} residual={layer.residual} mlp={layer.mlp} />
        <div className="probe-list">
          {layer.probes.map((probe) => (
            <div key={probe.name}>
              <span>{probe.name}</span>
              <code>[{probe.sample.map((value) => value.toFixed(2)).join(", ")}]</code>
            </div>
          ))}
        </div>
      </div>

      <div className="inspector-section">
        <h3>
          <Workflow size={17} />
          Attention 来源
        </h3>
        <div className="source-list">
          {sources.map((source) => (
            <div
              className="source-row"
              key={`${source.token.id}-${source.token.position}`}
            >
              <span>{source.token.text}</span>
              <meter min={0} max={1} value={source.weight} />
              <strong>{Math.round(source.weight * 100)}%</strong>
            </div>
          ))}
        </div>
      </div>

      <div className="inspector-section">
        <h3>
          <Database size={17} />
          KV Cache
        </h3>
        <div className="kv-list">
          <span>K {formatShape(decodeStep.kvCache.keyShape)}</span>
          <span>V {formatShape(decodeStep.kvCache.valueShape)}</span>
          <span>seq {decodeStep.kvCache.sequenceLength}</span>
          <span>{decodeStep.kvCache.memoryMB.toFixed(2)} MB / layer</span>
        </div>
        <p>{decodeStep.kvCache.update}</p>
      </div>

      <div className="inspector-section">
        <h3>
          <Database size={17} />
          模型族
        </h3>
        <p>{profile.summary}</p>
        <div className="model-spec-list">
          <span>Position: {profile.position}</span>
          <span>Norm: {profile.norm}</span>
          <span>MLP: {profile.mlp}</span>
          <span>KV cache: {profile.kvCache}</span>
        </div>
      </div>

      <div className="notice">
        <AlertTriangle size={17} />
        <span>
          {trace.source.type === "hf-export"
            ? `真实导出: ${trace.source.modelId}`
            : "内置 trace 是结构样例；用 scripts/export_trace.py 可替换为真实模型导出值。"}
        </span>
      </div>
    </aside>
  );
}

type StatsGridProps = {
  hidden: TensorStats;
  residual: TensorStats;
  mlp: TensorStats;
};

function StatsGrid({ hidden, residual, mlp }: StatsGridProps) {
  const rows = [
    ["hidden", hidden],
    ["residual", residual],
    ["mlp", mlp],
  ] as const;

  return (
    <div className="stats-grid">
      {rows.map(([label, stats]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>std {stats.std.toFixed(2)}</strong>
          <em>
            min {stats.min.toFixed(1)} / max {stats.max.toFixed(1)}
          </em>
        </div>
      ))}
    </div>
  );
}

function formatShape(shape: number[]) {
  return `[${shape.join(", ")}]`;
}
