import { AlertTriangle, Database, Info, Workflow } from "lucide-react";
import type { TokenInfo, TransformerStage } from "../types";

type Source = {
  token: TokenInfo;
  weight: number;
};

type InspectorPanelProps = {
  token: TokenInfo | undefined;
  stage: TransformerStage;
  sources: Source[];
  focus: string;
};

export function InspectorPanel({
  token,
  stage,
  sources,
  focus,
}: InspectorPanelProps) {
  const Icon = stage.icon;

  return (
    <aside className="panel inspector-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Inspector</span>
          <h2>当前剖面</h2>
        </div>
        <Info size={19} />
      </div>

      <div className="inspector-section token-inspector">
        <h3>Token</h3>
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
              <dd>{token.index}</dd>
            </div>
            <div>
              <dt>字节</dt>
              <dd>{token.bytes}</dd>
            </div>
          </dl>
        ) : (
          <p>暂无 token。</p>
        )}
      </div>

      <div className="inspector-section stage-inspector">
        <h3>
          <Icon size={17} />
          {stage.title}
        </h3>
        <p>{stage.detail}</p>
        <code>{stage.formula}</code>
      </div>

      <div className="inspector-section">
        <h3>
          <Workflow size={17} />
          Attention 来源
        </h3>
        <div className="source-list">
          {sources.map((source) => (
            <div className="source-row" key={`${source.token.text}-${source.token.index}`}>
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
          当前样例
        </h3>
        <p>{focus}</p>
      </div>

      <div className="notice">
        <AlertTriangle size={17} />
        <span>展示的是可解释教学模拟，不暴露真实隐藏思维链。</span>
      </div>
    </aside>
  );
}
