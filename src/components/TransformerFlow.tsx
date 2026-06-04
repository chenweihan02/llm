import type { TransformerStage } from "../types";

type TransformerFlowProps = {
  stages: TransformerStage[];
  activeStageId: string;
  onSelect: (stageId: string) => void;
};

export function TransformerFlow({
  stages,
  activeStageId,
  onSelect,
}: TransformerFlowProps) {
  return (
    <section className="panel flow-panel">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Step 02</span>
          <h2>Transformer 剖面</h2>
        </div>
        <span className="metric">教学模拟</span>
      </div>

      <div className="stage-track">
        {stages.map((stage, index) => {
          const Icon = stage.icon;

          return (
            <button
              className={`stage-node ${
                stage.id === activeStageId ? "active" : ""
              }`}
              key={stage.id}
              onClick={() => onSelect(stage.id)}
            >
              <span className="stage-index">{String(index + 1).padStart(2, "0")}</span>
              <span className="stage-icon">
                <Icon size={18} />
              </span>
              <span className="stage-title">{stage.title}</span>
              <span className="stage-short">{stage.short}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
