import type { CSSProperties } from "react";
import type { TraceToken } from "../types";

type TokenStripProps = {
  tokens: TraceToken[];
  selectedTokenPosition: number;
  onSelect: (position: number) => void;
};

export function TokenStrip({
  tokens,
  selectedTokenPosition,
  onSelect,
}: TokenStripProps) {
  return (
    <section className="panel token-panel">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Token Stream</span>
          <h2>输入与生成 token</h2>
        </div>
        <span className="metric">{tokens.length} traced tokens</span>
      </div>

      <div className="token-strip">
        {tokens.map((token) => (
          <button
            className={`token-chip ${
              token.position === selectedTokenPosition ? "selected" : ""
            }`}
            key={`${token.id}-${token.position}`}
            onClick={() => onSelect(token.position)}
            style={{ "--token-color": token.color } as CSSProperties}
            title={`token id: ${token.id}`}
          >
            <span className="token-text">{token.text}</span>
            <span className="token-meta">
              pos {token.position} · id {token.id}
            </span>
            <span className={`token-role ${token.role}`}>{token.role}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
