import type { TokenInfo } from "../types";

type TokenStripProps = {
  tokens: TokenInfo[];
  selectedTokenId: number;
  onSelect: (index: number) => void;
};

export function TokenStrip({
  tokens,
  selectedTokenId,
  onSelect,
}: TokenStripProps) {
  return (
    <section className="panel token-panel">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Step 01</span>
          <h2>Token 流</h2>
        </div>
        <span className="metric">{tokens.length} tokens</span>
      </div>

      <div className="token-strip">
        {tokens.map((token) => (
          <button
            className={`token-chip ${
              token.index === selectedTokenId ? "selected" : ""
            }`}
            key={`${token.text}-${token.index}`}
            onClick={() => onSelect(token.index)}
            style={{ "--token-color": token.color } as React.CSSProperties}
            title={`token id: ${token.id}`}
          >
            <span className="token-text">{token.text}</span>
            <span className="token-meta">#{token.index}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
