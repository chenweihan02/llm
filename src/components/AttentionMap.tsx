import type { TokenInfo } from "../types";

type AttentionMapProps = {
  tokens: TokenInfo[];
  matrix: number[][];
  selectedTokenId: number;
  layer: number;
  head: number;
  onSelect: (index: number) => void;
};

const maxVisibleTokens = 12;

export function AttentionMap({
  tokens,
  matrix,
  selectedTokenId,
  layer,
  head,
  onSelect,
}: AttentionMapProps) {
  const visibleTokens = tokens.slice(0, maxVisibleTokens);
  const selected = Math.min(selectedTokenId, visibleTokens.length - 1);

  return (
    <section className="panel attention-panel">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Step 03</span>
          <h2>Attention 热力图</h2>
        </div>
        <span className="metric">
          L{layer} / H{head}
        </span>
      </div>

      <div
        className="attention-grid"
        style={
          {
            "--token-count": visibleTokens.length,
          } as React.CSSProperties
        }
      >
        <span className="corner-cell" />
        {visibleTokens.map((token) => (
          <span className="axis-token top" key={`top-${token.index}`}>
            {token.text}
          </span>
        ))}

        {visibleTokens.map((target) => (
          <Row
            key={`row-${target.index}`}
            target={target}
            tokens={visibleTokens}
            weights={matrix[target.index] ?? []}
            selectedTokenId={selected}
            onSelect={onSelect}
          />
        ))}
      </div>

      <AttentionArc
        tokens={visibleTokens}
        weights={matrix[selected] ?? []}
        selected={selected}
      />
    </section>
  );
}

type RowProps = {
  target: TokenInfo;
  tokens: TokenInfo[];
  weights: number[];
  selectedTokenId: number;
  onSelect: (index: number) => void;
};

function Row({ target, tokens, weights, selectedTokenId, onSelect }: RowProps) {
  return (
    <>
      <button
        className={`axis-token left ${
          target.index === selectedTokenId ? "active" : ""
        }`}
        onClick={() => onSelect(target.index)}
      >
        {target.text}
      </button>
      {tokens.map((source) => {
        const value = weights[source.index] ?? 0;
        return (
          <button
            className={`heat-cell ${
              target.index === selectedTokenId || source.index === selectedTokenId
                ? "emphasis"
                : ""
            }`}
            key={`${target.index}-${source.index}`}
            onClick={() => onSelect(target.index)}
            style={
              {
                "--heat": value,
              } as React.CSSProperties
            }
            title={`${target.text} -> ${source.text}: ${Math.round(value * 100)}%`}
          >
            <span>{value > 0.1 ? Math.round(value * 100) : ""}</span>
          </button>
        );
      })}
    </>
  );
}

type AttentionArcProps = {
  tokens: TokenInfo[];
  weights: number[];
  selected: number;
};

function AttentionArc({ tokens, weights, selected }: AttentionArcProps) {
  if (tokens.length === 0) return null;

  const width = Math.max(520, tokens.length * 58);
  const height = 112;
  const step = width / Math.max(tokens.length, 1);
  const centerY = 86;
  const selectedX = selected * step + step / 2;

  return (
    <div className="arc-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        {tokens.map((token) => {
          const sourceX = token.index * step + step / 2;
          const weight = weights[token.index] ?? 0;
          if (!weight || token.index > selected) return null;

          const lift = 24 + Math.abs(selectedX - sourceX) * 0.16;
          const path = `M ${sourceX} ${centerY} Q ${
            (sourceX + selectedX) / 2
          } ${centerY - lift} ${selectedX} ${centerY}`;

          return (
            <path
              d={path}
              key={`arc-${token.index}`}
              opacity={0.25 + weight * 2}
              strokeWidth={1 + weight * 10}
            />
          );
        })}
        {tokens.map((token) => (
          <g
            className={token.index === selected ? "arc-token selected" : "arc-token"}
            key={`label-${token.index}`}
            transform={`translate(${token.index * step + step / 2}, ${centerY + 14})`}
          >
            <circle r="4" />
            <text y="22">{token.text}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}
