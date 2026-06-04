import type { CSSProperties } from "react";
import type { TraceLayer, TraceToken } from "../types";

type AttentionMapProps = {
  tokens: TraceToken[];
  layer: TraceLayer;
  selectedTokenPosition: number;
  onSelect: (position: number) => void;
};

export function AttentionMap({
  tokens,
  layer,
  selectedTokenPosition,
  onSelect,
}: AttentionMapProps) {
  const selected = Math.min(selectedTokenPosition, tokens.length - 1);

  return (
    <section className="panel attention-panel">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Attention Probe</span>
          <h2>因果注意力矩阵</h2>
        </div>
        <span className="metric">
          L{layer.index} / H{layer.attentionHead}
        </span>
      </div>

      <div
        className="attention-grid"
        style={
          {
            "--token-count": tokens.length,
          } as CSSProperties
        }
      >
        <span className="corner-cell" />
        {tokens.map((token) => (
          <span className="axis-token top" key={`top-${token.position}`}>
            {token.text}
          </span>
        ))}

        {tokens.map((target) => (
          <Row
            key={`row-${target.position}`}
            target={target}
            tokens={tokens}
            weights={layer.attention[target.position] ?? []}
            selectedTokenId={selected}
            onSelect={onSelect}
          />
        ))}
      </div>

      <AttentionArc
        tokens={tokens}
        weights={layer.attention[selected] ?? []}
        selected={selected}
      />
    </section>
  );
}

type RowProps = {
  target: TraceToken;
  tokens: TraceToken[];
  weights: number[];
  selectedTokenId: number;
  onSelect: (position: number) => void;
};

function Row({ target, tokens, weights, selectedTokenId, onSelect }: RowProps) {
  return (
    <>
      <button
        className={`axis-token left ${
          target.position === selectedTokenId ? "active" : ""
        }`}
        onClick={() => onSelect(target.position)}
      >
        {target.text}
      </button>
      {tokens.map((source) => {
        const value = weights[source.position] ?? 0;
        return (
          <button
            className={`heat-cell ${
              target.position === selectedTokenId || source.position === selectedTokenId
                ? "emphasis"
                : ""
            }`}
            key={`${target.position}-${source.position}`}
            onClick={() => onSelect(target.position)}
            style={
              {
                "--heat": value,
              } as CSSProperties
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
  tokens: TraceToken[];
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
          const sourceX = token.position * step + step / 2;
          const weight = weights[token.position] ?? 0;
          if (!weight || token.position > selected) return null;

          const lift = 24 + Math.abs(selectedX - sourceX) * 0.16;
          const path = `M ${sourceX} ${centerY} Q ${
            (sourceX + selectedX) / 2
          } ${centerY - lift} ${selectedX} ${centerY}`;

          return (
            <path
              d={path}
              key={`arc-${token.position}`}
              opacity={0.25 + weight * 2}
              strokeWidth={1 + weight * 10}
            />
          );
        })}
        {tokens.map((token) => (
          <g
            className={token.position === selected ? "arc-token selected" : "arc-token"}
            key={`label-${token.position}`}
            transform={`translate(${token.position * step + step / 2}, ${centerY + 14})`}
          >
            <circle r="4" />
            <text y="22">{token.text}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}
