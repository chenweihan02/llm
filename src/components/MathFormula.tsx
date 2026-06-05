import { useLayoutEffect, useMemo, useRef, useState } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

type MathFormulaProps = {
  latex: string;
  block?: boolean;
};

export function MathFormula({ latex, block = false }: MathFormulaProps) {
  const outerRef = useRef<HTMLSpanElement>(null);
  const innerRef = useRef<HTMLSpanElement>(null);
  const [scale, setScale] = useState(1);

  const html = useMemo(() => {
    try {
      return katex.renderToString(latex, {
        displayMode: block,
        output: "html",
        throwOnError: false,
        trust: false,
      });
    } catch {
      return escapeHtml(latex);
    }
  }, [block, latex]);

  useLayoutEffect(() => {
    if (!block) return;

    const outer = outerRef.current;
    const inner = innerRef.current;
    if (!outer || !inner) return;

    let frame = 0;
    const measure = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const availableWidth = outer.clientWidth;
        const naturalWidth = inner.scrollWidth;

        if (!availableWidth || !naturalWidth) return;

        const safeWidth = Math.max(1, availableWidth - 1);
        const nextScale = Math.min(1, safeWidth / naturalWidth);

        setScale((current) =>
          Math.abs(current - nextScale) < 0.001 ? current : nextScale,
        );
      });
    };

    measure();

    const observer = new ResizeObserver(measure);
    observer.observe(outer);

    document.fonts?.ready.then(measure).catch(() => undefined);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [block, html]);

  return (
    <span
      ref={outerRef}
      className={block ? "math-formula math-formula-block" : "math-formula"}
    >
      <span
        ref={innerRef}
        className="math-formula-content"
        style={block ? { transform: `scale(${scale})` } : undefined}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </span>
  );
}

function escapeHtml(value: string) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
