import { useEffect, useMemo, useState } from "react";
import { Brain, Code2, Sparkles } from "lucide-react";
import { AttentionMap } from "./components/AttentionMap";
import { ControlsPanel } from "./components/ControlsPanel";
import { InspectorPanel } from "./components/InspectorPanel";
import { PredictionPanel } from "./components/PredictionPanel";
import { TokenStrip } from "./components/TokenStrip";
import { TransformerFlow } from "./components/TransformerFlow";
import { examples, defaultInput } from "./data/examples";
import { transformerStages } from "./data/stages";
import { buildAttentionMatrix, topSources } from "./engines/attention";
import { predictNext, samplePrediction } from "./engines/sampler";
import { tokenize } from "./engines/tokenizer";
import "./style.css";

export default function App() {
  const [inputText, setInputText] = useState(defaultInput);
  const [selectedExampleId, setSelectedExampleId] = useState(examples[0].id);
  const [selectedTokenId, setSelectedTokenId] = useState(0);
  const [activeStageId, setActiveStageId] = useState("attention");
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(5);
  const [layer, setLayer] = useState(3);
  const [head, setHead] = useState(2);
  const [generatedTokens, setGeneratedTokens] = useState<string[]>([]);

  const fullText = `${inputText}${generatedTokens.join("")}`;
  const tokens = useMemo(() => tokenize(fullText), [fullText]);
  const matrix = useMemo(
    () => buildAttentionMatrix(tokens, layer, head),
    [tokens, layer, head],
  );
  const predictions = useMemo(
    () => predictNext(tokens, temperature, topK),
    [tokens, temperature, topK],
  );

  const selectedToken = tokens[selectedTokenId] ?? tokens[0];
  const activeStage =
    transformerStages.find((stage) => stage.id === activeStageId) ??
    transformerStages[0];
  const selectedExample =
    examples.find((example) => example.id === selectedExampleId) ?? examples[0];
  const sources = selectedToken
    ? topSources(tokens, matrix, selectedToken.index)
    : [];

  useEffect(() => {
    if (selectedTokenId >= tokens.length) {
      setSelectedTokenId(Math.max(0, tokens.length - 1));
    }
  }, [selectedTokenId, tokens.length]);

  function selectExample(exampleId: string) {
    const example = examples.find((item) => item.id === exampleId) ?? examples[0];
    setSelectedExampleId(example.id);
    setInputText(example.input);
    setGeneratedTokens([]);
    setSelectedTokenId(0);
  }

  function resetInput() {
    selectExample(examples[0].id);
    setTemperature(0.8);
    setTopK(5);
    setLayer(3);
    setHead(2);
    setActiveStageId("attention");
  }

  function generateStep() {
    const next = samplePrediction(predictions);
    if (!next) return;
    setGeneratedTokens((current) => [...current, next].slice(-18));
    setSelectedTokenId(tokens.length);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            <Brain size={22} />
          </span>
          <div>
            <p>LLM Dissection Lab</p>
            <h1>大模型可视化拆解实验室</h1>
          </div>
        </div>
        <div className="topbar-actions">
          <span className="status-pill">
            <Sparkles size={15} />
            Phase 1
          </span>
          <a
            className="github-link"
            href="https://github.com/"
            target="_blank"
            rel="noreferrer"
            title="GitHub"
          >
            <Code2 size={18} />
          </a>
        </div>
      </header>

      <div className="workspace">
        <ControlsPanel
          examples={examples}
          inputText={inputText}
          temperature={temperature}
          topK={topK}
          layer={layer}
          head={head}
          selectedExampleId={selectedExampleId}
          onInputChange={(value) => {
            setInputText(value);
            setGeneratedTokens([]);
            setSelectedTokenId(0);
          }}
          onExampleSelect={(example) => selectExample(example.id)}
          onTemperatureChange={setTemperature}
          onTopKChange={setTopK}
          onLayerChange={setLayer}
          onHeadChange={setHead}
          onReset={resetInput}
        />

        <div className="main-lab">
          <TokenStrip
            tokens={tokens}
            selectedTokenId={selectedTokenId}
            onSelect={setSelectedTokenId}
          />
          <TransformerFlow
            stages={transformerStages}
            activeStageId={activeStageId}
            onSelect={setActiveStageId}
          />
          <div className="visual-grid">
            <AttentionMap
              tokens={tokens}
              matrix={matrix}
              selectedTokenId={selectedTokenId}
              layer={layer}
              head={head}
              onSelect={setSelectedTokenId}
            />
            <PredictionPanel
              predictions={predictions}
              generatedTokens={generatedTokens}
              onGenerate={generateStep}
              onClear={() => {
                setGeneratedTokens([]);
                setSelectedTokenId(0);
              }}
            />
          </div>
        </div>

        <InspectorPanel
          token={selectedToken}
          stage={activeStage}
          sources={sources}
          focus={selectedExample.focus}
        />
      </div>
    </main>
  );
}
