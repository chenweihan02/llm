import { useEffect, useMemo, useState } from "react";
import { Brain, Code2, Cpu } from "lucide-react";
import { AttentionMap } from "./components/AttentionMap";
import { ArchitectureDiagram } from "./components/ArchitectureDiagram";
import { ControlsPanel } from "./components/ControlsPanel";
import { InspectorPanel } from "./components/InspectorPanel";
import { OperatorDebugger } from "./components/OperatorDebugger";
import { PredictionPanel } from "./components/PredictionPanel";
import { TokenStrip } from "./components/TokenStrip";
import { modelProfiles } from "./data/modelProfiles";
import { traces } from "./data/traces";
import type { StageId } from "./types";
import "./style.css";

export default function App() {
  const [selectedTraceId, setSelectedTraceId] = useState(traces[0].id);
  const [selectedLayerIndex, setSelectedLayerIndex] = useState(
    traces[0].layers[0].index,
  );
  const [selectedTokenPosition, setSelectedTokenPosition] = useState(
    traces[0].decodeSteps[0].inputPosition,
  );
  const [activeStageId, setActiveStageId] = useState<StageId>("attention");
  const [selectedDecodeIndex, setSelectedDecodeIndex] = useState(0);

  const selectedTrace =
    traces.find((trace) => trace.id === selectedTraceId) ?? traces[0];
  const selectedLayer =
    selectedTrace.layers.find((layer) => layer.index === selectedLayerIndex) ??
    selectedTrace.layers[0];
  const selectedToken =
    selectedTrace.tokens.find((token) => token.position === selectedTokenPosition) ??
    selectedTrace.tokens[0];
  const activeStage =
    selectedTrace.stages.find((stage) => stage.id === activeStageId) ??
    selectedTrace.stages[0];
  const selectedDecodeStep =
    selectedTrace.decodeSteps[selectedDecodeIndex] ?? selectedTrace.decodeSteps[0];
  const selectedModel =
    modelProfiles.find((profile) => profile.id === selectedTrace.modelProfileId) ??
    modelProfiles[0];
  const attentionSources = useMemo(() => {
    const row = selectedLayer.attention[selectedToken.position] ?? [];
    return selectedTrace.tokens
      .map((token) => ({
        token,
        weight: row[token.position] ?? 0,
      }))
      .filter((source) => source.weight > 0)
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 5);
  }, [selectedLayer, selectedToken.position, selectedTrace.tokens]);

  useEffect(() => {
    if (!selectedTrace.layers.some((layer) => layer.index === selectedLayerIndex)) {
      setSelectedLayerIndex(selectedTrace.layers[0].index);
    }
    if (!selectedTrace.tokens.some((token) => token.position === selectedTokenPosition)) {
      setSelectedTokenPosition(selectedTrace.decodeSteps[0].inputPosition);
    }
    if (!selectedTrace.stages.some((stage) => stage.id === activeStageId)) {
      setActiveStageId("attention");
    }
    if (!selectedTrace.decodeSteps[selectedDecodeIndex]) {
      setSelectedDecodeIndex(0);
    }
  }, [
    activeStageId,
    selectedDecodeIndex,
    selectedLayerIndex,
    selectedTokenPosition,
    selectedTrace,
  ]);

  function selectTrace(traceId: string) {
    const nextTrace = traces.find((trace) => trace.id === traceId) ?? traces[0];
    setSelectedTraceId(nextTrace.id);
    setSelectedLayerIndex(nextTrace.layers[0].index);
    setSelectedTokenPosition(nextTrace.decodeSteps[0].inputPosition);
    setSelectedDecodeIndex(0);
    setActiveStageId("attention");
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            <Brain size={22} />
          </span>
          <div>
            <p>LLM Inference Anatomy</p>
            <h1>大模型推理解剖台</h1>
          </div>
        </div>
        <div className="topbar-actions">
          <span className="status-pill">
            <Cpu size={15} />
            Trace driven
          </span>
          <a
            className="github-link"
            href="https://github.com/chenweihan02/llm"
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
          traces={traces}
          selectedTraceId={selectedTrace.id}
          selectedTrace={selectedTrace}
          selectedLayerIndex={selectedLayer.index}
          selectedDecodeIndex={selectedDecodeIndex}
          modelProfiles={modelProfiles}
          selectedModel={selectedModel}
          onTraceSelect={selectTrace}
          onLayerChange={setSelectedLayerIndex}
          onDecodeStepChange={setSelectedDecodeIndex}
        />

        <div className="main-lab">
          <ArchitectureDiagram
            trace={selectedTrace}
            layer={selectedLayer}
            decodeStep={selectedDecodeStep}
            selectedToken={selectedToken}
            activeStageId={activeStageId}
            onSelectStage={setActiveStageId}
          />
          <TokenStrip
            tokens={selectedTrace.tokens}
            selectedTokenPosition={selectedToken.position}
            onSelect={setSelectedTokenPosition}
          />
          <OperatorDebugger layer={selectedLayer} />
          <div className="visual-grid">
            <AttentionMap
              tokens={selectedTrace.tokens}
              layer={selectedLayer}
              selectedTokenPosition={selectedToken.position}
              onSelect={setSelectedTokenPosition}
            />
            <PredictionPanel
              decodeSteps={selectedTrace.decodeSteps}
              selectedDecodeIndex={selectedDecodeIndex}
              onSelectDecodeStep={setSelectedDecodeIndex}
            />
          </div>
        </div>

        <InspectorPanel
          trace={selectedTrace}
          stage={activeStage}
          token={selectedToken}
          layer={selectedLayer}
          decodeStep={selectedDecodeStep}
          sources={attentionSources}
          profile={selectedModel}
        />
      </div>
    </main>
  );
}
