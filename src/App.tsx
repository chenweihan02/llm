import { useEffect, useMemo, useState } from "react";
import { AttentionMap } from "./components/AttentionMap";
import { ArchitectureDiagram } from "./components/ArchitectureDiagram";
import { ControlsPanel } from "./components/ControlsPanel";
import { InspectorPanel } from "./components/InspectorPanel";
import { OperatorDebugger } from "./components/OperatorDebugger";
import { PredictionPanel } from "./components/PredictionPanel";
import { TokenStrip } from "./components/TokenStrip";
import { modelProfiles } from "./data/modelProfiles";
import { traces } from "./data/traces";
import { clampOperatorIndex, operatorToStage } from "./lib/operatorDebug";
import type { InferenceTrace, StageId } from "./types";
import "./style.css";

export default function App() {
  const [importedTraces, setImportedTraces] = useState<InferenceTrace[]>([]);
  const [traceImportError, setTraceImportError] = useState("");
  const [selectedTraceId, setSelectedTraceId] = useState(traces[0].id);
  const [selectedLayerIndex, setSelectedLayerIndex] = useState(
    traces[0].layers[0].index,
  );
  const [selectedTokenPosition, setSelectedTokenPosition] = useState(
    traces[0].decodeSteps[0].inputPosition,
  );
  const [activeStageId, setActiveStageId] = useState<StageId>("attention");
  const [selectedDecodeIndex, setSelectedDecodeIndex] = useState(0);
  const [selectedOperatorIndex, setSelectedOperatorIndex] = useState(0);
  const [isOperatorPlaying, setIsOperatorPlaying] = useState(false);

  const allTraces = useMemo(
    () => [...traces, ...importedTraces],
    [importedTraces],
  );
  const selectedTrace =
    allTraces.find((trace) => trace.id === selectedTraceId) ?? allTraces[0];
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
  const selectedOperator =
    selectedLayer.operators[
      clampOperatorIndex(selectedOperatorIndex, selectedLayer.operators)
    ];
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
    if (!selectedLayer.operators[selectedOperatorIndex]) {
      setSelectedOperatorIndex(0);
      setIsOperatorPlaying(false);
    }
  }, [
    activeStageId,
    selectedDecodeIndex,
    selectedLayerIndex,
    selectedLayer.operators,
    selectedOperatorIndex,
    selectedTokenPosition,
    selectedTrace,
  ]);

  useEffect(() => {
    setSelectedOperatorIndex(0);
    setIsOperatorPlaying(false);
  }, [selectedLayer.index, selectedTrace.id]);

  useEffect(() => {
    if (!selectedOperator) return;
    setActiveStageId(operatorToStage(selectedOperator));
  }, [selectedOperator]);

  useEffect(() => {
    if (!isOperatorPlaying || selectedLayer.operators.length === 0) return;

    const timer = window.setInterval(() => {
      setSelectedOperatorIndex((currentIndex) => {
        const nextIndex = currentIndex + 1;

        if (nextIndex >= selectedLayer.operators.length) {
          setIsOperatorPlaying(false);
          return currentIndex;
        }

        return nextIndex;
      });
    }, 1200);

    return () => window.clearInterval(timer);
  }, [isOperatorPlaying, selectedLayer.operators.length]);

  function selectTrace(traceId: string) {
    const nextTrace =
      allTraces.find((trace) => trace.id === traceId) ?? allTraces[0];
    setSelectedTraceId(nextTrace.id);
    setSelectedLayerIndex(nextTrace.layers[0].index);
    setSelectedTokenPosition(nextTrace.decodeSteps[0].inputPosition);
    setSelectedDecodeIndex(0);
    setSelectedOperatorIndex(0);
    setIsOperatorPlaying(false);
    setActiveStageId("attention");
  }

  function importTrace(value: unknown) {
    try {
      const importedTrace = normalizeImportedTrace(value);
      setImportedTraces((currentTraces) => {
        const withoutDuplicate = currentTraces.filter(
          (trace) => trace.id !== importedTrace.id,
        );
        return [...withoutDuplicate, importedTrace];
      });
      setTraceImportError("");
      setSelectedTraceId(importedTrace.id);
      setSelectedLayerIndex(importedTrace.layers[0].index);
      setSelectedTokenPosition(importedTrace.decodeSteps[0].inputPosition);
      setSelectedDecodeIndex(0);
      setSelectedOperatorIndex(0);
      setIsOperatorPlaying(false);
      setActiveStageId("attention");
    } catch (error) {
      setTraceImportError(
        error instanceof Error ? error.message : "Trace JSON 解析失败。",
      );
    }
  }

  function selectOperator(operatorIndex: number) {
    setSelectedOperatorIndex(
      clampOperatorIndex(operatorIndex, selectedLayer.operators),
    );
  }

  return (
    <main className="app-shell">
      <div className="workspace">
        <ControlsPanel
          traces={allTraces}
          selectedTraceId={selectedTrace.id}
          selectedTrace={selectedTrace}
          selectedLayer={selectedLayer}
          selectedLayerIndex={selectedLayer.index}
          selectedDecodeIndex={selectedDecodeIndex}
          selectedOperatorIndex={selectedOperatorIndex}
          isOperatorPlaying={isOperatorPlaying}
          onTraceSelect={selectTrace}
          onLayerChange={setSelectedLayerIndex}
          onDecodeStepChange={setSelectedDecodeIndex}
          onSelectOperator={selectOperator}
          onPlayingChange={setIsOperatorPlaying}
          onTraceImport={importTrace}
          traceImportError={traceImportError}
        />

        <div className="main-lab">
          <ArchitectureDiagram
            trace={selectedTrace}
            layer={selectedLayer}
            decodeStep={selectedDecodeStep}
          selectedToken={selectedToken}
          activeStageId={activeStageId}
          selectedOperatorIndex={selectedOperatorIndex}
          onSelectStage={setActiveStageId}
          onSelectOperator={selectOperator}
        />
          <OperatorDebugger
            layer={selectedLayer}
            selectedOperatorIndex={selectedOperatorIndex}
            onSelectOperator={selectOperator}
            tokens={selectedTrace.tokens}
          />
          <details className="auxiliary-debug-panel">
            <summary>辅助探针：Token / Attention / Logits</summary>
            <TokenStrip
              tokens={selectedTrace.tokens}
              selectedTokenPosition={selectedToken.position}
              onSelect={setSelectedTokenPosition}
            />
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
          </details>
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

function normalizeImportedTrace(value: unknown): InferenceTrace {
  if (!value || typeof value !== "object") {
    throw new Error("Trace JSON 必须是对象。");
  }

  if ("error" in value) {
    throw new Error(String((value as { error: unknown }).error));
  }

  const trace = value as InferenceTrace;

  if (!Array.isArray(trace.layers) || trace.layers.length === 0) {
    throw new Error("Trace JSON 缺少 layers。");
  }
  if (!Array.isArray(trace.tokens) || trace.tokens.length === 0) {
    throw new Error("Trace JSON 缺少 tokens。");
  }
  if (!Array.isArray(trace.decodeSteps) || trace.decodeSteps.length === 0) {
    throw new Error("Trace JSON 缺少 decodeSteps。");
  }
  if (!Array.isArray(trace.stages) || trace.stages.length === 0) {
    throw new Error("Trace JSON 缺少 stages。");
  }

  return {
    ...trace,
    id: trace.id || `imported-${Date.now()}`,
    title: trace.title || "Imported HF trace",
    layers: trace.layers.map((layer) => ({
      ...layer,
      probes: Array.isArray(layer.probes) ? layer.probes : [],
      operators: Array.isArray(layer.operators) ? layer.operators : [],
    })),
  };
}
