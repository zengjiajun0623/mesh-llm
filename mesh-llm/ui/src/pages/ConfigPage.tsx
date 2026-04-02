import { useCallback, useEffect, useReducer, useState } from "react";

export {
  configHistoryReducer,
  getAssignmentId,
  getSplitGroupId,
  type ConfigAction,
  type ConfigHistoryState,
} from "./config/configReducer";
import {
  configHistoryReducer,
  getAssignmentId,
  getSplitGroupId,
  type ConfigAction,
} from "./config/configReducer";

export {
  resizeSplitBoundaryInConfig,
  moveSplitAssignmentToNode,
  recombineSplitGroupInConfig,
  removeAssignmentFromConfig,
} from "../lib/configSplitOps";
import {
  resizeSplitBoundaryInConfig,
  moveSplitAssignmentToNode,
  recombineSplitGroupInConfig,
  removeAssignmentFromConfig,
  findAssignmentNodeId,
} from "../lib/configSplitOps";
import { getCrossNodeSplitGroupIds } from "../lib/configPageHelpers";
import { useConfigPageState } from "../hooks/useConfigPageState";
import { ConfigNodeList } from "../components/config/ConfigNodeList";

import type { StatusPayload } from "../App";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert";
import { ConfigErrorBoundary } from "../components/config/ConfigErrorBoundary";
import { DndContext } from "../components/config/DndContext";
import { EmptyNoNodes, NodeListSkeleton, CatalogSkeleton } from "../components/config/EmptyStates";
import { ConfigPageHeader } from "../components/config/ConfigPageHeader";
import { ConfigToolsPanel } from "../components/config/ConfigToolsPanel";
import { TomlEditor } from "../components/config/TomlEditor";
import { PlacementModeDialog } from "../components/config/PlacementModeDialog";
import { useOwnedNodes } from "../hooks/useOwnedNodes";
import { useConfigHydration } from "../hooks/useConfigHydration";
import { useConfigDirty } from "../hooks/useConfigDirty";
import { fetchAuthoredConfig, type ConfigValidationError } from "../lib/api";
import { clampConfigCtxSizes, createEmptyConfig, parseConfig } from "../lib/config";
import type { MeshConfig, ModelAssignment, ModelSplit, PlacementMode } from "../types/config";

export function ConfigPage({
  status,
  onRefreshStatus,
}: {
  status: StatusPayload | null;
  onRefreshStatus?: () => Promise<void> | void;
}) {
  const ownedNodes = useOwnedNodes(status);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedAssignmentId, setSelectedAssignmentId] = useState<string | null>(null);
  const [historyState, dispatch] = useReducer(configHistoryReducer, {
    config: createEmptyConfig(),
    past: [],
    future: [],
  });
  const config = historyState.config;
  const canUndo = historyState.past.length > 0;
  const canRedo = historyState.future.length > 0;
  const [savedConfig, setSavedConfig] = useState<MeshConfig>(createEmptyConfig());
  const [isConfigLoading, setIsConfigLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [tomlParseError, setTomlParseError] = useState<string | null>(null);
  const [assignmentErrors, setAssignmentErrors] = useState<Record<string, string>>({});
  const [backendValidationErrors, setBackendValidationErrors] = useState<ConfigValidationError[]>([]);
  const [pendingModeSwitch, setPendingModeSwitch] = useState<{
    nodeId: string;
    newMode: PlacementMode;
  } | null>(null);

  const isDirty = useConfigDirty({ config, savedConfig });

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        if (e.key === 'z' && !e.shiftKey) {
          e.preventDefault();
          dispatch({ type: "UNDO" });
        } else if (e.key === 'y' || (e.key === 'z' && e.shiftKey)) {
          e.preventDefault();
          dispatch({ type: "REDO" });
        }
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const handleSaveSuccess = useCallback((savedSnapshot: MeshConfig) => {
    setSavedConfig(savedSnapshot);
  }, []);

  const handleBackendErrors = useCallback((errors: ConfigValidationError[]) => {
    setBackendValidationErrors(errors);
  }, []);

  const handleRevert = useCallback(() => {
    dispatch({ type: 'SET_CONFIG', config: savedConfig });
    setAssignmentErrors({});
  }, [savedConfig]);

  useEffect(() => {
    let cancelled = false;

    const hydrateConfig = async () => {
      setIsConfigLoading(true);
      setLoadError(null);

      const result = await fetchAuthoredConfig();
      if (cancelled) return;

      if (!result.ok || !result.config) {
        setLoadError(result.error ?? "Failed to load authored config from /api/config");
        setIsConfigLoading(false);
        return;
      }

      const parsed = parseConfig(result.config);
      if (!parsed) {
        setLoadError("Invalid authored config returned by /api/config");
        setIsConfigLoading(false);
        return;
      }

      dispatch({ type: "SET_CONFIG", config: parsed });
      setSavedConfig(parsed);
      setAssignmentErrors({});
      setLoadError(null);
      setIsConfigLoading(false);
    };

    void hydrateConfig();
    return () => { cancelled = true; };
  }, []);

  useConfigHydration({ dispatch, isConfigLoading, ownedNodes, config, isDirty, setSavedConfig });

  const {
    ownershipNotice,
    ownedModelPeers,
    nodeConfigLookup,
    configAwareNodes,
    selectedNode,
    selectedCatalogNode,
    modelSizeLookup,
    advertisedModelKeysByNodeAndName,
    advertisedModelsByNode,
    vramAssignmentsByNode,
    assignedBytesByNode,
    totalVramByNode,
    catalogAssignTargets,
    allDropTargetsOvercommitted,
    splitValidationErrors,
    invalidReason,
    isConfigValid,
    selectedModelName,
    selectedAssignmentIds,
    nodeModelScansLookup,
    maxCtxByModel,
    nodeModelKeyLookup,
  } = useConfigPageState({
    status,
    config,
    selectedNodeId,
    selectedAssignmentId,
    ownedNodes,
    assignmentErrors,
    loadError,
    tomlParseError,
  });

  useEffect(() => {
    setSelectedNodeId((current) =>
      ownedNodes.some((node) => node.id === current)
        ? current
        : (ownedNodes[0]?.id ?? null),
    );
  }, [ownedNodes]);

  useEffect(() => {
    if (!selectedAssignmentId) return;

    const assignmentNodeId = findAssignmentNodeId(config, selectedAssignmentId);
    if (!assignmentNodeId) {
      setSelectedAssignmentId(null);
      return;
    }

    if (selectedNodeId !== assignmentNodeId) {
      setSelectedNodeId(assignmentNodeId);
    }
  }, [config, selectedAssignmentId, selectedNodeId]);

  const normalizeConfig = useCallback(
    (cfg: MeshConfig) => {
      const { config: clamped, clamped: wasClamped } = clampConfigCtxSizes(cfg, maxCtxByModel);
      return { config: clamped, modified: wasClamped };
    },
    [maxCtxByModel],
  );

  const handleAssignModel = useCallback(
    (modelName: string, _sizeBytes: number, nodeId: string) => {
      const gpuSuffixMatch = nodeId.match(/^(.+?)::gpu-(\d+)$/);
      const realNodeId = gpuSuffixMatch ? gpuSuffixMatch[1] : nodeId;
      const gpuIndex = gpuSuffixMatch
        ? Number.parseInt(gpuSuffixMatch[2], 10)
        : undefined;

      const modelKey = nodeModelKeyLookup.get(realNodeId)?.get(modelName);
      dispatch({
        type: "ASSIGN_MODEL",
        nodeId: realNodeId,
        model: { name: modelName, model_key: modelKey, gpu_index: gpuIndex, ctx_size: 4096 },
      });
      setAssignmentErrors({});
      setSelectedNodeId(realNodeId);
    },
    [nodeModelKeyLookup],
  );

  const handleUnassignModel = useCallback(
    (nodeId: string, modelName: string, assignmentId: string) => {
      const result = removeAssignmentFromConfig(config, { nodeId, assignmentId });
      const nextConfig = result.config;

      if (
        selectedModelName === modelName &&
        (!selectedAssignmentId || !findAssignmentNodeId(nextConfig, selectedAssignmentId))
      ) {
        if (result.replacementAssignmentId) {
          setSelectedAssignmentId(result.replacementAssignmentId);
          if (result.replacementNodeId) setSelectedNodeId(result.replacementNodeId);
        } else {
          setSelectedAssignmentId(null);
        }
      }

      dispatch({ type: 'SET_CONFIG', config: nextConfig });
      setAssignmentErrors({});
    },
    [config, selectedAssignmentId, selectedModelName],
  );

  const handleUpdateModel = useCallback(
    (nodeId: string, assignmentId: string, updates: Partial<ModelAssignment>) => {
      dispatch({ type: "UPDATE_MODEL", nodeId, assignmentId, updates });
      setAssignmentErrors({});
    },
    [],
  );

  const handleSplitModel = useCallback(
    (
      nodeId: string,
      modelName: string,
      blockA: { model_key: string; split: ModelSplit },
      blockB: { model_key: string; split: ModelSplit },
    ) => {
      const sourceAssignment = config.nodes
        .find((entry) => entry.node_id === nodeId)
        ?.models.find((model) => model.name === modelName && !model.split);
      const sharedAssignmentFields = sourceAssignment
        ? { ctx_size: sourceAssignment.ctx_size, moe_experts: sourceAssignment.moe_experts }
        : {};

      dispatch({ type: "UNASSIGN_MODEL", nodeId, modelName });
      dispatch({
        type: "ASSIGN_MODEL",
        nodeId,
        model: { name: modelName, model_key: blockA.model_key, ...sharedAssignmentFields, split: blockA.split },
      });
      dispatch({
        type: "ASSIGN_MODEL",
        nodeId,
        model: { name: modelName, model_key: blockB.model_key, ...sharedAssignmentFields, split: blockB.split },
      });

      setAssignmentErrors({});
      setSelectedNodeId(nodeId);
      setSelectedAssignmentId(
        getAssignmentId({ name: modelName, model_key: blockA.model_key, split: blockA.split }),
      );
    },
    [config],
  );

  const handleResizeSplitBoundary = useCallback(
    (nodeId: string, leftAssignmentId: string, rightAssignmentId: string, boundaryStart: number) => {
      const result = resizeSplitBoundaryInConfig(config, {
        nodeId, leftAssignmentId, rightAssignmentId, boundaryStart,
      });
      if (!result) return;

      dispatch({ type: "SET_CONFIG", config: result.config });
      setAssignmentErrors((current) => {
        const next = { ...current };
        delete next[leftAssignmentId];
        delete next[rightAssignmentId];
        return next;
      });

      if (selectedAssignmentId === leftAssignmentId) {
        setSelectedAssignmentId(result.leftAssignmentId);
      } else if (selectedAssignmentId === rightAssignmentId) {
        setSelectedAssignmentId(result.rightAssignmentId);
      }
    },
    [config, selectedAssignmentId],
  );

  const handleMoveSplitAssignment = useCallback(
    ({ assignmentId, sourceNodeId, targetNodeId }: {
      assignmentId: string; sourceNodeId: string; targetNodeId: string;
    }) => {
      const movedAssignment = config.nodes
        .flatMap((node) => node.models)
        .find((model) => getAssignmentId(model) === assignmentId);
      const movedGroupId = movedAssignment ? getSplitGroupId(movedAssignment) : null;

      const result = moveSplitAssignmentToNode(config, {
        assignmentId, sourceNodeId, targetNodeId,
        advertisedModelsByNode,
        advertisedModelKeysByNodeAndName,
      });

      if (!result.ok) {
        setAssignmentErrors((current) => ({ ...current, [assignmentId]: result.error }));
        setSelectedNodeId(sourceNodeId);
        setSelectedAssignmentId(assignmentId);
        return;
      }

      dispatch({ type: "SET_CONFIG", config: result.config });
      setAssignmentErrors((current) => {
        const next = { ...current };
        delete next[assignmentId];
        if (movedGroupId) delete next[movedGroupId];
        return next;
      });
      setSelectedNodeId(targetNodeId);
      setSelectedAssignmentId(result.assignmentId);
    },
    [advertisedModelKeysByNodeAndName, advertisedModelsByNode, config],
  );

  const handleRequestModeSwitch = useCallback(
    (nodeId: string, newMode: PlacementMode) => {
      if (newMode === "pooled") {
        setPendingModeSwitch({ nodeId, newMode });
      } else {
        dispatch({ type: "CLEAR_NODE_AND_SET_MODE", nodeId, mode: newMode });
        setAssignmentErrors({});
      }
    },
    [],
  );

  const handleConfirmModeSwitch = useCallback(() => {
    if (!pendingModeSwitch) return;
    const { nodeId, newMode } = pendingModeSwitch;
    dispatch({ type: "CLEAR_NODE_AND_SET_MODE", nodeId, mode: newMode });
    setAssignmentErrors({});
    setPendingModeSwitch(null);
  }, [pendingModeSwitch]);

  const handleCancelModeSwitch = useCallback(() => {
    setPendingModeSwitch(null);
  }, []);

  const handleConfigChange = useCallback((newConfig: MeshConfig) => {
    dispatch({ type: "SET_CONFIG", config: newConfig });
    setAssignmentErrors({});
  }, []);

  const handleRecombineGroup = useCallback(
    (nodeId: string, groupId: string) => {
      const result = recombineSplitGroupInConfig(config, { groupId, targetNodeId: nodeId });
      if (!result.ok) {
        setAssignmentErrors((current) => ({ ...current, [groupId]: result.error }));
        return;
      }

      dispatch({ type: "SET_CONFIG", config: result.config });
      setAssignmentErrors((current) => {
        const next = { ...current };
        delete next[groupId];
        return next;
      });
      setSelectedNodeId(nodeId);
      setSelectedAssignmentId(result.assignmentId);
    },
    [config],
  );

  return (
    <ConfigErrorBoundary>
      <div className="flex min-h-0 flex-col gap-6">
        <ConfigPageHeader
          config={config}
          savedConfig={savedConfig}
          isDirty={isDirty}
          isConfigValid={isConfigValid}
          invalidReason={invalidReason}
          isConfigLoading={isConfigLoading}
          loadError={loadError}
          onSaveSuccess={handleSaveSuccess}
          onRevert={handleRevert}
          onBackendErrors={handleBackendErrors}
          canUndo={canUndo}
          canRedo={canRedo}
          onUndo={() => dispatch({ type: "UNDO" })}
          onRedo={() => dispatch({ type: "REDO" })}
        />

        {backendValidationErrors
          .filter((err) => {
            const match = err.path.match(/^nodes\[(\d+)\]/);
            return !match;
          })
          .map((err) => (
            <p key={`be-unresolved::${err.code}::${err.path}`} className="text-destructive text-xs mt-1">
              {err.message}
            </p>
          ))}

        {ownershipNotice ? (
          <Alert
            data-testid="config-ownership-notice"
            variant={ownershipNotice.tone === "warning" ? "destructive" : "default"}
          >
            <AlertTitle>{ownershipNotice.title}</AlertTitle>
            <AlertDescription>{ownershipNotice.description}</AlertDescription>
          </Alert>
        ) : null}

        {configAwareNodes.length === 0 ? (
          <>
            <EmptyNoNodes />
            <TomlEditor
              config={config}
              onConfigChange={handleConfigChange}
              onParseErrorChange={setTomlParseError}
              onNormalizeConfig={normalizeConfig}
              className="max-h-[34rem]"
              panelClassName="max-h-[34rem]"
            />
          </>
        ) : (
          <DndContext
            selectedNodeId={selectedNodeId}
            onAssignModel={handleAssignModel}
            onMoveSplitAssignment={handleMoveSplitAssignment}
          >
            {isConfigLoading ? (
              <NodeListSkeleton />
            ) : (
              <ConfigNodeList
                config={config}
                configAwareNodes={configAwareNodes}
                vramAssignmentsByNode={vramAssignmentsByNode}
                nodeConfigLookup={nodeConfigLookup}
                nodeModelScansLookup={nodeModelScansLookup}
                nodeModelKeyLookup={nodeModelKeyLookup}
                modelSizeLookup={modelSizeLookup}
                splitValidationErrors={splitValidationErrors}
                backendValidationErrors={backendValidationErrors}
                crossNodeSplitGroupIds={getCrossNodeSplitGroupIds(config)}
                selectedNodeId={selectedNodeId}
                selectedAssignmentIds={selectedAssignmentIds}
                assignmentErrors={assignmentErrors}
                totalVramByNode={totalVramByNode}
                onSelectNode={setSelectedNodeId}
                onClearSelectedAssignment={() => setSelectedAssignmentId(null)}
                onRemoveModel={handleUnassignModel}
                onSelectAssignment={(nodeId, assignmentId) => {
                  setSelectedNodeId(nodeId);
                  setSelectedAssignmentId(assignmentId);
                }}
                onSplitModel={handleSplitModel}
                onRecombineGroup={handleRecombineGroup}
                onResizeSplitBoundary={handleResizeSplitBoundary}
                onUpdateModel={handleUpdateModel}
                onModeSwitch={handleRequestModeSwitch}
              />
            )}

            <PlacementModeDialog
              open={!!pendingModeSwitch}
              pendingNodeId={pendingModeSwitch?.nodeId}
              onConfirm={handleConfirmModeSwitch}
              onCancel={handleCancelModeSwitch}
            />

            {isConfigLoading ? (
              <CatalogSkeleton />
            ) : (
              <ConfigToolsPanel
                peers={ownedModelPeers}
                selectedCatalogNode={selectedCatalogNode}
                selectedNode={selectedNode}
                selectedNodeId={selectedNodeId}
                assignedBytesByNode={assignedBytesByNode}
                allDropTargetsLocked={allDropTargetsOvercommitted}
                config={config}
                onConfigChange={handleConfigChange}
                onTomlParseErrorChange={setTomlParseError}
                onNormalizeConfig={normalizeConfig}
                onRefreshStatus={onRefreshStatus}
                assignTargets={catalogAssignTargets}
                onAssignToNode={handleAssignModel}
              />
            )}
          </DndContext>
        )}
      </div>
    </ConfigErrorBoundary>
  );
}
