import { getAssignmentId, getSplitGroupId } from "../../pages/config/configReducer";
import type { AggregatedModel } from "../../lib/models";
import type { ConfigValidationError } from "../../lib/api";
import type { OwnedNode, NodeStatusTone } from "../../hooks/useOwnedNodes";
import type { MeshConfig, ModelAssignment, ModelSplit, PlacementMode, ScannedModelMetadata } from "../../types/config";
import type { VramAssignment } from "./VramContainer";
import { ConfigNodeSection } from "./ConfigNodeSection";

type ConfigAwareNode = OwnedNode & { statusLabel: string; statusTone: NodeStatusTone };

type SplitValidationError = { path: string; code: string; message: string };

type ConfigNodeListProps = {
  config: MeshConfig;
  configAwareNodes: ConfigAwareNode[];
  vramAssignmentsByNode: Map<string, VramAssignment[]>;
  nodeConfigLookup: Map<string, MeshConfig["nodes"][number]>;
  nodeModelScansLookup: Map<string, Map<string, ScannedModelMetadata>>;
  nodeModelKeyLookup: Map<string, Map<string, string>>;
  modelSizeLookup: Map<string, AggregatedModel>;
  splitValidationErrors: SplitValidationError[];
  backendValidationErrors: ConfigValidationError[];
  crossNodeSplitGroupIds: Set<string>;
  selectedNodeId: string | null;
  selectedAssignmentIds: string[];
  assignmentErrors: Record<string, string>;
  totalVramByNode: Map<string, number>;
  onSelectNode: (nodeId: string) => void;
  onClearSelectedAssignment: () => void;
  onRemoveModel: (nodeId: string, modelName: string, assignmentId: string) => void;
  onSelectAssignment: (nodeId: string, assignmentId: string) => void;
  onSplitModel: (
    nodeId: string,
    modelName: string,
    blockA: { model_key: string; split: ModelSplit },
    blockB: { model_key: string; split: ModelSplit },
  ) => void;
  onRecombineGroup: (nodeId: string, groupId: string) => void;
  onResizeSplitBoundary: (
    nodeId: string,
    leftAssignmentId: string,
    rightAssignmentId: string,
    boundaryStart: number,
  ) => void;
  onUpdateModel: (nodeId: string, assignmentId: string, updates: Partial<ModelAssignment>) => void;
  onModeSwitch: (nodeId: string, newMode: PlacementMode) => void;
  onBeginBatch?: () => void;
  onEndBatch?: () => void;
};

export function ConfigNodeList({
  config,
  configAwareNodes,
  vramAssignmentsByNode,
  nodeConfigLookup,
  nodeModelScansLookup,
  nodeModelKeyLookup,
  modelSizeLookup,
  splitValidationErrors,
  backendValidationErrors,
  crossNodeSplitGroupIds,
  selectedNodeId,
  selectedAssignmentIds,
  assignmentErrors,
  totalVramByNode,
  onSelectNode,
  onClearSelectedAssignment,
  onRemoveModel,
  onSelectAssignment,
  onSplitModel,
  onRecombineGroup,
  onResizeSplitBoundary,
  onUpdateModel,
  onModeSwitch,
  onBeginBatch,
  onEndBatch,
}: ConfigNodeListProps) {
  return (
    <div data-testid="config-node-sections" className="space-y-4">
      {configAwareNodes.map((node) => {
        const assignments = vramAssignmentsByNode.get(node.id) ?? [];
        const nodeSelectedAssignmentId =
          selectedAssignmentIds.find((assignmentId) =>
            assignments.some((assignment) => assignment.id === assignmentId),
          ) ?? null;
        const nodeSelectedAssignment = nodeSelectedAssignmentId
          ? (nodeConfigLookup
              .get(node.id)
              ?.models.find(
                (model) => getAssignmentId(model) === nodeSelectedAssignmentId,
              ) ?? null)
          : null;
        const nodeSelectedModelName = nodeSelectedAssignment?.name ?? null;
        const nodeSelectedMetadata = nodeSelectedModelName
          ? (nodeModelScansLookup.get(node.id)?.get(nodeSelectedModelName) ?? null)
          : null;
        const nodeSelectedAggregated = nodeSelectedModelName
          ? (modelSizeLookup.get(nodeSelectedModelName) ?? null)
          : null;
        const selectedGroupId = nodeSelectedAssignment
          ? getSplitGroupId(nodeSelectedAssignment)
          : null;
        const recombineError = selectedGroupId
          ? assignmentErrors[selectedGroupId]
          : null;
        const currentPlacementMode: PlacementMode =
          nodeConfigLookup.get(node.id)?.placement_mode ?? "pooled";

        const nodeConfigIdx = config.nodes.findIndex((n) => n.node_id === node.id);
        const nodeErrors = splitValidationErrors.filter((e) =>
          e.path.startsWith(`nodes[${nodeConfigIdx}]`),
        );

        return (
          <div key={node.id} className="space-y-2">
            <ConfigNodeSection
              node={node}
              isSelected={selectedNodeId === node.id}
              placementMode={currentPlacementMode}
              totalVramBytes={totalVramByNode.get(node.id) ?? Math.round(node.vramGb * 1e9)}
              assignments={assignments}
              selectedAssignmentId={nodeSelectedAssignmentId}
              selectedAssignmentIds={selectedAssignmentIds}
              selectedAssignment={nodeSelectedAssignment}
              selectedAggregated={nodeSelectedAggregated}
              selectedMetadata={nodeSelectedMetadata}
              selectedGroupId={selectedGroupId}
              recombineError={recombineError}
              modelScansLookup={nodeModelScansLookup.get(node.id)}
              modelKeyLookup={nodeModelKeyLookup.get(node.id)}
              availableNodeCount={configAwareNodes.length}
              crossNodeSplitGroupIds={crossNodeSplitGroupIds}
              onSelectNode={onSelectNode}
              onClearSelectedAssignment={onClearSelectedAssignment}
              onRemoveModel={(modelName, assignmentId) =>
                onRemoveModel(node.id, modelName, assignmentId)
              }
              onSelectAssignment={(assignmentId) =>
                onSelectAssignment(node.id, assignmentId)
              }
              onSplitModel={(modelName, blockA, blockB) =>
                onSplitModel(node.id, modelName, blockA, blockB)
              }
              onRecombineGroup={(groupId) => onRecombineGroup(node.id, groupId)}
              onResizeSplitBoundary={(leftAssignmentId, rightAssignmentId, boundaryStart) =>
                onResizeSplitBoundary(node.id, leftAssignmentId, rightAssignmentId, boundaryStart)
              }
              onUpdateModel={(assignmentId, updates) =>
                onUpdateModel(node.id, assignmentId, updates)
              }
              onBeginBatch={onBeginBatch}
              onEndBatch={onEndBatch}
              onPlacementModeChange={(newMode) => {
                if (newMode !== currentPlacementMode) {
                  onModeSwitch(node.id, newMode);
                }
              }}
            />
            {nodeErrors.map((err) => (
              <p key={`${err.code}::${err.path}`} className="text-destructive text-xs">
                {err.message}
              </p>
            ))}
            {backendValidationErrors
              .filter((err) => {
                const match = err.path.match(/^nodes\[(\d+)\]/);
                const nodeIdx = match ? parseInt(match[1], 10) : -1;
                return nodeIdx === nodeConfigIdx;
              })
              .map((err) => (
                <p key={`be::${err.code}::${err.path}`} className="text-destructive text-xs mt-1">
                  {err.message}
                </p>
              ))}
          </div>
        );
      })}
    </div>
  );
}
