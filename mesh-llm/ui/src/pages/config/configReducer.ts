import type { PlacementMode, ModelAssignment, MeshConfig } from "../../types/config";

export function getAssignmentId(
  assignment: Pick<ModelAssignment, "name" | "model_key" | "split" | "gpu_index">,
): string {
  const placement =
    assignment.gpu_index != null ? `gpu-${assignment.gpu_index}` : "pooled";
  if (assignment.split) {
    return `${assignment.name}::${assignment.model_key ?? ""}::${assignment.split.start}-${assignment.split.end}-${assignment.split.total}::${placement}`;
  }
  return `${assignment.name}::${placement}`;
}

export function getSplitGroupId(
  assignment: Pick<ModelAssignment, "name" | "model_key" | "split">,
): string | null {
  if (!assignment.split || !assignment.model_key) return null;
  return `${assignment.name}::${assignment.model_key}::${assignment.split.total}`;
}

export type ConfigAction =
  | { type: "ASSIGN_MODEL"; nodeId: string; model: ModelAssignment }
  | { type: "UNASSIGN_MODEL"; nodeId: string; modelName: string }
  | {
      type: "UPDATE_MODEL";
      nodeId: string;
      assignmentId: string;
      updates: Partial<ModelAssignment>;
    }
  | { type: "SET_CONFIG"; config: MeshConfig }
  | { type: "CLEAR_NODE_AND_SET_MODE"; nodeId: string; mode: PlacementMode }
  | { type: "BEGIN_BATCH" }
  | { type: "END_BATCH" }
  | { type: "UNDO" }
  | { type: "REDO" };

export function configReducer(state: MeshConfig, action: ConfigAction): MeshConfig {
  switch (action.type) {
    case "SET_CONFIG":
      return action.config;

    case "ASSIGN_MODEL": {
      const nodeIndex = state.nodes.findIndex(
        (n) => n.node_id === action.nodeId,
      );
      if (nodeIndex >= 0) {
        const node = state.nodes[nodeIndex];
        const isDuplicate = action.model.split
          ? false
          : node.models.some((m) => m.name === action.model.name && !m.split);
        if (isDuplicate) return state;
        const updatedNode = { ...node, models: [...node.models, action.model] };
        const nodes = [...state.nodes];
        nodes[nodeIndex] = updatedNode;
        return { ...state, nodes };
      }
      return {
        ...state,
        nodes: [
          ...state.nodes,
          { node_id: action.nodeId, models: [action.model] },
        ],
      };
    }

    case "UNASSIGN_MODEL": {
      const nodeIndex = state.nodes.findIndex(
        (n) => n.node_id === action.nodeId,
      );
      if (nodeIndex < 0) return state;
      const node = state.nodes[nodeIndex];
      const updatedNode = {
        ...node,
        models: node.models.filter((m) => m.name !== action.modelName),
      };
      const nodes = [...state.nodes];
      nodes[nodeIndex] = updatedNode;
      return { ...state, nodes };
    }

    case "UPDATE_MODEL": {
      const nodeIndex = state.nodes.findIndex(
        (n) => n.node_id === action.nodeId,
      );
      if (nodeIndex < 0) return state;
      const node = state.nodes[nodeIndex];
      const targetModel = node.models.find(
        (m) => getAssignmentId(m) === action.assignmentId,
      );
      if (!targetModel) return state;

      const targetGroupId = getSplitGroupId(targetModel);
      const updatedModels = node.models.map((model) => {
        const shouldUpdate = targetGroupId
          ? getSplitGroupId(model) === targetGroupId
          : getAssignmentId(model) === action.assignmentId;

        return shouldUpdate ? { ...model, ...action.updates } : model;
      });
      const updatedNode = { ...node, models: updatedModels };
      const nodes = [...state.nodes];
      nodes[nodeIndex] = updatedNode;
      return { ...state, nodes };
    }

    case "CLEAR_NODE_AND_SET_MODE": {
      const nodeIndex = state.nodes.findIndex(
        (n) => n.node_id === action.nodeId,
      );
      if (nodeIndex >= 0) {
        const nodes = [...state.nodes];
        nodes[nodeIndex] = {
          ...nodes[nodeIndex],
          models: [],
          placement_mode: action.mode,
        };
        return { ...state, nodes };
      }
      return {
        ...state,
        nodes: [
          ...state.nodes,
          { node_id: action.nodeId, models: [], placement_mode: action.mode },
        ],
      };
    }

    default:
      return state;
  }
}

export type ConfigHistoryState = {
  config: MeshConfig;
  past: MeshConfig[];
  future: MeshConfig[];
  batchSnapshot?: MeshConfig;
};

export function configHistoryReducer(
  state: ConfigHistoryState,
  action: ConfigAction,
): ConfigHistoryState {
  if (action.type === "BEGIN_BATCH") {
    if (state.batchSnapshot !== undefined) return state;
    return { ...state, batchSnapshot: state.config };
  }

  if (action.type === "END_BATCH") {
    const { batchSnapshot } = state;
    if (!batchSnapshot) return state;
    if (batchSnapshot === state.config) {
      return { ...state, batchSnapshot: undefined };
    }
    return {
      config: state.config,
      past: [...state.past, batchSnapshot].slice(-30),
      future: [],
      batchSnapshot: undefined,
    };
  }

  if (action.type === "UNDO") {
    const base = state.batchSnapshot ? { ...state, batchSnapshot: undefined } : state;
    if (base.past.length === 0) return base;
    const prev = base.past[base.past.length - 1];
    return {
      config: prev,
      past: base.past.slice(0, -1),
      future: [base.config, ...base.future],
    };
  }

  if (action.type === "REDO") {
    const base = state.batchSnapshot ? { ...state, batchSnapshot: undefined } : state;
    if (base.future.length === 0) return base;
    const next = base.future[0];
    return {
      config: next,
      past: [...base.past, base.config].slice(-30),
      future: base.future.slice(1),
    };
  }

  const newConfig = configReducer(state.config, action);
  if (newConfig === state.config) return state;

  if (state.batchSnapshot !== undefined) {
    return { ...state, config: newConfig, future: [] };
  }

  return {
    config: newConfig,
    past: [...state.past, state.config].slice(-30),
    future: [],
  };
}
