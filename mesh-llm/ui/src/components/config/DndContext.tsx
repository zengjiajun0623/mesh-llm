import { configure } from '@dnd-kit/abstract';
import { DragDropProvider, KeyboardSensor, PointerSensor } from '@dnd-kit/react';
import { useMemo, type ReactNode } from 'react';

type DragModelData = {
  type: 'model';
  modelName: string;
  sizeBytes: number;
  nodeIds: string[];
};

type DragSplitAssignmentData = {
  type: 'split-assignment';
  assignmentId: string;
  sourceNodeId: string;
  modelName: string;
  sizeBytes?: number;
};

function isDragModelData(data: unknown): data is DragModelData {
  if (!data || typeof data !== 'object') return false;
  const record = data as Record<string, unknown>;
  return record.type === 'model' && typeof record.modelName === 'string' && typeof record.sizeBytes === 'number';
}

function isDragSplitAssignmentData(data: unknown): data is DragSplitAssignmentData {
  if (!data || typeof data !== 'object') return false;
  const record = data as Record<string, unknown>;
  return (
    record.type === 'split-assignment'
    && typeof record.assignmentId === 'string'
    && typeof record.sourceNodeId === 'string'
    && typeof record.modelName === 'string'
  );
}

type DndContextProps = {
  children: ReactNode;
  selectedNodeId: string | null;
  onAssignModel?: (modelName: string, sizeBytes: number, nodeId: string) => void;
  onMoveSplitAssignment?: (move: { assignmentId: string; sourceNodeId: string; targetNodeId: string }) => void;
};

export const DRAG_INTERACTIVE_ATTRIBUTE = 'data-dnd-interactive';
export const VRAM_DROP_TARGET_PREFIX = 'vram-container:';
export const CONFIG_NODE_DROP_TARGET_PREFIX = 'config-node:';

function hasInteractiveDragAncestor(target: EventTarget | null) {
  return target instanceof Element && target.closest(`[${DRAG_INTERACTIVE_ATTRIBUTE}]`) != null;
}

function resolveDropTargetNodeId(targetId: string, selectedNodeId: string | null): string | null {
  if (targetId.startsWith(VRAM_DROP_TARGET_PREFIX)) {
    return targetId.slice(VRAM_DROP_TARGET_PREFIX.length) || null;
  }
  if (targetId.startsWith(CONFIG_NODE_DROP_TARGET_PREFIX)) {
    return targetId.slice(CONFIG_NODE_DROP_TARGET_PREFIX.length) || null;
  }
  if (targetId === 'vram-container') {
    return selectedNodeId;
  }
  return null;
}

export type { DragModelData, DragSplitAssignmentData };

export function DndContext({ children, selectedNodeId, onAssignModel, onMoveSplitAssignment }: DndContextProps) {
  const sensors = useMemo(
    () => [
      configure(PointerSensor, {
        preventActivation: (event) => hasInteractiveDragAncestor(event.target),
      }),
      configure(KeyboardSensor, {
        preventActivation: (event) => hasInteractiveDragAncestor(event.target),
      }),
    ],
    [],
  );

  return (
    <DragDropProvider
      sensors={sensors}
      onDragEnd={(event) => {
        if (event.canceled) return;

        const target = event.operation.target;
        const source = event.operation.source;

        if (!target || !source) return;

        const targetNodeId = resolveDropTargetNodeId(String(target.id), selectedNodeId);
        if (!targetNodeId) return;

        const data = source.data;
        if (isDragModelData(data)) {
          onAssignModel?.(data.modelName, data.sizeBytes, targetNodeId);
          return;
        }

        if (isDragSplitAssignmentData(data)) {
          onMoveSplitAssignment?.({
            assignmentId: data.assignmentId,
            sourceNodeId: data.sourceNodeId,
            targetNodeId,
          });
        }
      }}
    >
      {children}
    </DragDropProvider>
  );
}
