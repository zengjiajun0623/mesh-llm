import { useMemo } from 'react';

import type { OwnedNode } from '../../hooks/useOwnedNodes';
import { aggregateModels, type ModelCatalogPeer } from '../../lib/models';
import type { MeshConfig } from '../../types/config';
import type { NodeAssignTarget } from './CatalogContextMenu';
import { ModelCatalog } from './ModelCatalog';
import { TomlEditor } from './TomlEditor';

type SelectedCatalogNode = {
  id: string;
  hostname: string;
  vramBytes: number;
};

type ConfigToolsPanelProps = {
  peers: ModelCatalogPeer[];
  selectedCatalogNode: SelectedCatalogNode | null;
  selectedNode: OwnedNode | null;
  selectedNodeId: string | null;
  assignedBytesByNode: Map<string, number>;
  allDropTargetsLocked?: boolean;
  config: MeshConfig;
  onConfigChange: (config: MeshConfig) => void;
  onTomlParseErrorChange?: (error: string | null) => void;
  onNormalizeConfig?: (config: MeshConfig) => { config: MeshConfig; modified: boolean };
  onRefreshStatus?: () => Promise<void> | void;
  assignTargets?: NodeAssignTarget[];
  onAssignToNode?: (modelName: string, sizeBytes: number, nodeId: string) => void;
};

export function ConfigToolsPanel({
  peers,
  selectedCatalogNode,
  selectedNode,
  selectedNodeId,
  assignedBytesByNode,
  allDropTargetsLocked,
  config,
  onConfigChange,
  onTomlParseErrorChange,
  onNormalizeConfig,
  onRefreshStatus,
  assignTargets,
  onAssignToNode,
}: ConfigToolsPanelProps) {
  const fullyPlacedModels = useMemo(() => {
    const aggregated = aggregateModels(peers);
    const placed = new Set<string>();

    for (const model of aggregated) {
      if (
        model.nodeIds.length > 0 &&
        model.nodeIds.every((nodeId) => {
          const nodeConfig = config.nodes.find((n) => n.node_id === nodeId);
          return nodeConfig?.models.some((m) => m.name === model.name) ?? false;
        })
      ) {
        placed.add(model.name);
      }
    }

    return placed;
  }, [config.nodes, peers]);

  return (
    <div className="space-y-6" data-testid="config-layout">
      <div
        data-testid="config-tools-layout"
        className="grid grid-cols-1 gap-6 lg:min-h-[40rem] lg:grid-cols-2 lg:items-stretch"
      >
        <div className="min-w-0 lg:flex lg:min-h-full">
          <ModelCatalog
            peers={peers}
            selectedNode={
              selectedCatalogNode
                ? {
                    id: selectedCatalogNode.id,
                    hostname: selectedNode?.hostname ?? selectedCatalogNode.hostname,
                    vramBytes: selectedCatalogNode.vramBytes,
                  }
                : null
            }
            assignedBytes={
              selectedNodeId ? (assignedBytesByNode.get(selectedNodeId) ?? 0) : 0
            }
            disabled={allDropTargetsLocked}
            fullyPlacedModels={fullyPlacedModels}
            className="max-h-[min(88vh,68rem)] lg:flex-1"
            onRefreshStatus={onRefreshStatus}
            assignTargets={assignTargets}
            onAssignToNode={onAssignToNode}
          />
        </div>
        <div className="min-w-0 lg:flex lg:min-h-full">
          <TomlEditor
            config={config}
            onConfigChange={onConfigChange}
            onParseErrorChange={onTomlParseErrorChange}
            onNormalizeConfig={onNormalizeConfig}
            className="min-h-full max-h-none lg:flex-1"
            panelClassName="min-h-0 h-full max-h-none lg:h-full"
          />
        </div>
      </div>
    </div>
  );
}
