import { useCallback, useEffect, useRef } from 'react';
import { Info } from 'lucide-react';

import { type AggregatedModel } from '../../lib/models';
import { cn } from '../../lib/utils';
import { checkVramFit } from '../../lib/vram';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip';
import type { ContextMenuPosition } from './BlockContextMenu';

export type { ContextMenuPosition };

export type NodeAssignTarget = {
  id: string;
  hostname: string;
  vramBytes: number;
  assignedBytes: number;
  assignedModelNames: Set<string>;
};

type CatalogContextMenuProps = {
  position: ContextMenuPosition;
  model: AggregatedModel;
  assignTargets: NodeAssignTarget[];
  onAssignToNode: (nodeId: string) => void;
  onClose: () => void;
};

export function CatalogContextMenu({
  position,
  model,
  assignTargets,
  onAssignToNode,
  onClose,
}: CatalogContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    },
    [onClose],
  );

  const handlePointerDown = useCallback(
    (e: PointerEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    },
    [onClose],
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('pointerdown', handlePointerDown, true);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('pointerdown', handlePointerDown, true);
    };
  }, [handleKeyDown, handlePointerDown]);

  return (
    <TooltipProvider>
      <div
        ref={menuRef}
        data-testid="catalog-context-menu"
        data-config-assignment-interactive="true"
        role="menu"
        className="fixed z-50 min-w-[12rem] overflow-hidden rounded-md border border-border bg-popover py-1 text-popover-foreground shadow-md"
        style={{
          left: position.x,
          top: position.y,
          animationName: 'context-menu-in',
          animationDuration: '120ms',
          animationTimingFunction: 'ease-out',
          animationFillMode: 'both',
        }}
      >
        <div className="border-b border-border px-3 py-1.5 text-[11px] font-medium text-muted-foreground">
          Assign to node
        </div>

        {assignTargets.length === 0 ? (
          <div className="px-3 py-1.5 text-xs text-muted-foreground/60">No nodes available</div>
        ) : (
          assignTargets.map((target) => {
            const alreadyAssigned = target.assignedModelNames.has(model.name);
            const notAvailable = !model.nodeIds.includes(target.id);
            const vramCheck = checkVramFit(target.vramBytes, model.sizeBytes, target.assignedBytes);
            const noFit = !vramCheck.fits;

            const disabled = alreadyAssigned || notAvailable || noFit;
            const disabledReason = alreadyAssigned
              ? 'Already assigned to this node.'
              : notAvailable
                ? 'Model not available on this node.'
                : noFit
                  ? 'Not enough VRAM to fit this model.'
                  : null;

            return (
              <NodeMenuItem
                key={target.id}
                label={target.hostname || target.id}
                disabled={disabled}
                disabledReason={disabledReason}
                onClick={() => {
                  onAssignToNode(target.id);
                  onClose();
                }}
              />
            );
          })
        )}
      </div>
    </TooltipProvider>
  );
}

function NodeMenuItem({
  label,
  disabled,
  disabledReason,
  onClick,
}: {
  label: string;
  disabled: boolean;
  disabledReason: string | null;
  onClick: () => void;
}) {
  return (
    <div className="flex items-center gap-1 px-1">
      <button
        type="button"
        role="menuitem"
        disabled={disabled}
        onClick={onClick}
        className={cn(
          'flex min-w-0 flex-1 items-center rounded-sm px-2 py-1.5 text-left text-xs transition-colors',
          disabled
            ? 'cursor-not-allowed text-muted-foreground/40'
            : 'cursor-pointer text-foreground hover:bg-accent hover:text-accent-foreground',
        )}
      >
        {label}
      </button>
      {disabledReason ? (
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              aria-label="Why this node is unavailable"
              className="inline-flex h-6 w-6 flex-none items-center justify-center rounded-sm text-muted-foreground/70 transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              <Info className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          </TooltipTrigger>
          <TooltipContent side="right" className="max-w-64 text-pretty leading-relaxed">
            {disabledReason}
          </TooltipContent>
        </Tooltip>
      ) : null}
    </div>
  );
}
