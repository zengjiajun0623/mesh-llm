import { useDragOperation, useDraggable } from '@dnd-kit/react';
import { Package, RefreshCcwDot, Search, SearchX } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState, type ElementType, type ReactNode } from 'react';

import { CatalogContextMenu, type ContextMenuPosition, type NodeAssignTarget } from './CatalogContextMenu';

import { broadcastScan } from '../../lib/api';
import { aggregateModels, type AggregatedModel, type ModelCatalogPeer } from '../../lib/models';
import { cn } from '../../lib/utils';
import { checkVramFit } from '../../lib/vram';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Input } from '../ui/input';

type RefreshStatus =
  | { type: 'idle' }
  | { type: 'refreshing' }
  | { type: 'success'; refreshed: number; total: number }
  | { type: 'partial'; refreshed: number; total: number; failed: string[] }
  | { type: 'error'; error: string };

type SelectedCatalogNode = {
  id: string;
  hostname: string;
  vramBytes: number;
};

type ModelCatalogProps = {
  peers: ModelCatalogPeer[];
  selectedNode: SelectedCatalogNode | null;
  assignedBytes?: number;
  disabled?: boolean;
  fullyPlacedModels?: Set<string>;
  className?: string;
  scrollAreaClassName?: string;
  onRefreshStatus?: () => Promise<void> | void;
  assignTargets?: NodeAssignTarget[];
  onAssignToNode?: (modelName: string, sizeBytes: number, nodeId: string) => void;
};

type FilterType = 'all' | 'moe' | 'vision' | 'lt8' | '8to32' | 'gt32';

const FILTER_PILLS: { type: FilterType; label: string }[] = [
  { type: 'all', label: 'All' },
  { type: 'moe', label: 'MoE' },
  { type: 'vision', label: 'Vision' },
  { type: 'lt8', label: '< 8GB' },
  { type: '8to32', label: '8-32 GB' },
  { type: 'gt32', label: '> 32GB' },
];

const VISION_PATTERN = /mmproj|vision/i;

function matchesFilter(model: AggregatedModel, filter: FilterType): boolean {
  switch (filter) {
    case 'all':
      return true;
    case 'moe':
      return model.moe != null;
    case 'vision':
      return VISION_PATTERN.test(model.name);
    case 'lt8':
      return model.sizeBytes < 8e9;
    case '8to32':
      return model.sizeBytes >= 8e9 && model.sizeBytes <= 32e9;
    case 'gt32':
      return model.sizeBytes > 32e9;
  }
}

function formatSize(sizeGb: number) {
  return `${new Intl.NumberFormat('en-US', {
    minimumFractionDigits: sizeGb >= 100 ? 0 : 1,
    maximumFractionDigits: sizeGb >= 100 ? 0 : 1,
  }).format(sizeGb)} GB`;
}

function fitToneClass(fits: boolean | undefined) {
  if (fits === true) {
    return 'border-emerald-500/40 bg-emerald-500/5 ring-1 ring-emerald-500/25';
  }

  if (fits === false) {
    return 'opacity-50';
  }

  return 'border-border/70 bg-background/95';
}

function nodeLabelMap(peers: ModelCatalogPeer[]) {
  return new Map(peers.map((peer) => [peer.id, peer.hostname?.trim() || peer.id]));
}

function modelModeLabel(model: AggregatedModel) {
  if (model.moe) {
    return `MoE ${model.moe.nExpertUsed}/${model.moe.nExpert}`;
  }

  if (VISION_PATTERN.test(model.name)) {
    return 'Vision';
  }

  return 'Dense';
}

function isEditableElement(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }

  if (target.isContentEditable) {
    return true;
  }

  const tagName = target.tagName;
  if (tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT') {
    return true;
  }

  return target.closest('[contenteditable="true"], [role="textbox"]') != null;
}

function CatalogEmptyState({
  icon: Icon,
  title,
  description,
  detail,
  actions,
}: {
  icon?: ElementType;
  title: string;
  description: ReactNode;
  detail?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="flex min-h-[18rem] flex-1 flex-col items-center justify-center rounded-md border border-dashed border-border/70 bg-muted/20 px-6 py-8 text-center">
      {Icon ? <Icon className="mb-3 h-8 w-8 text-muted-foreground/60" aria-hidden="true" /> : null}
      <div className="max-w-md space-y-3">
        <div className="space-y-1.5">
          <div className="text-sm font-medium text-foreground">{title}</div>
          <div className="text-sm text-muted-foreground">{description}</div>
          {detail ? <div className="text-xs text-muted-foreground">{detail}</div> : null}
        </div>
        {actions ? <div className="flex flex-wrap items-center justify-center gap-2">{actions}</div> : null}
      </div>
    </div>
  );
}

function ModelCatalogCard({
  model,
  nodeNames,
  selectedNode,
  assignedBytes,
  disabled,
  fullyPlaced,
  useContentVisibility,
  onContextMenu,
}: {
  model: AggregatedModel;
  nodeNames: string[];
  selectedNode: SelectedCatalogNode | null;
  assignedBytes: number;
  disabled?: boolean;
  fullyPlaced?: boolean;
  useContentVisibility: boolean;
  onContextMenu?: (e: React.MouseEvent, model: AggregatedModel) => void;
}) {
  const dragId = `model:${model.name}`;
  const { source } = useDragOperation();
  const { ref, handleRef } = useDraggable({
    id: dragId,
    data: {
      type: 'model',
      modelName: model.name,
      sizeBytes: model.sizeBytes,
      nodeIds: model.nodeIds,
    },
    disabled,
  });

  const isActiveDrag = String(source?.id) === dragId;

  const fit = selectedNode ? checkVramFit(selectedNode.vramBytes, model.sizeBytes, assignedBytes) : null;

  return (
    <div
      ref={ref}
      data-testid="model-card"
      data-fits={fit ? String(fit.fits) : undefined}
      data-dragging={String(isActiveDrag)}
      className={cn(
        'group/model relative w-full rounded-lg border bg-background/95 p-3 text-left shadow-soft transition-[background-color,border-color,opacity,transform] duration-150 hover:bg-muted/20',
        fullyPlaced
          ? 'pointer-events-none cursor-default opacity-50'
          : disabled
            ? 'pointer-events-none cursor-default opacity-40'
            : fitToneClass(fit?.fits),
        !disabled && !fullyPlaced && (isActiveDrag ? 'cursor-grabbing bg-background opacity-85' : 'cursor-grab'),
      )}
      style={useContentVisibility ? { contentVisibility: 'auto', containIntrinsicSize: 'auto 112px' } : undefined}
    >
      <button
        ref={handleRef}
        type="button"
        aria-label={model.name}
        data-fits={fit ? String(fit.fits) : undefined}
        data-testid="model-card-drag-handle"
        className="absolute inset-0 z-10 rounded-lg bg-transparent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        onContextMenu={onContextMenu ? (e) => onContextMenu(e, model) : undefined}
      />

      <div className="relative z-0 flex min-w-0 flex-col gap-2">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 space-y-1">
            <div className="text-sm font-medium leading-5 [overflow-wrap:anywhere]">{model.name}</div>
            <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
              <Badge className="border-primary/30 bg-primary/10 px-2 py-0.5 text-[11px] text-primary">
                {modelModeLabel(model)}
              </Badge>
              <Badge className="px-2 py-0.5 text-[11px]">
                {model.nodeIds.length} node{model.nodeIds.length === 1 ? '' : 's'}
              </Badge>
              {fullyPlaced ? (
                <Badge className="border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[11px] text-emerald-700 dark:text-emerald-300">
                  On all nodes
                </Badge>
              ) : null}
              {selectedNode ? (
                <Badge
                  className={cn(
                    'px-2 py-0.5 text-[11px]',
                    fit?.fits
                      ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-200'
                      : 'border-border/60 bg-background/70 text-muted-foreground',
                  )}
                >
                  {fit?.fits ? 'Fits selected node' : 'No fit on selected node'}
                </Badge>
              ) : null}
            </div>
          </div>

          <div className="shrink-0 text-right">
            <Badge className="px-2 py-0.5 text-[11px] text-foreground">{formatSize(model.sizeGb)}</Badge>
            {selectedNode && fit ? (
              <div className="mt-1 text-[11px] text-muted-foreground">uses {fit.usedPercent.toFixed(0)}%</div>
            ) : null}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
          <span className="text-muted-foreground/50">Available on</span>
          {nodeNames.map((nodeName) => (
            <Badge key={nodeName} className="px-2 py-0.5 text-[11px]">
              {nodeName}
            </Badge>
          ))}
        </div>
      </div>
    </div>
  );
}

export function ModelCatalog({
  peers,
  selectedNode,
  assignedBytes = 0,
  disabled,
  fullyPlacedModels,
  className,
  scrollAreaClassName,
  onRefreshStatus,
  assignTargets,
  onAssignToNode,
}: ModelCatalogProps) {
  const [search, setSearch] = useState('');
  const [activeFilter, setActiveFilter] = useState<FilterType>('all');
  const [isSearchFocused, setIsSearchFocused] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const [refreshStatus, setRefreshStatus] = useState<RefreshStatus>({ type: 'idle' });
  const refreshDismissRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [catalogContextMenu, setCatalogContextMenu] = useState<{
    position: ContextMenuPosition;
    model: AggregatedModel;
  } | null>(null);

  const handleCardContextMenu = useCallback((e: React.MouseEvent, model: AggregatedModel) => {
    if (!onAssignToNode || !assignTargets) return;
    e.preventDefault();
    e.stopPropagation();
    setCatalogContextMenu({ position: { x: e.clientX, y: e.clientY }, model });
  }, [onAssignToNode, assignTargets]);

  const closeContextMenu = useCallback(() => {
    setCatalogContextMenu(null);
  }, []);

  useEffect(() => {
    return () => {
      if (refreshDismissRef.current) clearTimeout(refreshDismissRef.current);
    };
  }, []);

  const handleRefreshModels = useCallback(async () => {
    setRefreshStatus({ type: 'refreshing' });

    const result = await broadcastScan();

    if (result.ok) {
      setRefreshStatus({
        type: 'success',
        refreshed: result.refreshed ?? 1,
        total: result.total ?? 1,
      });
    } else if (result.refreshed && result.refreshed > 0) {
      setRefreshStatus({
        type: 'partial',
        refreshed: result.refreshed,
        total: result.total ?? result.refreshed,
        failed: result.failed ?? [],
      });
    } else {
      setRefreshStatus({ type: 'error', error: 'Failed to refresh models' });
    }

    try {
      await onRefreshStatus?.();
    } catch (error) {
      console.warn('Failed to refresh config status after model scan', error);
    }

    if (refreshDismissRef.current) clearTimeout(refreshDismissRef.current);
    refreshDismissRef.current = setTimeout(() => {
      setRefreshStatus({ type: 'idle' });
    }, 5000);
  }, [onRefreshStatus]);

  const isRefreshing = refreshStatus.type === 'refreshing';

  const models = useMemo(() => aggregateModels(peers), [peers]);
  const peerLabels = useMemo(() => nodeLabelMap(peers), [peers]);
  const searchQuery = search.trim();

  const hasNonAllFilter = activeFilter !== 'all';
  const hasSearchQuery = searchQuery.length > 0;
  const showSearch = models.length > 8 || hasNonAllFilter;
  const useContentVisibility = models.length > 50;
  const activeFilterLabel = FILTER_PILLS.find((pill) => pill.type === activeFilter)?.label ?? 'All';
  const modelCountLabel = `${models.length.toLocaleString()} model${models.length === 1 ? '' : 's'}`;

  const filteredModels = useMemo(() => {
    let result = models;

    if (activeFilter !== 'all') {
      result = result.filter((model) => matchesFilter(model, activeFilter));
    }

    const query = searchQuery.toLowerCase();
    if (query.length > 0) {
      result = result.filter((model) => model.name.toLowerCase().includes(query));
    }

    return result;
  }, [activeFilter, models, searchQuery]);

  const handleFilterClick = useCallback((filter: FilterType) => {
    setActiveFilter((current) => (current === filter ? 'all' : filter));
  }, []);

  const focusSearchInput = useCallback(() => {
    const input = searchInputRef.current;
    if (!input) {
      return;
    }

    input.focus({ preventScroll: true });
    input.select();
  }, []);

  useEffect(() => {
    if (!showSearch) {
      return;
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.defaultPrevented) {
        return;
      }

      if ((event.metaKey || event.ctrlKey) && !event.shiftKey && !event.altKey && event.key.toLowerCase() === 'k') {
        if (isEditableElement(event.target)) {
          return;
        }

        event.preventDefault();
        focusSearchInput();
      }
    }

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [focusSearchInput, showSearch]);

  return (
    <Card className={cn('flex min-h-[22rem] max-h-[34rem] flex-col overflow-hidden', className)}>
      <CardHeader className="border-b border-border">
        <div className="min-w-0 space-y-1">
          <div className="flex items-start justify-between gap-3">
            <CardTitle className="text-lg font-semibold">Model catalog</CardTitle>
            <div className="flex shrink-0 items-center gap-2">
              <Badge className="px-2 py-0.5 text-[11px]">{modelCountLabel}</Badge>
              <button
                data-testid="refresh-models-btn"
                type="button"
                onClick={handleRefreshModels}
                disabled={isRefreshing}
                title="Rescan models on all nodes"
                className={cn(
                  'inline-flex h-7 w-7 items-center justify-center rounded-md border border-border text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-60',
                )}
              >
                <RefreshCcwDot aria-hidden="true" className={cn('h-3.5 w-3.5', isRefreshing && 'animate-spin')} />
                <span className="sr-only">Refresh models</span>
              </button>
            </div>
          </div>
          <p className="text-sm text-muted-foreground">
            Browse locally scanned GGUF models across your owned nodes. Click a node section to preview fit before assignment.
          </p>
        </div>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col gap-4 pt-3">
        {showSearch || models.length > 8 ? (
          <div className="rounded-lg border border-border/70 bg-muted/15 p-2 shadow-soft">
            <div
              className={cn(
                'flex flex-col gap-2 transition-[gap] duration-200 ease-out lg:min-h-10 lg:flex-row lg:flex-nowrap lg:items-center',
                isSearchFocused ? 'lg:gap-0' : 'lg:gap-3',
              )}
            >
              {showSearch ? (
                <div
                  className={cn(
                    'min-w-0 transition-[flex-basis,width,max-width] duration-200 ease-out lg:min-w-[10rem]',
                    isSearchFocused
                      ? 'lg:flex-1 lg:basis-0 lg:w-auto lg:max-w-none'
                      : 'lg:flex-none lg:basis-auto lg:w-[12rem] lg:max-w-[13rem]',
                  )}
                >
                  <div className="flex h-9 w-full max-w-full items-center gap-2 rounded-md border border-border/70 bg-background/90 px-3 transition-[border-color,box-shadow] duration-200 ease-out focus-within:border-primary/40 focus-within:ring-1 focus-within:ring-ring/40">
                    <Search aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                    <Input
                      ref={searchInputRef}
                      value={search}
                      onChange={(event) => setSearch(event.target.value)}
                      onFocus={() => setIsSearchFocused(true)}
                      onBlur={() => setIsSearchFocused(false)}
                      placeholder="Search models"
                      data-testid="model-search"
                      aria-label="Search models"
                      className="min-w-0 flex-1 h-auto border-0 bg-transparent px-0 py-0 text-sm shadow-none placeholder:text-muted-foreground/75 focus-visible:ring-0 focus-visible:ring-offset-0"
                    />
                  </div>
                </div>
              ) : null}

              {models.length > 8 ? (
                <div
                  className={cn(
                    'flex min-w-0 items-center transition-[flex-basis,width,max-width,opacity] duration-200 ease-out',
                    isSearchFocused
                      ? 'lg:flex-none lg:basis-0 lg:w-0 lg:max-w-0 lg:justify-end lg:overflow-hidden lg:opacity-0'
                      : 'lg:flex-1 lg:basis-0 lg:w-auto lg:max-w-none lg:justify-end',
                  )}
                >
                  <fieldset
                    className={cn(
                      'm-0 flex min-w-0 max-w-full flex-wrap items-center gap-1 rounded-md border border-border/60 bg-background/40 p-1 transition-[max-width,opacity,height,padding,gap] duration-200 ease-out lg:justify-end',
                      isSearchFocused
                        ? 'lg:pointer-events-none lg:h-0 lg:max-w-0 lg:flex-wrap lg:gap-0 lg:overflow-hidden lg:border-transparent lg:p-0 lg:opacity-0'
                        : 'lg:h-auto lg:flex-nowrap lg:opacity-100',
                    )}
                    aria-label="Filter models"
                  >
                    <legend className="sr-only">Filter models</legend>
                    {FILTER_PILLS.map((pill) => (
                      <button
                        key={pill.type}
                        type="button"
                        data-testid={`filter-pill-${pill.type}`}
                        onClick={() => handleFilterClick(pill.type)}
                        className={cn(
                          'whitespace-nowrap rounded-md px-2.5 py-1.5 text-[11px] font-medium transition-[background-color,border-color,color,box-shadow] duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-0',
                          activeFilter === pill.type
                            ? 'border border-primary/40 bg-primary/10 text-primary shadow-soft'
                            : 'border border-transparent bg-transparent text-muted-foreground hover:border-border/80 hover:bg-background/80 hover:text-foreground',
                        )}
                      >
                        {pill.label}
                      </button>
                    ))}
                  </fieldset>
                </div>
              ) : null}
            </div>
          </div>
        ) : null}

        {models.length === 0 ? (
          <CatalogEmptyState
            icon={Package}
            title="Catalog is empty"
            description={
              <>
                No models found. Add GGUF files to <code className="rounded bg-muted px-1.5 py-0.5 text-[11px] text-foreground">~/.models/</code>
              </>
            }
            detail="Models show up here after one of your owned nodes scans and advertises them."
          />
        ) : filteredModels.length === 0 ? (
          <CatalogEmptyState
            icon={SearchX}
            title="No matching models"
            description={
              hasSearchQuery && hasNonAllFilter ? (
                <>
                  Nothing matches <span className="font-medium text-foreground">“{searchQuery}”</span> with the{' '}
                  <span className="font-medium text-foreground">{activeFilterLabel}</span> filter.
                </>
              ) : hasSearchQuery ? (
                <>
                  Nothing matches <span className="font-medium text-foreground">“{searchQuery}”</span>.
                </>
              ) : (
                <>
                  No models match the <span className="font-medium text-foreground">{activeFilterLabel}</span> filter.
                </>
              )
            }
            detail={
              hasSearchQuery && hasNonAllFilter
                ? 'Clear the search or reset filters to get back to the full catalog.'
                : hasSearchQuery
                  ? 'Try a broader search or clear it to browse the full catalog again.'
                  : 'Reset filters to return to the full catalog.'
            }
            actions={
              <>
                {hasSearchQuery ? (
                  <Button variant="outline" size="sm" onClick={() => setSearch('')}>
                    Clear search
                  </Button>
                ) : null}
                {hasNonAllFilter ? (
                  <Button variant="ghost" size="sm" onClick={() => setActiveFilter('all')}>
                    Reset filters
                  </Button>
                ) : null}
              </>
            }
          />
        ) : (
          <div
            data-testid="model-catalog"
            className="flex min-h-0 flex-1 flex-col rounded-md border border-border/70 bg-muted/10 p-3"
          >
            <div
              data-testid="model-catalog-scroll-area"
              className={cn('min-h-0 flex-1 overflow-y-auto pr-3', scrollAreaClassName)}
            >
              <div className="flex flex-col gap-2.5">
                {filteredModels.map((model) => {
                  const isFullyPlaced = fullyPlacedModels?.has(model.name) ?? false;
                  return (
                    <ModelCatalogCard
                      key={model.name}
                      model={model}
                      nodeNames={model.nodeIds.map((nodeId) => peerLabels.get(nodeId) ?? nodeId)}
                      selectedNode={selectedNode}
                      assignedBytes={assignedBytes}
                      disabled={disabled || isFullyPlaced}
                      fullyPlaced={isFullyPlaced}
                      useContentVisibility={useContentVisibility}
                      onContextMenu={onAssignToNode && assignTargets ? handleCardContextMenu : undefined}
                    />
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </CardContent>

      {refreshStatus.type === 'success' && (
        <div
          data-testid="refresh-toast"
          className="fixed bottom-4 right-4 z-50 max-w-sm p-4 rounded-lg bg-green-900/80 text-green-100 border border-green-700/50 shadow-lg text-sm"
        >
          Refreshed models on {refreshStatus.refreshed}/{refreshStatus.total} node{refreshStatus.total !== 1 ? 's' : ''}
        </div>
      )}

      {refreshStatus.type === 'partial' && (
        <div
          data-testid="refresh-toast"
          className="fixed bottom-4 right-4 z-50 max-w-sm p-4 rounded-lg bg-amber-900/80 text-amber-100 border border-amber-700/50 shadow-lg text-sm"
        >
          Refreshed: {refreshStatus.refreshed}/{refreshStatus.total}. Failed: {refreshStatus.failed.join(', ')}
        </div>
      )}

      {refreshStatus.type === 'error' && (
        <div
          data-testid="refresh-toast"
          className="fixed bottom-4 right-4 z-50 max-w-sm p-4 rounded-lg bg-red-900/80 text-red-100 border border-red-700/50 shadow-lg text-sm"
        >
          {refreshStatus.error}
        </div>
      )}

      {catalogContextMenu && onAssignToNode && assignTargets ? (
        <CatalogContextMenu
          position={catalogContextMenu.position}
          model={catalogContextMenu.model}
          assignTargets={assignTargets}
          onAssignToNode={(nodeId) =>
            onAssignToNode(catalogContextMenu.model.name, catalogContextMenu.model.sizeBytes, nodeId)
          }
          onClose={closeContextMenu}
        />
      ) : null}
    </Card>
  );
}
