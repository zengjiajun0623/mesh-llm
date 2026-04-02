import { Info, TriangleAlert } from 'lucide-react';

import type { AggregatedModel } from '../../lib/models';
import { estimateAssignmentBreakdownBytes } from '../../lib/vram';
import type { ModelAssignment, ScannedModelMetadata } from '../../types/config';
import { ContextWindowControl } from './ContextWindowControl';
import { MoeSlider } from './MoeSlider';

type ModelDetailPanelProps = {
  assignmentId?: string | null;
  modelName: string | null;
  assignment: ModelAssignment | null;
  aggregated: AggregatedModel | null;
  metadata: ScannedModelMetadata | null;
  onUpdateModel?: (assignmentId: string, updates: Partial<ModelAssignment>) => void;
  onBeginBatch?: () => void;
  onEndBatch?: () => void;
};

function MetadataRow({ label, value }: { label: string; value: string | number | undefined | null }) {
  if (value == null || value === '') return null;
  return (
    <div className="min-w-0 rounded-md border border-border/40 bg-muted/10 px-3 py-2">
      <div className="text-[11px] font-medium text-muted-foreground">{label}</div>
      <div className="mt-1 text-xs font-medium text-foreground font-mono leading-5 [overflow-wrap:anywhere]">{value}</div>
    </div>
  );
}

function formatNumber(n: number | undefined | null): string | null {
  if (n == null || !Number.isFinite(n)) return null;
  return n.toLocaleString();
}

function formatCtx(n: number | undefined | null): string | null {
  if (n == null || !Number.isFinite(n)) return null;
  if (n >= 1024 * 1024) return `${Math.round(n / (1024 * 1024))}M`;
  if (n >= 1024) return `${Math.round(n / 1024)}K`;
  return String(n);
}

function formatGb(bytes: number | undefined | null): string | null {
  if (bytes == null || !Number.isFinite(bytes)) return null;
  const gb = bytes / 1e9;
  return `${gb >= 100 ? Math.round(gb) : gb.toFixed(1)} GB`;
}

function EmptyDetailState() {
  return (
    <div
      data-testid="detail-panel-empty"
      className="flex min-h-[10rem] flex-col items-center justify-center rounded-md border border-dashed bg-muted/20 px-4 py-6 text-center"
    >
      <Info className="mb-2 h-5 w-5 text-muted-foreground/50" aria-hidden="true" />
      <span className="text-xs text-muted-foreground">Select a model block to inspect it</span>
    </div>
  );
}

export function ModelDetailPanel({ assignmentId, modelName, assignment, aggregated, metadata, onUpdateModel, onBeginBatch, onEndBatch }: ModelDetailPanelProps) {
  if (!modelName || !assignment) {
    return <EmptyDetailState />;
  }

  const resolvedAssignmentId = assignmentId ?? modelName;

  const splitRange = assignment.split
    ? `${assignment.split.start}–${assignment.split.end} of ${assignment.split.total}`
    : null;

  const errorMessage = (assignment as ModelAssignment & { _errorMessage?: string })._errorMessage ?? null;
  const vramBreakdown = aggregated
    ? estimateAssignmentBreakdownBytes(aggregated.sizeBytes, assignment.ctx_size, metadata)
    : null;
  const hasMoeControl = Boolean(aggregated?.moe);

  return (
    <div
      data-testid="model-detail-panel"
      data-config-assignment-interactive="true"
      className="space-y-4 rounded-md border border-border bg-card p-4"
    >
      <div className="space-y-1">
        <h3 className="text-sm font-semibold tracking-tight text-foreground truncate" title={modelName}>
          {modelName}
        </h3>
        {metadata?.architecture ? (
          <span className="text-xs text-muted-foreground">{metadata.architecture}</span>
        ) : null}
      </div>

      {errorMessage ? (
        <div
          data-testid="detail-panel-error"
          className="rounded border border-destructive/50 bg-destructive/10 px-3 py-1.5 text-xs text-destructive"
        >
          {errorMessage}
        </div>
      ) : null}

      {metadata ? (
        <div
          data-testid="detail-panel-metadata"
          className="grid gap-2 border-t border-border/40 pt-3 grid-cols-2 sm:grid-cols-4 xl:grid-cols-6"
        >
          <MetadataRow label="Quantization" value={metadata.quantization_type} />
          <MetadataRow label="Context length" value={formatCtx(metadata.context_length)} />
          {vramBreakdown ? <MetadataRow label="Core weights" value={formatGb(vramBreakdown.weightsBytes)} /> : null}
          {vramBreakdown ? <MetadataRow label="Context cache" value={formatGb(vramBreakdown.contextBytes)} /> : null}
          {vramBreakdown ? <MetadataRow label="Estimated total" value={formatGb(vramBreakdown.totalBytes)} /> : null}
          <MetadataRow label="Embedding size" value={formatNumber(metadata.embedding_length)} />
          <MetadataRow label="Total layers" value={formatNumber(metadata.total_layers)} />
          <MetadataRow label="Attention heads" value={formatNumber(metadata.attention?.head_count)} />
          <MetadataRow label="KV heads" value={formatNumber(metadata.attention?.head_count_kv)} />
          <MetadataRow label="Offloadable layers" value={formatNumber(metadata.total_offloadable_layers)} />
          {splitRange ? <MetadataRow label="Split range" value={splitRange} /> : null}
          {metadata.rope?.kind ? <MetadataRow label="RoPE type" value={metadata.rope.kind} /> : null}
          {metadata.rope?.freq_base ? <MetadataRow label="RoPE freq base" value={formatNumber(metadata.rope.freq_base)} /> : null}
          {metadata.experts?.expert_count ? (
            <MetadataRow
              label="Experts"
              value={`${metadata.experts.expert_count} total, ${metadata.experts.expert_used_count ?? '?'} active`}
            />
          ) : null}
          {metadata.tokenizer?.model ? <MetadataRow label="Tokenizer" value={metadata.tokenizer.model} /> : null}
          {metadata.dense_split_capable != null ? (
            <MetadataRow label="Dense split" value={metadata.dense_split_capable ? 'Capable' : 'Not supported'} />
          ) : null}
        </div>
      ) : (
        <div className="flex flex-col items-center gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-4 py-5 text-center">
          <TriangleAlert className="h-5 w-5 text-amber-500" aria-hidden="true" />
          <p className="text-xs font-medium text-amber-600 dark:text-amber-400">No GGUF metadata available</p>
          <p className="max-w-sm text-[11px] leading-relaxed text-muted-foreground">
            KV cache and VRAM estimates are best-guess only. The model may fail to load if actual memory requirements exceed available VRAM.
          </p>
        </div>
      )}

      {onUpdateModel ? (
        <div
          className={hasMoeControl ? 'grid gap-3 border-t border-border/40 pt-3 md:grid-cols-2' : 'space-y-3 border-t border-border/40 pt-3'}
        >
          <ContextWindowControl
            modelName={modelName}
            currentCtxSize={assignment.ctx_size ?? 4096}
            modelSizeBytes={aggregated?.sizeBytes ?? 0}
            metadata={metadata}
            onCtxSizeChange={(n) => onUpdateModel(resolvedAssignmentId, { ctx_size: n })}
            onDragStart={onBeginBatch}
            onDragEnd={onEndBatch}
          />
          {aggregated?.moe ? (
            <MoeSlider
              modelName={modelName}
              moe={aggregated.moe}
              currentExperts={assignment.moe_experts ?? aggregated.moe.minExpertsPerNode}
              modelSizeBytes={aggregated.sizeBytes}
              onExpertsChange={(n) => onUpdateModel(resolvedAssignmentId, { moe_experts: n })}
            />
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
