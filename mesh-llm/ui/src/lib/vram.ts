import type { ScannedModelMetadata } from '../types/config';

const GB = 1_000_000_000;
const LARGE_MODEL_BYTES = 50 * GB;
const MEDIUM_MODEL_BYTES = 5 * GB;
const KV_ALIGNMENT_TOKENS = 256;
export const GPU_SYSTEM_OVERHEAD_BYTES = 512 * 1024 * 1024;

export function checkVramFit(
  nodeVramBytes: number,
  modelSizeBytes: number,
  assignedBytes: number,
): { fits: boolean; usedPercent: number } {
  if (!Number.isFinite(nodeVramBytes) || nodeVramBytes <= 0) {
    return { fits: false, usedPercent: 0 };
  }

  const required = modelSizeBytes * 1.1 + GPU_SYSTEM_OVERHEAD_BYTES;
  const available = nodeVramBytes - assignedBytes;

  return {
    fits: required <= available,
    usedPercent: (required / nodeVramBytes) * 100,
  };
}

function normalizePositiveNumber(value?: number | null): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    return null;
  }

  return value;
}

function alignCtxSize(ctxSize: number): number {
  return Math.ceil(ctxSize / KV_ALIGNMENT_TOKENS) * KV_ALIGNMENT_TOKENS;
}

function resolveKvElementBytes(modelSizeBytes: number): number {
  if (modelSizeBytes >= LARGE_MODEL_BYTES) return 0.5;
  if (modelSizeBytes >= MEDIUM_MODEL_BYTES) return 1;
  return 2;
}

function deriveHeadDim(embeddingLength?: number | null, headCount?: number | null): number | null {
  const normalizedEmbeddingLength = normalizePositiveNumber(embeddingLength);
  const normalizedHeadCount = normalizePositiveNumber(headCount);
  if (!normalizedEmbeddingLength || !normalizedHeadCount) return null;

  return normalizedEmbeddingLength / normalizedHeadCount;
}

function normalizeArchitecture(metadata?: ScannedModelMetadata | null): string {
  return metadata?.architecture?.toLowerCase() ?? '';
}

function inferKvHeadRatio(metadata?: ScannedModelMetadata | null): number | null {
  const architecture = normalizeArchitecture(metadata);
  if (architecture.includes('qwen')) return 1 / 6;
  if (architecture.includes('gemma')) return 1 / 2;
  if (architecture.includes('mistral') || architecture.includes('mixtral') || architecture.includes('llama')) {
    return 1 / 4;
  }
  return null;
}

function estimateLegacyKvCacheBytes(ctxSize: number, modelSizeBytes = 0): number {
  const DEFAULT_LAYERS = 40;
  const DEFAULT_KV_HEADS = 8;
  const DEFAULT_HEAD_DIM = 128;
  const elementBytes = resolveKvElementBytes(modelSizeBytes);
  return Math.round(ctxSize * DEFAULT_LAYERS * DEFAULT_KV_HEADS * DEFAULT_HEAD_DIM * 2 * elementBytes);
}

export function estimateKvCacheBytes(
  ctxSize?: number,
  modelSizeBytes = 0,
  metadata?: ScannedModelMetadata | null,
): number {
  if (typeof ctxSize !== 'number' || !Number.isFinite(ctxSize) || ctxSize <= 0) {
    return 0;
  }

  const effectiveCtxSize = alignCtxSize(ctxSize);
  const totalLayers = normalizePositiveNumber(metadata?.total_layers);
  const embeddingLength = normalizePositiveNumber(metadata?.embedding_length);
  const headCount = normalizePositiveNumber(metadata?.attention?.head_count);
  const kvHeadCount = normalizePositiveNumber(metadata?.attention?.head_count_kv) ?? headCount;
  const keyLength = normalizePositiveNumber(metadata?.attention?.key_length)
    ?? deriveHeadDim(embeddingLength, headCount);
  const valueLength = normalizePositiveNumber(metadata?.attention?.value_length)
    ?? deriveHeadDim(embeddingLength, headCount);
  const bytesPerK = resolveKvElementBytes(modelSizeBytes);
  const bytesPerV = resolveKvElementBytes(modelSizeBytes);

  if (totalLayers && kvHeadCount && keyLength && valueLength) {
    return Math.round(
      effectiveCtxSize
      * totalLayers
      * kvHeadCount
      * ((keyLength * bytesPerK) + (valueLength * bytesPerV)),
    );
  }

  const kvHeadRatio = inferKvHeadRatio(metadata);
  if (totalLayers && embeddingLength && kvHeadRatio) {
    return Math.round(
      effectiveCtxSize
      * totalLayers
      * embeddingLength
      * kvHeadRatio
      * (bytesPerK + bytesPerV),
    );
  }

  return estimateLegacyKvCacheBytes(effectiveCtxSize, modelSizeBytes);
}

export function estimateAssignmentBreakdownBytes(
  modelSizeBytes: number,
  ctxSize?: number,
  metadata?: ScannedModelMetadata | null,
) {
  const weightsBytes = Math.max(0, modelSizeBytes);
  const contextBytes = estimateKvCacheBytes(ctxSize, modelSizeBytes, metadata);

  return {
    weightsBytes,
    contextBytes,
    totalBytes: weightsBytes + contextBytes,
  };
}

export function estimateAssignmentSizeBytes(
  modelSizeBytes: number,
  ctxSize?: number,
  metadata?: ScannedModelMetadata | null,
): number {
  return estimateAssignmentBreakdownBytes(modelSizeBytes, ctxSize, metadata).totalBytes;
}
