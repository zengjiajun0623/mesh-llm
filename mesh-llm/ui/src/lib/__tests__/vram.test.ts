import { describe, expect, it } from 'vitest';

import { checkVramFit, estimateAssignmentBreakdownBytes, GPU_SYSTEM_OVERHEAD_BYTES } from '../vram';
import { fitsInPooled, fitsOnGpu, gpuTargets } from '../hardware';
import type { ScannedModelMetadata } from '../../types/config';

const exactMetadata: ScannedModelMetadata = {
  architecture: 'qwen3',
  embedding_length: 5120,
  total_layers: 64,
  attention: {
    head_count: 24,
    head_count_kv: 4,
    key_length: 128,
    value_length: 128,
  },
};

describe('checkVramFit', () => {
  it('matches the 1.1x overhead fit rule from election.rs', () => {
    const fits = checkVramFit(24_000_000_000, 20_000_000_000, 0);
    const noFit = checkVramFit(24_000_000_000, 22_000_000_000, 0);

    expect(fits.fits).toBe(true);
    // usedPercent = required / nodeVram = (20B*1.1 + 512MB) / 24B
    expect(fits.usedPercent).toBeCloseTo(((20_000_000_000 * 1.1 + GPU_SYSTEM_OVERHEAD_BYTES) / 24_000_000_000) * 100);

    expect(noFit.fits).toBe(false);
    // usedPercent = (22B*1.1 + 512MB) / 24B
    expect(noFit.usedPercent).toBeCloseTo(((22_000_000_000 * 1.1 + GPU_SYSTEM_OVERHEAD_BYTES) / 24_000_000_000) * 100);
  });

  it('accounts for bytes already assigned to the selected node', () => {
    const result = checkVramFit(24_000_000_000, 12_000_000_000, 12_000_000_000);

    expect(result.fits).toBe(false);
    // usedPercent = required / nodeVram (independent of existing assignedBytes)
    expect(result.usedPercent).toBeCloseTo(((12_000_000_000 * 1.1 + GPU_SYSTEM_OVERHEAD_BYTES) / 24_000_000_000) * 100);
  });

  it('8GB model on 9GB GPU does not fit with 512MB system overhead', () => {
    const modelBytes = 8_000_000_000;
    const gpuBytes = 9_000_000_000;
    const required = modelBytes * 1.1 + GPU_SYSTEM_OVERHEAD_BYTES;
    // 8*1.1 + 0.5 = 9.3GB > 9GB
    expect(required).toBeGreaterThan(gpuBytes);
    const result = checkVramFit(gpuBytes, modelBytes, 0);
    expect(result.fits).toBe(false);
  });

  it('8GB model on 10GB GPU fits with 512MB system overhead', () => {
    const modelBytes = 8_000_000_000;
    const gpuBytes = 10_000_000_000;
    const required = modelBytes * 1.1 + GPU_SYSTEM_OVERHEAD_BYTES;
    // 8*1.1 + 0.5 = 9.3GB < 10GB
    expect(required).toBeLessThan(gpuBytes);
    const result = checkVramFit(gpuBytes, modelBytes, 0);
    expect(result.fits).toBe(true);
  });
});

describe('estimateAssignmentBreakdownBytes', () => {
  it('separates core weights from model-aware context cache', () => {
    const breakdown = estimateAssignmentBreakdownBytes(22_000_000_000, 8192, exactMetadata);

    expect(breakdown.weightsBytes).toBe(22_000_000_000);
    expect(breakdown.contextBytes).toBe(536_870_912);
    expect(breakdown.totalBytes).toBe(22_536_870_912);
  });

  it('falls back to architecture heuristics when exact attention metadata is missing', () => {
    const breakdown = estimateAssignmentBreakdownBytes(22_000_000_000, 8192, {
      architecture: 'qwen3',
      embedding_length: 5120,
      total_layers: 64,
    });

    expect(breakdown.contextBytes).toBe(894_784_853);
  });

  it('falls back to the legacy estimate when metadata is unavailable', () => {
    const breakdown = estimateAssignmentBreakdownBytes(22_000_000_000, 8192, null);

    expect(breakdown.contextBytes).toBe(671_088_640);
  });
});

describe('pooled vs separate fit math', () => {
  const GPU_24GB = { name: 'RTX 4090', vram_bytes: 24_000_000_000 };
  const GPU_8GB = { name: 'RTX 3070', vram_bytes: 8_000_000_000 };

  describe('two 24 GB GPUs, 30 GB model', () => {
    const gpus = [GPU_24GB, GPU_24GB];
    const modelBytes = 30_000_000_000;

    it('fits in pooled mode (48 GB aggregate)', () => {
      expect(fitsInPooled(modelBytes, gpus)).toBe(true);
    });

    it('does not fit on GPU 0 alone', () => {
      const targets = gpuTargets(gpus);
      expect(fitsOnGpu(modelBytes, targets[0])).toBe(false);
    });

    it('does not fit on GPU 1 alone', () => {
      const targets = gpuTargets(gpus);
      expect(fitsOnGpu(modelBytes, targets[1])).toBe(false);
    });
  });

  describe('two 24 GB GPUs, 20 GB model', () => {
    const gpus = [GPU_24GB, GPU_24GB];
    const modelBytes = 20_000_000_000;

    it('fits in pooled mode', () => {
      expect(fitsInPooled(modelBytes, gpus)).toBe(true);
    });

    it('fits on GPU 0 alone', () => {
      const targets = gpuTargets(gpus);
      expect(fitsOnGpu(modelBytes, targets[0])).toBe(true);
    });

    it('fits on GPU 1 alone', () => {
      const targets = gpuTargets(gpus);
      expect(fitsOnGpu(modelBytes, targets[1])).toBe(true);
    });
  });

  describe('empty GPU array', () => {
    it('pooled mode returns false when no GPUs', () => {
      expect(fitsInPooled(1_000_000, [])).toBe(false);
    });

    it('gpuTargets returns empty array for empty input', () => {
      expect(gpuTargets([])).toHaveLength(0);
    });

    it('gpuTargets returns empty array for undefined input', () => {
      expect(gpuTargets(undefined)).toHaveLength(0);
    });
  });

  describe('one 8 GB GPU', () => {
    const gpus = [GPU_8GB];

    it('fitsInPooled and fitsOnGpu agree for model that fits', () => {
      const modelBytes = 5_000_000_000;
      const targets = gpuTargets(gpus);
      expect(fitsInPooled(modelBytes, gpus)).toBe(true);
      expect(fitsOnGpu(modelBytes, targets[0])).toBe(true);
    });

    it('fitsInPooled and fitsOnGpu agree for model that does not fit', () => {
      const tooBig = 10_000_000_000;
      const targets = gpuTargets(gpus);
      expect(fitsInPooled(tooBig, gpus)).toBe(false);
      expect(fitsOnGpu(tooBig, targets[0])).toBe(false);
    });

    it('gpuTargets returns array of length 1 with correct shape', () => {
      const targets = gpuTargets(gpus);
      expect(targets).toHaveLength(1);
      expect(targets[0].index).toBe(0);
      expect(targets[0].name).toBe('RTX 3070');
      expect(targets[0].vramBytes).toBe(8_000_000_000);
      expect(targets[0].label).toBe('GPU 0 · RTX 3070 · 8.0 GB');
    });
  });
});
