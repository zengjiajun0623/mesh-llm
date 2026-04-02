import { estimateKvCacheBytes } from '../../lib/vram';
import type { ScannedModelMetadata } from '../../types/config';

const MIN_CTX = 512;
const FALLBACK_MAX_CTX = 1_000_000;
const SLIDER_STEPS = 1000;

type ContextWindowControlProps = {
  modelName: string;
  currentCtxSize: number;
  modelSizeBytes?: number;
  metadata?: ScannedModelMetadata | null;
  onCtxSizeChange: (n: number) => void;
  onDragStart?: () => void;
  onDragEnd?: () => void;
};

function formatCtxLabel(size: number) {
  if (size >= 1_000_000) return `${Math.round(size / 1_000_000)}M`;
  if (size >= 1024) return `${Math.round(size / 1024)}K`;
  return String(size);
}

type ContextScaleMarksProps = {
  min: number;
  max: number;
  currentValue: number;
  onMarkClick: (value: number) => void;
};

export function ContextScaleMarks({ min, max, currentValue, onMarkClick }: ContextScaleMarksProps) {
  const majorMarks: number[] = [];
  let v = 1;
  while (v <= max) {
    if (v >= min) majorMarks.push(v);
    v *= 2;
  }
  if (majorMarks.length === 0 || majorMarks[majorMarks.length - 1] !== max) {
    majorMarks.push(max);
  }

  // Halfway ticks between each pair of adjacent major marks (geometric midpoint)
  const minorMarks: number[] = [];
  for (let i = 0; i < majorMarks.length - 1; i++) {
    minorMarks.push(Math.sqrt(majorMarks[i] * majorMarks[i + 1]));
  }

  const logMin = Math.log2(min);
  const logMax = Math.log2(max);
  const logRange = logMax - logMin;

  return (
    <div className="relative h-6 w-full px-1.5" data-testid="ctx-scale-marks">
      {minorMarks.map((mark) => {
        const pct = ((Math.log2(mark) - logMin) / logRange) * 100;
        const active = mark <= currentValue;
        return (
          <div
            key={`minor-${mark}`}
            className="absolute flex flex-col items-center"
            style={{ left: `${pct}%`, transform: 'translateX(-50%)' }}
          >
            <div className={`h-1 w-px transition-colors ${active ? 'bg-muted-foreground/50' : 'bg-muted-foreground/20'}`} />
          </div>
        );
      })}
      {majorMarks.map((mark) => {
        const pct = ((Math.log2(mark) - logMin) / logRange) * 100;
        const active = mark <= currentValue;
        return (
          <button
            key={mark}
            type="button"
            className="absolute flex flex-col items-center cursor-pointer"
            style={{ left: `${pct}%`, transform: 'translateX(-50%)' }}
            onClick={() => onMarkClick(mark)}
            tabIndex={-1}
          >
            <div className={`h-2.5 w-px transition-colors ${active ? 'bg-foreground/60' : 'bg-muted-foreground/30'}`} />
            <span
              className={`mt-1 text-[9px] tabular-nums font-mono leading-none select-none transition-colors hover:text-accent ${
                active ? 'text-foreground' : 'text-muted-foreground/50'
              }`}
            >
              {formatCtxLabel(mark)}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function logToLinear(logVal: number, logMin: number, logMax: number, linMin: number, linMax: number): number {
  const t = (logVal - logMin) / (logMax - logMin);
  return linMin + t * (linMax - linMin);
}

function linearToLog(linVal: number, linMin: number, linMax: number, logMin: number, logMax: number): number {
  const t = (linVal - linMin) / (linMax - linMin);
  return logMin + t * (logMax - logMin);
}

export function ContextWindowControl({
  modelName,
  currentCtxSize,
  modelSizeBytes = 0,
  metadata,
  onCtxSizeChange,
  onDragStart,
  onDragEnd,
}: ContextWindowControlProps) {
  const metadataMaxCtx = metadata?.context_length;
  const maxCtx = typeof metadataMaxCtx === 'number' && Number.isFinite(metadataMaxCtx) && metadataMaxCtx >= MIN_CTX
    ? metadataMaxCtx
    : FALLBACK_MAX_CTX;
  const kvGb = estimateKvCacheBytes(currentCtxSize, modelSizeBytes, metadata) / 1e9;

  const logMin = Math.log2(MIN_CTX);
  const logMax = Math.log2(maxCtx);
  const sliderValue = logToLinear(Math.log2(currentCtxSize), logMin, logMax, 0, SLIDER_STEPS);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const sliderVal = parseFloat(e.target.value);
    const logVal = linearToLog(sliderVal, 0, SLIDER_STEPS, logMin, logMax);
    const raw = Math.pow(2, logVal);
    const snapped = Math.max(MIN_CTX, Math.min(maxCtx, Math.round(raw / 128) * 128));
    onCtxSizeChange(snapped);
  };

  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={`ctx-resize-${modelName}`} className="text-xs font-medium text-foreground">
          Context
        </label>
        <span className="tabular-nums text-xs text-muted-foreground font-mono">
          {formatCtxLabel(currentCtxSize)} ctx
        </span>
      </div>
      <input
        id={`ctx-resize-${modelName}`}
        data-testid="ctx-resize-handle"
        type="range"
        min={0}
        max={SLIDER_STEPS}
        step={1}
        value={sliderValue}
        onChange={handleSliderChange}
        onPointerDown={onDragStart}
        onPointerUp={onDragEnd}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-muted/60 accent-primary focus-visible:outline-none
          [&::-moz-range-thumb]:h-3 [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:rounded-full
          [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-primary
          [&::-moz-range-thumb]:transition-transform [&::-moz-range-thumb]:duration-150 [&::-moz-range-thumb]:hover:scale-125
          [&::-moz-range-track]:h-1.5 [&::-moz-range-track]:rounded-full [&::-moz-range-track]:bg-muted/60
          [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary
          [&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:duration-150 [&::-webkit-slider-thumb]:hover:scale-125
          [&:focus-visible::-webkit-slider-thumb]:shadow-[0_0_0_3px_hsl(var(--ring))]
          [&:focus-visible::-moz-range-thumb]:shadow-[0_0_0_3px_hsl(var(--ring))]"
      />
      <ContextScaleMarks min={MIN_CTX} max={maxCtx} currentValue={currentCtxSize} onMarkClick={onCtxSizeChange} />
      <div className="text-center text-[10px] text-muted-foreground/70 font-mono">
        +{kvGb.toFixed(1)} GB KV cache
      </div>
    </div>
  );
}
