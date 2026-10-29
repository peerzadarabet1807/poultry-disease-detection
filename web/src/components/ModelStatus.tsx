import type { ModelState } from '../hooks/useClassifier';
import { formatBytes } from '../lib/fileValidation';
import { RefreshIcon } from './icons';

interface Props {
  model: ModelState;
  lastInferenceMs?: number;
  onRetry: () => void;
}

export function ModelStatus({ model, lastInferenceMs, onRetry }: Props) {
  if (model.status === 'error') {
    return (
      <div role="alert" className="flex flex-wrap items-center gap-3 text-sm text-alert-ink">
        <span className="size-2 rounded-full bg-alert-line" aria-hidden="true" />
        <span className="min-w-0 flex-1">{model.message}</span>
        <button type="button" className="btn btn-secondary !min-h-9 !px-3 !text-sm" onClick={onRetry}>
          <RefreshIcon /> Retry
        </button>
      </div>
    );
  }

  if (model.status === 'loading') {
    const downloading = model.phase === 'download';
    const pct = Math.round(model.fraction * 100);
    const label =
      model.phase === 'connect'
        ? 'Connecting to the API…'
        : downloading
          ? `Downloading the model: ${formatBytes(model.fraction * __MODEL_BYTES__)} of ${formatBytes(__MODEL_BYTES__)}`
          : 'Preparing the model (compiling kernels)…';
    return (
      <div className="space-y-2 text-sm" data-testid="model-status" data-state="loading">
        <div className="flex items-baseline justify-between gap-3 text-ink-2">
          <span id="model-progress-label">{label}</span>
          {downloading && <span className="tabular text-ink-3">{pct}%</span>}
        </div>
        <div
          role="progressbar"
          aria-labelledby="model-progress-label"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={downloading ? pct : undefined}
          className="h-1.5 overflow-hidden rounded-full bg-track"
        >
          <div
            className={`h-full rounded-full bg-yolk transition-[width] duration-300 ${downloading ? '' : 'w-1/3 animate-pulse'}`}
            style={downloading ? { width: `${pct}%` } : undefined}
          />
        </div>
        {downloading && (
          <p className="text-xs text-ink-3">
            One-time download; the model is cached in your browser for next time.
          </p>
        )}
      </div>
    );
  }

  return (
    <p
      className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-ink-2"
      data-testid="model-status"
      data-state="ready"
    >
      <span className="size-2 rounded-full bg-healthy-line" aria-hidden="true" />
      <span className="font-medium text-ink">Model ready</span>
      <span aria-hidden="true">·</span>
      <span data-testid="backend">{model.info.detail}</span>
      {lastInferenceMs !== undefined && (
        <>
          <span aria-hidden="true">·</span>
          <span className="tabular" data-testid="inference-ms">
            last inference {Math.round(lastInferenceMs)} ms
          </span>
        </>
      )}
    </p>
  );
}
