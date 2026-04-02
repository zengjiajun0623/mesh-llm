import type { ConfigValidationError } from '../../lib/api';
import type { MeshConfig } from '../../types/config';
import { SaveConfig } from './SaveConfig';

type ConfigPageHeaderProps = {
  config: MeshConfig;
  savedConfig?: MeshConfig;
  isDirty: boolean;
  isConfigValid: boolean;
  invalidReason?: string | null;
  isConfigLoading: boolean;
  loadError: string | null;
  onSaveSuccess: (savedConfig: MeshConfig) => void;
  onRevert?: () => void;
  onBackendErrors?: (errors: ConfigValidationError[]) => void;
  canUndo?: boolean;
  canRedo?: boolean;
  onUndo?: () => void;
  onRedo?: () => void;
};

export function ConfigPageHeader({
  config,
  savedConfig,
  isDirty,
  isConfigValid,
  invalidReason,
  isConfigLoading,
  loadError,
  onSaveSuccess,
  onRevert,
  onBackendErrors,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
}: ConfigPageHeaderProps) {
  return (
    <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
      <div className="min-w-0 space-y-1">
        <h1 className="text-lg font-semibold tracking-tight">Configuration</h1>
        <p className="text-sm text-muted-foreground">
          Select one of your owned nodes, then drag models from the catalog into
          the VRAM container to plan assignments.
        </p>
        {isConfigLoading ? (
          <p
            data-testid="config-load-loading"
            className="text-xs text-muted-foreground"
          >
            Loading authored config from /api/config…
          </p>
        ) : null}
        {loadError ? (
          <p data-testid="config-load-error" className="text-xs text-destructive">
            {loadError}
          </p>
        ) : null}
      </div>
      <SaveConfig
        config={config}
        savedConfig={savedConfig}
        isDirty={isDirty}
        isConfigValid={isConfigValid}
        invalidReason={invalidReason}
        onSaveSuccess={onSaveSuccess}
        onRevert={onRevert}
        onBackendErrors={onBackendErrors}
        canUndo={canUndo}
        canRedo={canRedo}
        onUndo={onUndo}
        onRedo={onRedo}
      />
    </div>
  );
}
