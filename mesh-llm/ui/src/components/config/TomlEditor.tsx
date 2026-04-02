import { Brackets, Check, X } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { codeToHtml } from 'shiki';

import { parseConfig, serializeConfig } from '../../lib/config';
import { cn } from '../../lib/utils';
import type { MeshConfig } from '../../types/config';

export interface TomlEditorProps {
  config: MeshConfig;
  onConfigChange: (config: MeshConfig) => void;
  onParseErrorChange?: (error: string | null) => void;
  onNormalizeConfig?: (config: MeshConfig) => { config: MeshConfig; modified: boolean };
  className?: string;
  panelClassName?: string;
}

const EDITOR_FONT_FAMILY = 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
const EDITOR_FONT_SIZE = '13px';
const EDITOR_LINE_HEIGHT = '1.5';
const EDITOR_PADDING = '12px';
const DEBOUNCE_MS = 150;
const SHIKI_DEBOUNCE_MS = 200;

export function TomlEditor({
  config,
  onConfigChange,
  onParseErrorChange,
  onNormalizeConfig,
  className,
  panelClassName,
}: TomlEditorProps) {
  const [tomlText, setTomlText] = useState<string>(() => serializeConfig(config));
  const [parseError, setParseError] = useState<string | null>(null);

  // Break feedback-loop: 'toml' = TOML side just dispatched; skip next config effect
  const syncSource = useRef<'toml' | null>(null);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shikiTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const highlightDivRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const lineCount = tomlText.split('\n').length;

  // Visual → TOML: whenever config prop changes, serialize and update editor
  useEffect(() => {
    if (syncSource.current === 'toml') {
      // The config changed because we dispatched from TOML side; skip to avoid
      // overwriting what the user is typing with the re-serialized form.
      syncSource.current = null;
      return;
    }
    const serialized = serializeConfig(config);
    setTomlText(serialized);
    setParseError(null);
    onParseErrorChange?.(null);
  }, [config, onParseErrorChange]);

  useEffect(() => {
    return () => {
      if (debounceTimerRef.current !== null) {
        clearTimeout(debounceTimerRef.current);
      }
      if (shikiTimerRef.current !== null) {
        clearTimeout(shikiTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (shikiTimerRef.current !== null) {
      clearTimeout(shikiTimerRef.current);
    }
    shikiTimerRef.current = setTimeout(() => {
      void codeToHtml(tomlText, { lang: 'toml', theme: 'github-dark' }).then((html) => {
        if (highlightDivRef.current) {
          highlightDivRef.current.innerHTML = html;
          const pre = highlightDivRef.current.querySelector('pre');
          if (pre) {
            pre.style.backgroundColor = 'transparent';
          }
        }
      });
    }, SHIKI_DEBOUNCE_MS);
    return () => {
      if (shikiTimerRef.current !== null) {
        clearTimeout(shikiTimerRef.current);
      }
    };
  }, [tomlText]);

  // TOML → Visual: debounced parse on every keystroke
  const handleTextareaChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const value = e.target.value;
      setTomlText(value);

      if (debounceTimerRef.current !== null) {
        clearTimeout(debounceTimerRef.current);
      }
      debounceTimerRef.current = setTimeout(() => {
        const parsed = parseConfig(value);
        if (parsed !== null) {
          let configToDispatch = parsed;
          let wasNormalized = false;

          if (onNormalizeConfig) {
            const result = onNormalizeConfig(parsed);
            configToDispatch = result.config;
            wasNormalized = result.modified;
          }

          setParseError(null);
          onParseErrorChange?.(null);

          // When normalized (e.g. ctx_size clamped), skip setting syncSource so
          // the config→TOML effect re-serializes the corrected value into the editor.
          if (!wasNormalized) {
            syncSource.current = 'toml';
          }

          onConfigChange(configToDispatch);
        } else {
          setParseError('Invalid TOML');
          onParseErrorChange?.('Invalid TOML');
        }
      }, DEBOUNCE_MS);
    },
    [onConfigChange, onNormalizeConfig, onParseErrorChange],
  );

  const handleScroll = useCallback(() => {
    if (textareaRef.current && highlightDivRef.current) {
      highlightDivRef.current.scrollTop = textareaRef.current.scrollTop;
      highlightDivRef.current.scrollLeft = textareaRef.current.scrollLeft;
    }
  }, []);

  return (
    <div
      className={cn(
        'flex min-h-0 flex-col overflow-hidden rounded-lg border border-border/70 bg-card shadow-soft',
        className,
      )}
    >
      <div className="flex w-full items-center justify-between gap-4 px-4 py-3 text-sm font-medium">
        <div className="flex min-w-0 items-center gap-3">
          <Brackets className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <span className="truncate font-mono text-xs">Configuration TOML</span>
          <span className="shrink-0 font-mono text-xs font-normal text-muted-foreground">{lineCount} lines</span>
        </div>
        <div className="shrink-0">
          {parseError !== null ? (
            <span
              data-testid="toml-error"
              className="flex items-center gap-1 text-xs text-destructive"
            >
              <X className="h-3 w-3 flex-none" aria-hidden="true" />
              <span>{parseError}</span>
            </span>
          ) : (
            <span data-testid="toml-valid" className="flex items-center gap-1 text-xs text-emerald-500">
              <Check className="h-3 w-3 flex-none" aria-hidden="true" />
            </span>
          )}
        </div>
      </div>

      <div
        data-testid="toml-editor-panel"
        className={cn(
          'relative min-h-[16rem] flex-1 resize-y overflow-hidden border-t border-border',
          panelClassName,
        )}
      >
          {/*
           * Syntax-highlighted layer — pointer-events-none, visually under the textarea.
           * The shiki output wraps in its own <pre>; we normalise its styles via Tailwind
           * arbitrary child selectors so it aligns exactly with the textarea content.
           */}
          <div
            ref={highlightDivRef}
            data-testid="toml-highlight"
            aria-hidden
            className="absolute inset-0 pointer-events-none overflow-auto [&_pre]:m-0 [&_pre]:font-mono [&_pre]:text-[13px] [&_pre]:leading-[1.5] [&_pre]:p-[12px] [&_pre]:whitespace-pre-wrap [&_pre]:[overflow-wrap:anywhere] [&_pre]:bg-transparent"
            style={{ scrollbarWidth: 'none' }}
          >
            <pre
              className="m-0"
              style={{
                padding: EDITOR_PADDING,
                fontFamily: EDITOR_FONT_FAMILY,
                fontSize: EDITOR_FONT_SIZE,
                lineHeight: EDITOR_LINE_HEIGHT,
                whiteSpace: 'pre-wrap',
                overflowWrap: 'anywhere',
              }}
            >
              <code>{tomlText}</code>
            </pre>
          </div>
          {/*
           * Transparent textarea overlay — same font metrics as the pre above.
           * Text is invisible (WebkitTextFillColor transparent) so the highlighted
           * layer shows through. The caret remains visible via caretColor.
           */}
          <textarea
            ref={textareaRef}
            data-testid="toml-textarea"
            value={tomlText}
            onChange={handleTextareaChange}
            onScroll={handleScroll}
            spellCheck={false}
            autoCapitalize="off"
            autoCorrect="off"
            className="absolute inset-0 w-full h-full resize-none border-none outline-none bg-transparent"
            style={{
              fontFamily: EDITOR_FONT_FAMILY,
              fontSize: EDITOR_FONT_SIZE,
              lineHeight: EDITOR_LINE_HEIGHT,
              padding: EDITOR_PADDING,
              color: 'transparent',
              WebkitTextFillColor: 'transparent',
              caretColor: 'hsl(var(--foreground))',
            }}
          />
        </div>
    </div>
  );
}
