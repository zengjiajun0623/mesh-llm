import { render, screen } from '@testing-library/react';
import { useDraggable } from '@dnd-kit/react';
import { describe, expect, it, vi } from 'vitest';

import { DndContext } from '../DndContext';

if (!HTMLElement.prototype.setPointerCapture) {
  HTMLElement.prototype.setPointerCapture = () => {};
}

if (!HTMLElement.prototype.releasePointerCapture) {
  HTMLElement.prototype.releasePointerCapture = () => {};
}

if (!Document.prototype.elementFromPoint) {
  Document.prototype.elementFromPoint = () => document.body;
}

if (!Document.prototype.getAnimations) {
  Document.prototype.getAnimations = () => [];
}

if (!Element.prototype.getAnimations) {
  Element.prototype.getAnimations = () => [];
}

if (!window.matchMedia) {
  window.matchMedia = () => ({
    matches: false,
    media: '',
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  });
}

function setRect(element: HTMLElement, rect: Omit<DOMRect, 'toJSON'> & { toJSON?: () => unknown }) {
  Object.defineProperty(element, 'getBoundingClientRect', {
    configurable: true,
    value: () => ({
      ...rect,
      toJSON: rect.toJSON ?? (() => rect),
    }),
  });
}

function TestModelDraggable({ modelName, sizeBytes }: { modelName: string; sizeBytes: number }) {
  const { ref, handleRef } = useDraggable({
    id: `model:${modelName}`,
    data: { type: 'model', modelName, sizeBytes, nodeIds: [] },
  });

  return (
    <div ref={ref as React.RefCallback<HTMLDivElement>} data-testid="test-draggable">
      <button
        ref={handleRef as React.RefCallback<HTMLButtonElement>}
        type="button"
        data-testid="drag-handle"
      >
        Drag me
      </button>
    </div>
  );
}

describe('DndContext', () => {
  it('renders children within the drag-drop provider', () => {
    render(
      <DndContext selectedNodeId="node-a">
        <div data-testid="child">Hello</div>
      </DndContext>,
    );

    expect(screen.getByTestId('child')).toBeVisible();
    expect(screen.getByText('Hello')).toBeVisible();
  });

  it('renders children even when no node is selected', () => {
    render(
      <DndContext selectedNodeId={null}>
        <div data-testid="child">Content</div>
      </DndContext>,
    );

    expect(screen.getByTestId('child')).toBeVisible();
  });

  it('accepts an onAssignModel callback without error', () => {
    const onAssign = vi.fn();

    render(
      <DndContext selectedNodeId="node-a" onAssignModel={onAssign}>
        <div>Content</div>
      </DndContext>,
    );

    expect(screen.getByText('Content')).toBeVisible();
    expect(onAssign).not.toHaveBeenCalled();
  });

  it('renders a draggable child without error', () => {
    render(
      <DndContext selectedNodeId="node-a">
        <TestModelDraggable modelName="GLM-4.7-Flash-Q4_K_M" sizeBytes={10_000_000_000} />
      </DndContext>,
    );

    expect(screen.getByTestId('drag-handle')).toBeVisible();
  });
});
