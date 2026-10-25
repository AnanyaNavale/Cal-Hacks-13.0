// Explicitly export Layer so StackLayer.tsx can resolve it
export type Layer = 'ui' | 'api' | 'logic' | 'data' | 'auth' | 'infra';

export type LayerUpdate = {
  t: number;
  layer: Layer;
  status: 'new' | 'changed' | 'unchanged';
  files: string[];
  summary?: string;
  flags: ('new_dependency' | 'touches_auth')[];
};
