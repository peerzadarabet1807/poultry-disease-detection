import { useEffect, useState } from 'react';
import { loadSamples, type SampleManifest } from '../lib/samples';

export type SamplesState =
  | { status: 'loading' }
  | { status: 'ready'; manifest: SampleManifest }
  | { status: 'error'; message: string };

export function useSamples(): SamplesState {
  const [state, setState] = useState<SamplesState>({ status: 'loading' });
  useEffect(() => {
    let alive = true;
    loadSamples().then(
      (manifest) => alive && setState({ status: 'ready', manifest }),
      (e: unknown) =>
        alive && setState({ status: 'error', message: e instanceof Error ? e.message : String(e) }),
    );
    return () => {
      alive = false;
    };
  }, []);
  return state;
}
