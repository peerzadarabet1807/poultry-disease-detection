import type { ClassId } from './classes';

export interface SampleInfo {
  id: string;
  file: string;
  thumb: string;
  trueClass: ClassId;
  set: 'internal' | 'external';
  /** extra remark for the hard cases, empty otherwise */
  note: string;
  /** original file in the Zenodo dataset */
  source: string;
  doi: string;
}

export interface SampleManifest {
  credit: string;
  license: string;
  samples: SampleInfo[];
}

export const SAMPLES_BASE = `${import.meta.env.BASE_URL}samples/`;

export async function loadSamples(): Promise<SampleManifest> {
  const res = await fetch(`${SAMPLES_BASE}manifest.json`);
  if (!res.ok) throw new Error(`Could not load the sample images (HTTP ${res.status}).`);
  return (await res.json()) as SampleManifest;
}

export async function fetchSampleBlob(sample: SampleInfo): Promise<Blob> {
  const res = await fetch(`${SAMPLES_BASE}${sample.file}`);
  if (!res.ok) throw new Error(`Could not load ${sample.file} (HTTP ${res.status}).`);
  return res.blob();
}
