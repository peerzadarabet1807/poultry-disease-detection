import { REPO_URL, ZENODO_V2, ZENODO_V3 } from '../lib/links';
import { NOT_A_DIAGNOSIS } from '../lib/verdict';

export function Footer() {
  return (
    <footer className="mt-20 border-t border-line pb-[env(safe-area-inset-bottom)]">
      <div className="mx-auto max-w-6xl space-y-4 px-4 py-10 text-sm text-ink-3 sm:px-6">
        <p className="font-semibold text-ink" data-testid="disclaimer">
          {NOT_A_DIAGNOSIS}
        </p>
        <p className="max-w-3xl">
          Built by Peerzada Mohammad Rabet. Photos and labels: Machuve et al.,{' '}
          <a className="link" href={ZENODO_V2}>
            Machine Learning Dataset for Poultry Diseases Diagnostics
          </a>{' '}
          and its{' '}
          <a className="link" href={ZENODO_V3}>
            PCR-annotated version
          </a>{' '}
          (Zenodo, CC BY 4.0), collected in the Arusha and Kilimanjaro regions of Tanzania.
        </p>
        <p>
          <a href={REPO_URL} className="link font-medium">
            Source code
          </a>{' '}
          (MIT) · TensorFlow.js · Model version <code className="text-xs">{__MODEL_VERSION__}</code>
        </p>
      </div>
    </footer>
  );
}
