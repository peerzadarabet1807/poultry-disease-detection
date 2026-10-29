/**
 * The same app built with VITE_INFERENCE_MODE=api, against a running backend (docker compose up). Opt-in:
 *
 *   docker compose up -d
 *   VITE_INFERENCE_MODE=api VITE_API_URL=http://localhost:8000 npx vite build --outDir dist-api
 *   npx vite preview --outDir dist-api --port 4174 &
 *   E2E_API=1 npx playwright test --project=api
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';

const FIXTURES = fileURLToPath(new URL('../fixtures/', import.meta.url));
const { classNames, samples } = JSON.parse(
  readFileSync(join(FIXTURES, 'sample-references.json'), 'utf8'),
) as {
  classNames: string[];
  samples: { id: string; kerasPredicted: string; kerasProbabilities: number[]; category: string }[];
};

test('API engine gives the same result card as the browser engine', async ({ page }) => {
  await page.goto('./');
  await expect(page.getByTestId('model-status')).toHaveAttribute('data-state', 'ready');
  await expect(page.getByTestId('backend')).toContainText('Server inference');
  await expect(page.getByText('Your photo is sent to the API')).toBeVisible();

  for (const id of ['f07', 'f18']) {
    const ref = samples.find((s) => s.id === id)!;
    await page.locator(`[data-sample="${id}"]`).click();
    const verdict = page.getByTestId('verdict');
    await expect(verdict).toHaveAttribute('data-category', ref.category);
    // f18 is split almost evenly; the server's training-path JPEG decode tips it to NCD, the browser's to
    // Healthy (models/PARITY.md). Both say "uncertain", which is what the user sees.
    if (ref.category !== 'uncertain') await expect(verdict).toHaveAttribute('data-top', ref.kerasPredicted);
    const rows = page.getByTestId('prob-row');
    await expect(rows).toHaveCount(4);
    const probs: Record<string, number> = {};
    for (const row of await rows.all()) {
      probs[(await row.getAttribute('data-class'))!] = Number(await row.getAttribute('data-probability'));
    }
    // the server decodes JPEGs with TensorFlow's default (training) IDCT, the references use the browser's
    const diff = Math.max(...classNames.map((c, i) => Math.abs(probs[c] - ref.kerasProbabilities[i])));
    expect(diff).toBeLessThan(0.15);
    await expect(page.getByTestId('cam-canvas')).toBeVisible();
  }
});
