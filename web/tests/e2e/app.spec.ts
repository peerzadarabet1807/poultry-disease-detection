import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';
import { readNpy } from '../utils/npy';

interface SampleReference {
  id: string;
  file: string;
  kerasPredicted: string;
  kerasProbabilities: number[];
  category: string;
}

const FIXTURES = fileURLToPath(new URL('../fixtures/', import.meta.url));
const ROOT = join(FIXTURES, '..', '..', '..');
const { classNames, samples: references } = JSON.parse(
  readFileSync(join(FIXTURES, 'sample-references.json'), 'utf8'),
) as { classNames: string[]; samples: SampleReference[] };
const reference = (id: string) => references.find((r) => r.id === id)!;
/** Keras (accurate-IDCT decode) vs TF.js float16 on WebGL/WASM, browser decode. */
const MAX_PROB_DIFF = 0.03;

async function waitForModel(page: Page) {
  await expect(page.getByTestId('model-status')).toHaveAttribute('data-state', 'ready', { timeout: 240_000 });
}

async function classifySample(page: Page, id: string) {
  await page.locator(`[data-sample="${id}"]`).click();
  await expect(page.getByTestId('result')).toBeVisible();
  await expect(page.getByTestId('result')).toContainText(`Dataset label`);
}

async function readProbabilities(page: Page): Promise<number[]> {
  const rows = page.getByTestId('prob-row');
  await expect(rows).toHaveCount(4);
  const byClass: Record<string, number> = {};
  for (const row of await rows.all()) {
    byClass[(await row.getAttribute('data-class'))!] = Number(await row.getAttribute('data-probability'));
  }
  return classNames.map((c) => byClass[c]);
}

async function expectMatchesReference(page: Page, id: string): Promise<number> {
  const ref = reference(id);
  const verdict = page.getByTestId('verdict');
  await expect(verdict).toHaveAttribute('data-top', ref.kerasPredicted);
  await expect(verdict).toHaveAttribute('data-category', ref.category);
  const probs = await readProbabilities(page);
  const diff = Math.max(...probs.map((p, i) => Math.abs(p - ref.kerasProbabilities[i])));
  expect(diff, `${id}: max |prob diff| vs Keras`).toBeLessThanOrEqual(MAX_PROB_DIFF);
  return diff;
}

async function tensorCount(page: Page): Promise<number> {
  return page.evaluate(async () => {
    const debug = (globalThis as unknown as { __poultryDebug: { memory(): Promise<{ numTensors: number }> } })
      .__poultryDebug;
    return (await debug.memory()).numTensors;
  });
}

async function uploadFile(page: Page, path: string) {
  await page.getByTestId('file-input').setInputFiles(path);
}

test.describe('poultry droppings classifier', () => {
  test('classifies samples like Keras, with responsible copy @cross-browser', async ({ page }) => {
    await page.goto('./');
    await expect(page.getByRole('heading', { level: 1 })).toContainText('What do these droppings');
    await expect(page.getByText('not a veterinary diagnosis').first()).toBeVisible();
    await waitForModel(page);

    await classifySample(page, 'f07'); // Newcastle disease
    await expectMatchesReference(page, 'f07');
    await expect(page.getByTestId('verdict-title')).toHaveText('Pattern consistent with Newcastle disease');
    await expect(page.getByTestId('ncd-notice')).toContainText('notifiable');
    await expect(page.getByTestId('live-region')).toContainText(
      'Result: Pattern consistent with Newcastle disease',
    );

    await classifySample(page, 'f03'); // Healthy
    await expectMatchesReference(page, 'f03');
    await expect(page.getByTestId('verdict')).toContainText('not a clean bill of health');
    await expect(page.getByTestId('ncd-notice')).toHaveCount(0);

    await classifySample(page, 'f18'); // split between two classes
    await expectMatchesReference(page, 'f18');
    await expect(page.getByTestId('verdict-title')).toHaveText('Uncertain: no verdict shown');
    await expect(page.getByTestId('verdict')).toContainText('usability heuristic');
    await expect(page.getByTestId('cam-canvas')).toBeVisible();
  });

  test('every sample matches its Keras reference', async ({ page }) => {
    await page.goto('./');
    await waitForModel(page);
    const diffs: Record<string, number> = {};
    for (const ref of references) {
      await classifySample(page, ref.id);
      diffs[ref.id] = await expectMatchesReference(page, ref.id);
    }
    console.log('max |prob diff| per sample:', JSON.stringify(diffs));
  });

  test('browser preprocessing reproduces the Python model input @cross-browser', async ({ page }, info) => {
    const fixtures = JSON.parse(readFileSync(join(FIXTURES, 'reference', 'reference.json'), 'utf8')) as {
      images: { id: string; file: string; seen: string; kerasPredicted: string }[];
    };
    await page.goto('./');
    await waitForModel(page);
    const rows = [];
    for (const id of ['f12', 'f05', 'f01']) {
      const ref = fixtures.images.find((r) => r.id === id)!;
      await uploadFile(page, join(ROOT, 'tests', 'fixtures', 'images', ref.file));
      await expect(page.getByTestId('verdict')).toBeVisible();
      const seen = await page.evaluate(() =>
        (
          globalThis as unknown as { __poultryDebug: { lastSeenRgb(): number[] } }
        ).__poultryDebug.lastSeenRgb(),
      );
      const expected = readNpy(readFileSync(join(FIXTURES, 'reference', ref.seen))).data as Uint8Array;
      let differing = 0;
      let maxDiff = 0;
      for (let i = 0; i < expected.length; i++) {
        const d = Math.abs(seen[i] - expected[i]);
        if (d) differing++;
        maxDiff = Math.max(maxDiff, d);
      }
      const top = await page.getByTestId('verdict').getAttribute('data-top');
      rows.push({ id, file: ref.file, valuesDiffering: differing, maxDiff, top, keras: ref.kerasPredicted });
      expect(top, `${id}: top class`).toBe(ref.kerasPredicted);
      await page.getByRole('button', { name: 'Classify another photo' }).click();
    }
    const report = {
      description:
        'Model input (224 x 224 RGB) computed in the browser from the original JPEG (browser decoder + TypeScript ' +
        'resize) vs the Python package on the accurate-IDCT decode. 150,528 values per image.',
      project: info.project.name,
      browser: page.context().browser()?.browserType().name(),
      images: rows,
    };
    console.log(JSON.stringify(report));
    mkdirSync(join(ROOT, 'results', 'parity'), { recursive: true });
    writeFileSync(
      join(ROOT, 'results', 'parity', `browser_input_${info.project.name}.json`),
      JSON.stringify(report, null, 2) + '\n',
    );
  });

  test('rejects files that are not images @cross-browser', async ({ page }) => {
    await page.goto('./');
    await uploadFile(page, join(FIXTURES, 'not-an-image.txt'));
    await expect(page.getByTestId('upload-error')).toContainText("doesn't look like a photo");
  });

  test('reads HEIC photos @cross-browser', async ({ page }) => {
    await page.goto('./');
    await waitForModel(page);
    await uploadFile(page, join(FIXTURES, 'sample.heic'));
    await expect(page.getByTestId('verdict')).toBeVisible();
  });

  test('explains the EXIF rotation tag @cross-browser', async ({ page }) => {
    await page.goto('./');
    await waitForModel(page);
    await uploadFile(page, join(FIXTURES, 'exif-rotated.jpg'));
    await expect(page.getByTestId('input-notes')).toContainText('rotation tag');
  });

  test('does not leak tensors', async ({ page }) => {
    await page.goto('./');
    await waitForModel(page);
    await classifySample(page, 'f01');
    const after1 = await tensorCount(page);
    for (const id of ['f05', 'f09', 'f12']) await classifySample(page, id);
    expect(await tensorCount(page)).toBe(after1);
  });

  test('runs on the WebAssembly backend', async ({ page }) => {
    await page.goto('./?backend=wasm');
    await waitForModel(page);
    await expect(page.getByTestId('backend')).toContainText('WebAssembly');
    await classifySample(page, 'f09');
    await expectMatchesReference(page, 'f09');
  });

  test('is keyboard accessible @cross-browser', async ({ page, isMobile, browserName }) => {
    test.skip(isMobile, 'no keyboard on the phone profile');
    // Playwright's WebKit build on Windows never moves focus off <body> with Tab, even to buttons;
    // keyboard navigation is covered by Chromium and Firefox.
    test.skip(browserName === 'webkit' && process.platform === 'win32', 'WebKit on Windows ignores Tab');
    await page.goto('./');
    await page.keyboard.press('Tab');
    await expect(page.getByRole('link', { name: 'Skip to content' })).toBeFocused();
    await waitForModel(page);
    const first = page.getByTestId('sample').first();
    await first.focus();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('heading', { name: /^Result/ })).toBeFocused();
    await page.getByRole('button', { name: /Switch to dark theme/ }).click();
    await expect(page.locator('html')).toHaveClass(/dark/);
  });
});
