/**
 * Capture the README screenshots from the running production build (never from mock-ups).
 *
 *   npm run build && npm run preview      # in one terminal
 *   npm run screenshots                   # in another  -> ../docs/screenshots/*.png
 *
 * Uses the real GPU where the machine has one (headless Chromium otherwise falls back to software WebGL,
 * which makes the inference times shown in the UI unrealistically slow).
 */
import { mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, devices, type Browser, type BrowserContextOptions, type Page } from '@playwright/test';

const BASE = process.env.BASE_URL ?? 'http://localhost:4173/poultry-disease-detection/';
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const OUT = join(ROOT, 'docs', 'screenshots');
const GPU_ARGS = ['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist'];
const DESKTOP: BrowserContextOptions = { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 };
/** result views: tall enough for the whole result card */
const DESKTOP_TALL: BrowserContextOptions = { viewport: { width: 1440, height: 1180 }, deviceScaleFactor: 1 };

async function launch(): Promise<Browser> {
  try {
    return await chromium.launch({ channel: 'chromium', args: GPU_ARGS });
  } catch {
    return chromium.launch();
  }
}

async function settle(page: Page, ms = 900) {
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(ms);
}

async function modelReady(page: Page) {
  await page.locator('[data-testid="model-status"][data-state="ready"]').waitFor({ timeout: 180_000 });
}

/** Scroll an element to just below the sticky header. */
async function scrollToId(page: Page, id: string) {
  await page.evaluate((target) => {
    const el = document.getElementById(target);
    const header = document.querySelector('header')?.getBoundingClientRect().height ?? 64;
    if (el)
      window.scrollTo({
        top: el.getBoundingClientRect().top + window.scrollY - header - 16,
        behavior: 'instant',
      });
  }, id);
}

async function showResult(page: Page) {
  await page.getByTestId('verdict').waitFor({ timeout: 120_000 });
  await settle(page, 1200);
  await scrollToId(page, 'result');
  await settle(page, 400);
}

async function shot(page: Page, name: string) {
  const path = join(OUT, name);
  await page.screenshot({ path });
  console.log('saved', path);
}

async function main() {
  mkdirSync(OUT, { recursive: true });
  const browser = await launch();

  // 01, 09: desktop 1440 x 900, light
  {
    const ctx = await browser.newContext({ ...DESKTOP, colorScheme: 'light' });
    const page = await ctx.newPage();
    await page.goto(BASE);
    await modelReady(page);
    await settle(page);
    await shot(page, '01-landing.png');
    await scrollToId(page, 'about');
    await settle(page);
    await shot(page, '09-about-model.png');
    await ctx.close();
  }

  // 03-08: results, light
  {
    const ctx = await browser.newContext({ ...DESKTOP_TALL, colorScheme: 'light' });
    const page = await ctx.newPage();
    await page.goto(BASE);
    await modelReady(page);
    const samples: [string, string][] = [
      ['f09', '03-result-coccidiosis.png'],
      ['f03', '04-result-healthy.png'],
      ['f05', '05-result-salmonella.png'],
      ['f07', '06-result-newcastle.png'],
      ['f18', '07-result-uncertain.png'],
    ];
    for (const [id, name] of samples) {
      await page.locator(`[data-sample="${id}"]`).click();
      await showResult(page);
      await shot(page, name);
    }
    // a photo that is not poultry droppings (CC0, see docs/ood/README.md): still a confident answer
    await page.getByTestId('file-input').setInputFiles(join(ROOT, 'docs', 'ood', 'cow_dung.jpg'));
    await showResult(page);
    await shot(page, '08-not-poultry.png');
    await ctx.close();
  }

  // 02: model download in progress
  {
    const ctx = await browser.newContext({ ...DESKTOP, colorScheme: 'light' });
    // shards download in parallel; stagger them so the progress bar moves visibly
    await ctx.route('**/model/group1-shard*', async (route) => {
      const n = Number(/shard(\d+)of/.exec(route.request().url())?.[1] ?? 1);
      await new Promise((r) => setTimeout(r, 700 * n));
      await route.continue();
    });
    const page = await ctx.newPage();
    await page.goto(BASE);
    await page.waitForFunction(() => {
      const bar = document.querySelector('[role="progressbar"]');
      return Number(bar?.getAttribute('aria-valuenow') ?? 0) >= 40;
    });
    await settle(page, 200);
    await shot(page, '02-model-loading.png');
    await ctx.close();
  }

  // 10: phone
  {
    const ctx = await browser.newContext({ ...devices['Pixel 7'], colorScheme: 'light' });
    const page = await ctx.newPage();
    await page.goto(BASE);
    await modelReady(page);
    await page.locator('[data-sample="f12"]').click();
    await showResult(page);
    await shot(page, '10-mobile.png');
    await ctx.close();
  }

  // 11: dark mode
  {
    const ctx = await browser.newContext({ ...DESKTOP_TALL, colorScheme: 'dark' });
    const page = await ctx.newPage();
    await page.goto(BASE);
    await modelReady(page);
    await page.locator('[data-sample="f11"]').click();
    await showResult(page);
    await shot(page, '11-dark-mode.png');
    await ctx.close();
  }

  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
