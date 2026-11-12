/**
 * Browser latency of the input-check image encoder candidates, for the candidate study.
 *
 * Serves web/bench/ (vite.config.ts there) and opens it in Playwright Chromium once per model, runtime and
 * backend, in three profiles:
 *   desktop-headless  headless Chromium (WebGL through SwiftShader, software rendering)
 *   desktop-gpu       headed Chromium on this machine's GPU
 *   phone             Pixel 7 emulation with 4x CPU throttling (Lighthouse's mobile setting), headless
 * Models: every build/gate/<name>/tfjs/<variant>/model.json (TF.js, WebGL and WASM backends) and every
 * build/gate/<name>/onnx/*.onnx (ONNX Runtime Web, WASM). Writes results/input_check/browser_bench.json.
 *
 * Usage (from web/): npx tsx scripts/bench-gate.ts [--only <name>] [--profile <profile>] [--runtime tfjs|ort] [--variant <variant>]
 */
import { existsSync, readdirSync, statSync, writeFileSync, mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import { chromium, devices, type Browser, type BrowserContext } from '@playwright/test';
import { createServer } from 'vite';

const ROOT = join(import.meta.dirname, '..', '..');
const BUILD = join(ROOT, 'build', 'gate');
const OUT = join(ROOT, 'results', 'input_check', 'browser_bench.json');
const PORT = 4180;

type Job = {
  name: string;
  variant: string;
  runtime: 'tfjs' | 'ort';
  backend: string;
  url: string;
  bytes: number;
};

function arg(flag: string): string | undefined {
  const i = process.argv.indexOf(flag);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

function dirBytes(dir: string): { bytes: number; gzipBytes: number } {
  let bytes = 0;
  let gzipBytes = 0;
  for (const f of readdirSync(dir)) {
    const data = readFileSync(join(dir, f));
    bytes += data.length;
    gzipBytes += gzipSync(data).length;
  }
  return { bytes, gzipBytes };
}

function jobs(only?: string): Job[] {
  const out: Job[] = [];
  for (const name of readdirSync(BUILD)) {
    if (only && name !== only) continue;
    const tfjs = join(BUILD, name, 'tfjs');
    if (existsSync(tfjs)) {
      for (const variant of readdirSync(tfjs)) {
        const { bytes } = dirBytes(join(tfjs, variant));
        for (const backend of ['webgl', 'wasm']) {
          out.push({
            name,
            variant,
            runtime: 'tfjs',
            backend,
            url: `/gate/${name}/tfjs/${variant}/model.json`,
            bytes,
          });
        }
      }
    }
    const onnx = join(BUILD, name, 'onnx');
    if (existsSync(onnx)) {
      for (const f of readdirSync(onnx).filter((x) => x.endsWith('.onnx'))) {
        const variant = f.replace('image_encoder.', '').replace('.onnx', '');
        const bytes = statSync(join(onnx, f)).size;
        out.push({ name, variant, runtime: 'ort', backend: 'wasm', url: `/gate/${name}/onnx/${f}`, bytes });
      }
    }
  }
  return out;
}

async function context(profile: string): Promise<{ browser: Browser; ctx: BrowserContext }> {
  const browser = await chromium.launch({ headless: profile !== 'desktop-gpu' });
  const ctx = await browser.newContext(profile === 'phone' ? { ...devices['Pixel 7'] } : {});
  return { browser, ctx };
}

async function main() {
  const server = await createServer({ configFile: join(ROOT, 'web', 'bench', 'vite.config.ts') });
  await server.listen(PORT);
  const profiles = arg('--profile') ? [arg('--profile')!] : ['desktop-headless', 'desktop-gpu', 'phone'];
  const results: unknown[] = existsSync(OUT) ? JSON.parse(readFileSync(OUT, 'utf8')).results : [];
  const list = jobs(arg('--only')).filter(
    (j) =>
      (!arg('--runtime') || j.runtime === arg('--runtime')) &&
      (!arg('--variant') || j.variant === arg('--variant')),
  );
  const keep = results.filter(
    (r) =>
      !list.some(
        (j) =>
          (r as Job).name === j.name &&
          (r as Job).variant === j.variant &&
          (r as Job).runtime === j.runtime &&
          (r as Job).backend === j.backend &&
          profiles.includes((r as { profile: string }).profile),
      ),
  );
  const fresh: unknown[] = [];
  for (const profile of profiles) {
    for (const job of list) {
      const { browser, ctx } = await context(profile);
      const page = await ctx.newPage();
      if (profile === 'phone') {
        const cdp = await ctx.newCDPSession(page);
        await cdp.send('Emulation.setCPUThrottlingRate', { rate: 4 });
      }
      const q = new URLSearchParams({
        runtime: job.runtime,
        model: job.url,
        backend: job.backend,
        runs: '20',
      });
      await page.goto(`http://localhost:${PORT}/?${q}`);
      await page.waitForFunction(() => window.__benchResult !== undefined, null, { timeout: 600_000 });
      const r = (await page.evaluate(() => window.__benchResult)) as Record<string, unknown>;
      const row = { profile, ...job, ...r };
      fresh.push(row);
      console.log(
        `${profile.padEnd(16)} ${job.name.padEnd(30)} ${job.variant.padEnd(8)} ${job.runtime}/${job.backend.padEnd(6)} ` +
          (r.ok
            ? `load ${Number(r.loadMs).toFixed(0)} ms, warm-up ${Number(r.warmupMs).toFixed(0)} ms, median ${Number(r.medianMs).toFixed(1)} ms`
            : `FAILED ${r.error}`),
      );
      await browser.close();
    }
  }
  await server.close();
  const ortWasm = join(
    ROOT,
    'web',
    'bench',
    'node_modules',
    'onnxruntime-web',
    'dist',
    'ort-wasm-simd-threaded.wasm',
  );
  const doc = {
    description:
      'Browser latency of the input-check image encoder candidates (web/scripts/bench-gate.ts). Times in ms; ' +
      'loadMs includes fetching from a local server, so it measures parsing and weight upload, not network.',
    machine: { cpus: (await import('node:os')).cpus()[0]?.model, platform: process.platform },
    runtimeDownload: {
      'onnxruntime-web 1.20.1 ort-wasm-simd-threaded.wasm': existsSync(ortWasm)
        ? { bytes: statSync(ortWasm).size, gzipBytes: gzipSync(readFileSync(ortWasm)).length }
        : null,
    },
    results: [...keep, ...fresh],
  };
  mkdirSync(join(ROOT, 'results', 'input_check'), { recursive: true });
  writeFileSync(OUT, JSON.stringify(doc, null, 2) + '\n');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
