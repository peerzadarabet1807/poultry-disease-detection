// Copy the committed results (the single source of truth in ../results, ../models, ../docs) into the web
// app. The About section and the verdict copy render these files; nothing in the UI hard-codes a metric.
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');
const root = join(web, '..');
const data = join(web, 'src', 'data');
const files = [
  ['models/class_names.json', 'class_names.json'],
  ['results/metrics.json', 'metrics.json'],
  ['results/experiment_summary.json', 'experiment_summary.json'],
  ['results/posthoc/bootstrap_ci.json', 'bootstrap_ci.json'],
  ['results/posthoc/calibration.json', 'calibration.json'],
  ['results/posthoc/thresholds.json', 'thresholds.json'],
  ['results/posthoc/run_variability.json', 'run_variability.json'],
  ['results/posthoc/external_confusions.json', 'external_confusions.json'],
  ['results/parity/tfjs_float16.json', 'tfjs_parity.json'],
  ['docs/ood/predictions.json', 'ood.json'],
];

mkdirSync(data, { recursive: true });
let ok = true;
for (const [from, to] of files) {
  const src = join(root, from);
  if (!existsSync(src)) {
    console.error(`missing ${from}`);
    ok = false;
    continue;
  }
  copyFileSync(src, join(data, to));
  console.log(`synced ${from} -> web/src/data/${to}`);
}
process.exit(ok ? 0 : 1);
