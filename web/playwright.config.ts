import { defineConfig, devices } from '@playwright/test';

const BASE = 'http://localhost:4173/poultry-disease-detection/';

/**
 * E2E tests run against the production build (`npm run build` first). Firefox, WebKit and an emulated phone
 * run the tests tagged @cross-browser; select projects with --project. Headless Chromium uses software WebGL
 * (SwiftShader), so model loading and inference are slower than on a real GPU; timeouts are generous.
 */
export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 300_000,
  expect: { timeout: 120_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: BASE,
    trace: 'retain-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] }, grep: /@cross-browser/ },
    { name: 'webkit', use: { ...devices['Desktop Safari'] }, grep: /@cross-browser/ },
    { name: 'mobile', use: { ...devices['Pixel 7'] }, grep: /@cross-browser/ },
  ],
  webServer: {
    command: 'npm run preview',
    url: BASE,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
