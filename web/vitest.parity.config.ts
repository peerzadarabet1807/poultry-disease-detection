import { defineConfig } from 'vitest/config';

// Parity test: Node environment (WASM backend), long timeouts. Runs on the committed model and fixtures.
export default defineConfig({
  test: {
    environment: 'node',
    globals: true,
    include: ['tests/parity.node.test.ts'],
    testTimeout: 600_000,
    hookTimeout: 120_000,
    pool: 'forks',
  },
});
