import { CLASS_IDS, CLASS_INFO } from '../lib/classes';
import type { SetSummary } from '../lib/metrics';

/**
 * Confusion matrix as an accessible table: counts, shaded by the share of each true class (row). Shading is
 * capped at 45% so the ink-coloured numbers keep AA contrast in both themes.
 */
export function ConfusionMatrix({ set }: { set: SetSummary }) {
  return (
    <figure className="min-w-0 space-y-2">
      <figcaption className="text-sm font-semibold text-ink">
        {set.label} <span className="font-normal text-ink-3">(n = {set.n.toLocaleString('en')})</span>
      </figcaption>
      <div className="overflow-x-auto">
        <table className="tabular w-full border-separate border-spacing-1 text-center text-sm">
          <thead>
            <tr>
              <th scope="col" className="text-left text-xs font-medium text-ink-3">
                True ↓
              </th>
              {CLASS_IDS.map((id) => (
                <th key={id} scope="col" className="px-1 text-xs font-medium text-ink-2">
                  {CLASS_INFO[id].short}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {set.confusion.map((row, r) => {
              const total = row.reduce((a, b) => a + b, 0);
              return (
                <tr key={CLASS_IDS[r]}>
                  <th scope="row" className="pr-1 text-left text-xs font-medium text-ink-2">
                    {CLASS_INFO[CLASS_IDS[r]].short}
                  </th>
                  {row.map((count, c) => {
                    const share = total ? count / total : 0;
                    const diagonal = r === c;
                    return (
                      <td
                        key={CLASS_IDS[c]}
                        className={`rounded-md px-1 py-2 text-ink ${diagonal ? 'font-semibold' : ''}`}
                        style={{
                          background: `color-mix(in oklab, ${diagonal ? 'var(--accent)' : 'var(--disease-line)'} ${Math.round(
                            share * 45,
                          )}%, var(--surface-2))`,
                        }}
                        title={`${count} of ${total} ${CLASS_INFO[CLASS_IDS[r]].name} images (${(share * 100).toFixed(1)}%)`}
                      >
                        {count}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </figure>
  );
}
