import { ChevronIcon } from './icons';

const TIPS = [
  'One dropping, photographed close up from above so it fills much of the frame, like the samples below.',
  'Daylight or bright, even light; no flash glare, in focus.',
  'One spot per photo. The model judges the whole picture, background included.',
  'The training photos were taken with phones on poultry farms in northern Tanzania. Other floors, litter and lighting are less familiar to the model.',
];

interface Props {
  open: boolean;
  onToggle: (open: boolean) => void;
}

export function PhotoGuidelines({ open, onToggle }: Props) {
  return (
    <details
      id="photo-guidelines"
      open={open}
      onToggle={(e) => onToggle((e.currentTarget as HTMLDetailsElement).open)}
      className="group rounded-xl border border-line bg-surface/70 px-4 py-3"
    >
      <summary className="flex scroll-mt-24 cursor-pointer list-none items-center justify-between gap-3 font-semibold text-ink [&::-webkit-details-marker]:hidden">
        Photos the model can work with
        <ChevronIcon className="shrink-0 text-ink-3 transition-transform group-open:rotate-180" />
      </summary>
      <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-ink-2">
        {TIPS.map((t) => (
          <li key={t}>{t}</li>
        ))}
      </ul>
    </details>
  );
}
