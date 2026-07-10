/** DocuMind mark: a page with one highlighted line — the product's signature is showing
 *  exactly which passage an answer came from. Colours track the theme via CSS vars. */
export function Logo({ size = 28, className }: { size?: number; className?: string }) {
  const faint = "color-mix(in oklch, var(--ink) 32%, transparent)";
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      className={className}
      aria-hidden="true"
    >
      <rect x="6" y="3.5" width="20" height="25" rx="3.5" fill="var(--surface)" stroke="var(--ink)" strokeWidth="1.6" />
      <line x1="10.5" y1="10.5" x2="21.5" y2="10.5" stroke={faint} strokeWidth="1.6" strokeLinecap="round" />
      <rect x="9.5" y="14.3" width="13" height="3.4" rx="1.7" fill="var(--primary)" />
      <line x1="10.5" y1="21.7" x2="17.5" y2="21.7" stroke={faint} strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}
