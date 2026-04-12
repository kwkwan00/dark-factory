/** Open the per-run detail popup in a new window.
 *
 * Uses a hash route (``#/run-detail?run_id=…``) so the same Vite bundle
 * serves the popup — ``main.tsx`` inspects the hash and mounts
 * ``<RunDetailWindow/>`` instead of the full ``<App/>``. Falls back to a
 * same-tab navigation if the browser blocks ``window.open`` (popup
 * blockers) so the user isn't stranded.
 */
export function openRunDetail(runId: string): void {
  const url = `${window.location.pathname}#/run-detail?run_id=${encodeURIComponent(runId)}`;
  const features =
    "popup=yes,width=1400,height=900,resizable=yes,scrollbars=yes";
  const w = window.open(url, `run-detail-${runId}`, features);
  if (!w) {
    // Popup blocked — open in a new tab as a fallback.
    window.open(url, "_blank");
  }
}
