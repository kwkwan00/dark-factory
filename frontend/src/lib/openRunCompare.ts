/**
 * Open the run comparison in a popup window.
 * Uses the same hash-route pattern as openRunDetail.
 */
export function openRunCompare(runA: string, runB: string): void {
  const url = `${window.location.pathname}#/run-compare?run_a=${encodeURIComponent(runA)}&run_b=${encodeURIComponent(runB)}`;
  const features =
    "popup=yes,width=1400,height=900,resizable=yes,scrollbars=yes";
  const w = window.open(url, `run-compare-${runA}-${runB}`, features);
  if (!w) {
    window.open(url, "_blank");
  }
}
