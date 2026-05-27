/** Open the Requirements Refinery whitepaper in a popup window.
 *
 * Mirrors the ``openRunDetail`` / ``openAboutDarkFactory`` pattern:
 * hash route (``#/about-requirements-refinery``) so the same Vite
 * bundle serves the popup, with a same-tab fallback if
 * ``window.open`` is blocked.
 */
export function openAboutRequirementsRefinery(): void {
  const url = `${window.location.pathname}#/about-requirements-refinery`;
  const features =
    "popup=yes,width=1280,height=900,resizable=yes,scrollbars=yes";
  const w = window.open(url, "about-requirements-refinery-window", features);
  if (!w) {
    window.open(url, "_blank");
  }
}
