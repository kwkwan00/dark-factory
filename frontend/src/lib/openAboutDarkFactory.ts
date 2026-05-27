/** Open the About Dark Factory popup in a new window.
 *
 * Mirrors the ``openRunDetail`` pattern: hash route
 * (``#/about-dark-factory``) so the same Vite bundle serves the
 * popup, with a same-tab fallback if ``window.open`` is blocked.
 */
export function openAboutDarkFactory(): void {
  const url = `${window.location.pathname}#/about-dark-factory`;
  const features =
    "popup=yes,width=1200,height=900,resizable=yes,scrollbars=yes";
  const w = window.open(url, "about-dark-factory-window", features);
  if (!w) {
    window.open(url, "_blank");
  }
}
