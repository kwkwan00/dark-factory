import { useEffect } from "react";
import AboutTab from "./AboutTab";

/** Popup-window wrapper around ``AboutTab``.
 *
 * Mounted by ``main.tsx::pickRoot`` when the URL hash is
 * ``#/about-dark-factory``. Reuses the existing AboutTab content
 * unchanged — this wrapper only supplies the popup chrome
 * (app-header + ``popup`` badge) so the window feels like part of
 * the same UI as ``RunDetailWindow``.
 *
 * Reachable from the Manufacture tab's Run History header via the
 * "About" button (``openAboutDarkFactory``).
 */
export default function AboutDarkFactory() {
  useEffect(() => {
    document.title = "About · AI Dark Factory";
  }, []);

  return (
    <>
      <header className="app-header">
        <h1>About Dark Factory</h1>
        <span className="badge">popup</span>
      </header>

      <main className="tab-content">
        <AboutTab />
      </main>
    </>
  );
}
