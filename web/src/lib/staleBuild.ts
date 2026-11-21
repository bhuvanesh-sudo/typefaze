// After a redeploy, a tab opened earlier still points at old hashed code files that no longer exist.
// Vite fires "vite:preloadError" when such a dynamic import fails; reload once to pick up the new build.

const KEY = "typefaze-stale-reload";

export function installStaleBuildReload(target: EventTarget = window, reload: () => void = () => location.reload()) {
  target.addEventListener("vite:preloadError", (event) => {
    let already = false;
    try {
      already = sessionStorage.getItem(KEY) === "1";
      sessionStorage.setItem(KEY, "1");
    } catch {
      already = true; // no storage: don't risk a reload loop
    }
    if (already) return; // the error is real this time; let the app show it
    event.preventDefault();
    reload();
  });
}
