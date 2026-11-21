import { beforeEach, describe, expect, it, vi } from "vitest";
import { installStaleBuildReload } from "./staleBuild";

describe("installStaleBuildReload", () => {
  beforeEach(() => sessionStorage.clear());

  it("reloads once when a code chunk from an old deploy is gone, but never loops", () => {
    const reload = vi.fn();
    const target = new EventTarget();
    installStaleBuildReload(target, reload);
    const fire = () => {
      const e = new Event("vite:preloadError", { cancelable: true });
      target.dispatchEvent(e);
      return e.defaultPrevented;
    };
    expect(fire()).toBe(true);
    expect(reload).toHaveBeenCalledTimes(1);
    fire();
    expect(reload).toHaveBeenCalledTimes(1); // second failure in the same tab: let the error show
  });
});
