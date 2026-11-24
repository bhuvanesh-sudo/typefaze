import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const css = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf8");

describe("styles", () => {
  it("uses the Ember palette with the bone accent", () => {
    for (const hex of ["#14110e", "#1c1814", "#2d2721", "#f1ebe2", "#9a9082", "#5a4a35", "#d6c3a3"]) {
      expect(css.toLowerCase(), hex).toContain(hex);
    }
  });

  it("turns motion off for people who ask for less", () => {
    expect(css).toContain("prefers-reduced-motion: reduce");
  });

  it("uses the spring easing for presses and arrivals", () => {
    expect(css).toContain("cubic-bezier(.34, 1.56, .64, 1)");
  });

  it("shows the tracing outline fully drawn (a draw-in animation would be blocked by the font build)", () => {
    expect(css).not.toContain("draw-outline");
    expect(css).not.toMatch(/\.trace path[^}]*stroke-dashoffset/);
  });

  it("draws with a simple dot cursor, not a crosshair", () => {
    const rule = css.match(/\.pad-stack canvas \{[^}]*\}/)?.[0] ?? "";
    expect(rule).toMatch(/cursor: url\("data:image\/svg\+xml[^)]*circle/);
    expect(rule).not.toMatch(/cursor: crosshair/);
  });
});
