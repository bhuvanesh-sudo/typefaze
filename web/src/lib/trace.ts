// Bitmap -> font outline. Upsample 4x (bilinear), threshold, follow pixel boundaries into closed
// contours, simplify, then smooth into quadratic curves while keeping sharp corners sharp.
// Output is in font units: x relative to column 32, y up from the baseline (row 50).

import { Path } from "opentype.js";
import { BASELINE_ROW, CENTER_COL, EM_PX, SIZE, type GlyphBitmap } from "./glyphs";

const UP = 4;
const N = SIZE * UP; // 256
const UNITS_PER_PX = 1000 / EM_PX;
const SIMPLIFY_EPS = 0.9; // in upsampled pixels
const CORNER_COS = Math.cos((100 * Math.PI) / 180); // turns sharper than this stay corners

type Pt = [number, number];

function upsampleBinary(g: GlyphBitmap): Uint8Array {
  const out = new Uint8Array(N * N);
  const at = (x: number, y: number) =>
    g[Math.min(SIZE - 1, Math.max(0, y)) * SIZE + Math.min(SIZE - 1, Math.max(0, x))];
  for (let Y = 0; Y < N; Y++) {
    const sy = (Y + 0.5) / UP - 0.5;
    const y0 = Math.floor(sy);
    const fy = sy - y0;
    for (let X = 0; X < N; X++) {
      const sx = (X + 0.5) / UP - 0.5;
      const x0 = Math.floor(sx);
      const fx = sx - x0;
      const v =
        at(x0, y0) * (1 - fx) * (1 - fy) + at(x0 + 1, y0) * fx * (1 - fy) +
        at(x0, y0 + 1) * (1 - fx) * fy + at(x0 + 1, y0 + 1) * fx * fy;
      out[Y * N + X] = v > 0 ? 1 : 0;
    }
  }
  return out;
}

/** Follow the cracks between ink and paper. Every loop keeps ink on the same side, so holes wind opposite to outlines. */
function traceLoops(b: Uint8Array): Pt[][] {
  const ink = (x: number, y: number) => x >= 0 && y >= 0 && x < N && y < N && b[y * N + x] === 1;
  const out = new Map<number, number[]>(); // vertex key -> outgoing edge ids
  const edges: [number, number, number, number][] = [];
  const key = (x: number, y: number) => y * (N + 1) + x;
  const add = (x1: number, y1: number, x2: number, y2: number) => {
    const id = edges.push([x1, y1, x2, y2]) - 1;
    const k = key(x1, y1);
    (out.get(k) ?? out.set(k, []).get(k)!).push(id);
  };
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++) {
      if (!ink(x, y)) continue;
      if (!ink(x, y - 1)) add(x, y, x + 1, y);
      if (!ink(x + 1, y)) add(x + 1, y, x + 1, y + 1);
      if (!ink(x, y + 1)) add(x + 1, y + 1, x, y + 1);
      if (!ink(x - 1, y)) add(x, y + 1, x, y);
    }
  const used = new Uint8Array(edges.length);
  const loops: Pt[][] = [];
  for (let s = 0; s < edges.length; s++) {
    if (used[s]) continue;
    const loop: Pt[] = [];
    let e = s;
    while (!used[e]) {
      used[e] = 1;
      const [x1, y1, x2, y2] = edges[e];
      loop.push([x1, y1]);
      const cands = (out.get(key(x2, y2)) ?? []).filter((c) => !used[c]);
      if (!cands.length) break;
      if (cands.length === 1) e = cands[0];
      else {
        // at a diagonal touch, turn right so 4-connected regions stay separate
        const dx = x2 - x1, dy = y2 - y1;
        e = cands.find((c) => {
          const [a, b2, c2, d] = edges[c];
          return c2 - a === -dy && d - b2 === dx;
        }) ?? cands[0];
      }
    }
    if (loop.length >= 4) loops.push(loop);
  }
  return loops;
}

function dropCollinear(loop: Pt[]): Pt[] {
  const n = loop.length;
  return loop.filter((p, i) => {
    const a = loop[(i - 1 + n) % n], c = loop[(i + 1) % n];
    return (p[0] - a[0]) * (c[1] - p[1]) - (p[1] - a[1]) * (c[0] - p[0]) !== 0;
  });
}

function rdp(pts: Pt[], eps: number): Pt[] {
  if (pts.length < 3) return pts;
  const [ax, ay] = pts[0], [bx, by] = pts[pts.length - 1];
  const len = Math.hypot(bx - ax, by - ay) || 1;
  let maxD = 0, idx = 0;
  for (let i = 1; i < pts.length - 1; i++) {
    const d = Math.abs((bx - ax) * (ay - pts[i][1]) - (ax - pts[i][0]) * (by - ay)) / len;
    if (d > maxD) { maxD = d; idx = i; }
  }
  if (maxD <= eps) return [pts[0], pts[pts.length - 1]];
  return [...rdp(pts.slice(0, idx + 1), eps).slice(0, -1), ...rdp(pts.slice(idx), eps)];
}

function simplifyClosed(loop: Pt[]): Pt[] {
  if (loop.length < 4) return loop;
  // split at the point farthest from the first one so both halves are open polylines
  let far = 0, fd = -1;
  for (let i = 0; i < loop.length; i++) {
    const d = (loop[i][0] - loop[0][0]) ** 2 + (loop[i][1] - loop[0][1]) ** 2;
    if (d > fd) { fd = d; far = i; }
  }
  const a = rdp(loop.slice(0, far + 1), SIMPLIFY_EPS);
  const b = rdp([...loop.slice(far), loop[0]], SIMPLIFY_EPS);
  return [...a.slice(0, -1), ...b.slice(0, -1)];
}

const signedArea = (l: Pt[]) =>
  l.reduce((s, p, i) => { const q = l[(i + 1) % l.length]; return s + p[0] * q[1] - q[0] * p[1]; }, 0) / 2;

export function traceGlyph(g: GlyphBitmap): Path {
  const path = new Path();
  let loops = traceLoops(upsampleBinary(g)).map(dropCollinear).map(simplifyClosed).filter((l) => l.length >= 3);
  if (!loops.length) return path;

  // to font units: x from column 32, y up from the baseline
  const fontLoops = loops.map((l) =>
    l.map(([X, Y]): Pt => [(X / UP - CENTER_COL) * UNITS_PER_PX, (BASELINE_ROW - Y / UP) * UNITS_PER_PX]),
  );
  // PostScript outlines: outer contours counter-clockwise. Flip everything if the biggest contour is clockwise.
  const biggest = fontLoops.reduce((a, b) => (Math.abs(signedArea(b)) > Math.abs(signedArea(a)) ? b : a));
  const flip = signedArea(biggest) < 0;
  loops = fontLoops.map((l) => (flip ? [...l].reverse() : l));

  const r = (v: number) => Math.round(v * 10) / 10;
  for (const l of loops) {
    const n = l.length;
    const isCorner = l.map((p, i) => {
      const a = l[(i - 1 + n) % n], c = l[(i + 1) % n];
      const v1: Pt = [p[0] - a[0], p[1] - a[1]], v2: Pt = [c[0] - p[0], c[1] - p[1]];
      const cos = (v1[0] * v2[0] + v1[1] * v2[1]) / ((Math.hypot(...v1) * Math.hypot(...v2)) || 1);
      return cos < CORNER_COS;
    });
    const mid = (i: number): Pt => {
      const p = l[i % n], q = l[(i + 1) % n];
      return [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
    };
    // start at the midpoint of the last edge, then for each vertex: corner -> line through it, else curve
    const start = mid(n - 1);
    path.moveTo(r(start[0]), r(start[1]));
    for (let i = 0; i < n; i++) {
      const m = mid(i);
      if (isCorner[i]) {
        path.lineTo(r(l[i][0]), r(l[i][1]));
        path.lineTo(r(m[0]), r(m[1]));
      } else {
        path.quadraticCurveTo(r(l[i][0]), r(l[i][1]), r(m[0]), r(m[1]));
      }
    }
    path.close();
  }
  return path;
}
