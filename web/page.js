// Whole image in, text in reading order out: det.onnx finds the lines, ocr.js reads each,
// then rows are ordered and fragments / rejected lines dropped (LineOCR.read in
// runtime/line_engine.py). Same steps as OCRCore's PageReader (Swift).
//
// ponytail: no page orientation (sideways / upside-down), deskew-and-redetect, gap splitting,
// column blocks or tall boxes read turned. A phone held upright over one column does not need
// them; add when testing says so.
import { createRecognizer, isAcceptable } from "./ocr.js";

export const DEFAULT_LIMIT = 1280;
// det.yml PostProcess values, which override the runtime defaults.
const THRESH = 0.2;
const BOX_THRESH = 0.4;
const UNCLIP_RATIO = 1.4;
const MIN_SIZE = 3;
const MAX_CANDIDATES = 3000;
// ImageNet statistics in the runtime's channel order: it feeds BGR.
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

/**
 * @typedef {[number, number]} Point
 * @typedef {Point[]} Quad corners top-left, top-right, bottom-right, bottom-left, in image pixels
 * @typedef {{ text: string, confidence: number, quad: Quad }} PageLine
 * @typedef {{ lines: PageLine[], text: string, detected: number, detMs: number, recMs: number }} PageReading
 */

/**
 * @param {typeof import("onnxruntime-web")} ort
 * @returns {Promise<{ read: (rgba: Uint8Array | Uint8ClampedArray, width: number, height: number, limit?: number) => Promise<PageReading> }>}
 */
export async function createPageReader(ort, detBytes, recBytes, dictText) {
  const detector = await ort.InferenceSession.create(detBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const recognizer = await createRecognizer(ort, recBytes, dictText);

  return {
    async read(rgba, width, height, limit = DEFAULT_LIMIT) {
      const start = performance.now();
      const input = detectorInput(rgba, width, height, limit);
      const tensor = new ort.Tensor("float32", input.values, [1, 3, input.height, input.width]);
      const { fetch_name_0: prob } = await detector.run({ x: tensor });
      const [, , ph, pw] = prob.dims;
      const quads = boxes(prob.data, pw, ph, input.paddedWidth / pw, input.paddedHeight / ph, input.pad, width, height);
      const detected = performance.now();

      const readings = [];
      for (const quad of quads) {
        const crop = cropQuad(rgba, width, height, quad);
        readings.push(await recognizer.read(crop.pixels, crop.width, crop.height));
      }
      const recognized = performance.now();

      const dropped = fragments(quads, readings.map((r) => r.text));
      const lines = [];
      const rowTexts = [];
      for (const row of rows(quads)) {
        const kept = row.filter((i) => !dropped.has(i) && isAcceptable(readings[i].text, readings[i].confidence));
        for (const i of kept) lines.push({ text: readings[i].text, confidence: readings[i].confidence, quad: quads[i] });
        if (kept.length) rowTexts.push(kept.map((i) => readings[i].text).join(" "));
      }
      return {
        lines,
        text: rowTexts.join("\n"),
        detected: quads.length,
        detMs: detected - start,
        recMs: recognized - detected,
      };
    },
  };
}

// ---------------------------------------------------------------------------------------
// Detector input: white border, then the padded image scaled so its long side is `limit`
// (multiples of 32, at most 4x), bilinear, normalised, channel-major B, G, R.

const roundHalfEven = (x) => {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; // Python round()
};

function detectorInput(rgba, width, height, limit) {
  const pad = Math.max(8, Math.trunc(0.05 * Math.min(width, height)));
  const w = width + 2 * pad;
  const h = height + 2 * pad;
  const s = Math.min(limit / Math.max(w, h), 4);
  const rw = Math.max(32, roundHalfEven((w * s) / 32) * 32);
  const rh = Math.max(32, roundHalfEven((h * s) / 32) * 32);

  const taps = (d, scale, size) => {
    const f = (d + 0.5) * scale - 0.5;
    let i = Math.floor(f);
    let t = f - i;
    if (i < 0) [i, t] = [0, 0];
    if (i >= size - 1) [i, t] = [size - 1, 0];
    return [i, Math.min(i + 1, size - 1), t];
  };
  const at = (x, y, c) => {
    const ix = x - pad;
    const iy = y - pad;
    return ix < 0 || iy < 0 || ix >= width || iy >= height ? 255 : rgba[(iy * width + ix) * 4 + c];
  };
  const xTaps = Array.from({ length: rw }, (_, x) => taps(x, w / rw, w));
  const plane = rw * rh;
  const values = new Float32Array(3 * plane);
  for (let y = 0; y < rh; y++) {
    const [y0, y1, ty] = taps(y, h / rh, h);
    for (let x = 0; x < rw; x++) {
      const [x0, x1, tx] = xTaps[x];
      for (let channel = 0; channel < 3; channel++) {
        const c = 2 - channel; // B, G, R from RGBA
        const top = at(x0, y0, c) * (1 - tx) + at(x1, y0, c) * tx;
        const bottom = at(x0, y1, c) * (1 - tx) + at(x1, y1, c) * tx;
        const byte = Math.round(top * (1 - ty) + bottom * ty);
        values[channel * plane + y * rw + x] = (byte / 255 - MEAN[channel]) / STD[channel];
      }
    }
  }
  return { values, width: rw, height: rh, pad, paddedWidth: w, paddedHeight: h };
}

// ---------------------------------------------------------------------------------------
// DB post-processing: threshold, one box per 8-connected region (its min-area rectangle),
// kept if wide enough and confident enough inside, grown by `unclip`, mapped to the image.
// ponytail: outer regions only; cv2.findContours(RETR_LIST) also returns hole contours, which
// on line blobs are rare and score low.

function boxes(prob, width, height, scaleX, scaleY, pad, imageWidth, imageHeight) {
  const visited = new Uint8Array(width * height);
  const stack = [];
  const quads = [];
  let candidates = 0;
  for (let start = 0; start < width * height; start++) {
    if (prob[start] <= THRESH || visited[start]) continue;
    if (++candidates > MAX_CANDIDATES) break;
    const extents = new Map(); // y -> [leftmost, rightmost]
    visited[start] = 1;
    stack.push(start);
    while (stack.length) {
      const i = stack.pop();
      const x = i % width;
      const y = (i - x) / width;
      const e = extents.get(y);
      extents.set(y, e ? [Math.min(e[0], x), Math.max(e[1], x)] : [x, x]);
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const n = ny * width + nx;
          if (!visited[n] && prob[n] > THRESH) {
            visited[n] = 1;
            stack.push(n);
          }
        }
      }
    }
    const points = [...extents].flatMap(([y, [x0, x1]]) => [[x0, y], [x1, y]]);
    const rect = minAreaRect(points);
    if (Math.min(rect.w, rect.h) < MIN_SIZE || score(prob, width, height, corners(rect)) < BOX_THRESH) continue;
    // ponytail: float geometry. pyclipper truncates the corners to integers first, which moves the
    // runtime's boxes up-left by under a pixel; copying that flipped other characters.
    const perimeter = 2 * (rect.w + rect.h);
    if (perimeter === 0) continue;
    const d = (rect.w * rect.h * UNCLIP_RATIO) / perimeter;
    const grown = { ...rect, w: rect.w + 2 * d, h: rect.h + 2 * d };
    if (Math.min(grown.w, grown.h) < MIN_SIZE + 2) continue;
    const quad = corners(grown).map(([x, y]) => [
      Math.min(Math.max(x * scaleX - pad, 0), imageWidth - 1),
      Math.min(Math.max(y * scaleY - pad, 0), imageHeight - 1),
    ]);
    const xs = quad.map((p) => p[0]);
    const ys = quad.map((p) => p[1]);
    if (Math.max(...xs) - Math.min(...xs) >= 2 && Math.max(...ys) - Math.min(...ys) >= 2) quads.push(quad);
  }
  return quads;
}

/** _box_score: mean probability inside the box (corners truncated to whole pixels, as fillPoly). */
function score(prob, width, height, box) {
  const xs = box.map((p) => p[0]);
  const ys = box.map((p) => p[1]);
  const clamp = (v, max) => Math.min(Math.max(v, 0), max);
  const x0 = clamp(Math.floor(Math.min(...xs)), width - 1);
  const x1 = clamp(Math.ceil(Math.max(...xs)), width - 1);
  const y0 = clamp(Math.floor(Math.min(...ys)), height - 1);
  const y1 = clamp(Math.ceil(Math.max(...ys)), height - 1);
  const poly = box.map(([x, y]) => [Math.trunc(x - x0), Math.trunc(y - y0)]);
  let sum = 0;
  let count = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      if (!contains(poly, [x - x0, y - y0])) continue;
      sum += prob[y * width + x];
      count++;
    }
  }
  return count ? sum / count : 0;
}

// ---------------------------------------------------------------------------------------
// Geometry.

/** order_points: sort by x; the left pair and the right pair each by y. */
function ordered(points) {
  const byX = [...points].sort((a, b) => a[0] - b[0]);
  const [l0, l1] = byX.slice(0, 2).sort((a, b) => a[1] - b[1]);
  const [r0, r1] = byX.slice(2).sort((a, b) => a[1] - b[1]);
  return [l0, r0, r1, l1];
}

/** cv2.boxPoints + order_points. */
function corners({ cx, cy, ax, ay, w, h }) {
  const [nx, ny] = [-ay, ax];
  return ordered(
    [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]].map(([u, v]) => [
      cx + u * ax + v * nx,
      cy + u * ay + v * ny,
    ]),
  );
}

/** Andrew's monotone chain, counter-clockwise, no collinear points. */
function convexHull(points) {
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length <= 2) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (list) => {
    const out = [];
    for (const q of list) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], q) <= 0) out.pop();
      out.push(q);
    }
    out.pop();
    return out;
  };
  return [...half(p), ...half([...p].reverse())];
}

/** cv2.minAreaRect: one side lies on a hull edge (rotating calipers). */
export function minAreaRect(points) {
  const hull = convexHull(points);
  if (hull.length < 2) return { cx: hull[0]?.[0] ?? 0, cy: hull[0]?.[1] ?? 0, ax: 1, ay: 0, w: 0, h: 0 };
  let best = null;
  let bestArea = Infinity;
  hull.forEach((a, i) => {
    const b = hull[(i + 1) % hull.length];
    const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (!length) return;
    const [ax, ay] = [(b[0] - a[0]) / length, (b[1] - a[1]) / length];
    let [minU, maxU, minV, maxV] = [Infinity, -Infinity, Infinity, -Infinity];
    for (const [px, py] of hull) {
      const u = (px - a[0]) * ax + (py - a[1]) * ay;
      const v = -(px - a[0]) * ay + (py - a[1]) * ax;
      [minU, maxU, minV, maxV] = [Math.min(minU, u), Math.max(maxU, u), Math.min(minV, v), Math.max(maxV, v)];
    }
    const area = (maxU - minU) * (maxV - minV);
    if (area < bestArea) {
      bestArea = area;
      const [cu, cv] = [(minU + maxU) / 2, (minV + maxV) / 2];
      best = { cx: a[0] + cu * ax - cv * ay, cy: a[1] + cu * ay + cv * ax, ax, ay, w: maxU - minU, h: maxV - minV };
    }
  });
  // Every edge zero-length: a 1-pixel region (its leftmost and rightmost pixel coincide).
  return best ?? { cx: hull[0][0], cy: hull[0][1], ax: 1, ay: 0, w: 0, h: 0 };
}

/** Point in a convex quad, edges included. */
function contains(quad, [px, py]) {
  let positive = false;
  let negative = false;
  for (let i = 0; i < 4; i++) {
    const [ax, ay] = quad[i];
    const [bx, by] = quad[(i + 1) % 4];
    const cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    if (cross > 0) positive = true;
    if (cross < 0) negative = true;
  }
  return !(positive && negative);
}

const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** crop_line: the quad warped to an upright w x h image, bicubic (cv2.INTER_CUBIC), border replicated. */
function cropQuad(rgba, width, height, quad) {
  const w = Math.max(1, Math.trunc(Math.max(distance(quad[0], quad[1]), distance(quad[2], quad[3]))));
  const h = Math.max(1, Math.trunc(Math.max(distance(quad[0], quad[3]), distance(quad[1], quad[2]))));
  const map = squareToQuad(quad);
  const pixels = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [sx, sy] = map(x / w, y / h);
      sampleBicubic(rgba, width, height, sx, sy, pixels, (y * w + x) * 4);
    }
  }
  return { pixels, width: w, height: h };
}

/** Projective map of the unit square onto the quad, corner for corner (Heckbert). */
function squareToQuad(q) {
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = q;
  const sx = x0 - x1 + x2 - x3;
  const sy = y0 - y1 + y2 - y3;
  const [dx1, dx2, dy1, dy2] = [x1 - x2, x3 - x2, y1 - y2, y3 - y2];
  const den = dx1 * dy2 - dx2 * dy1;
  const affine = (sx === 0 && sy === 0) || den === 0;
  const g = affine ? 0 : (sx * dy2 - dx2 * sy) / den;
  const hh = affine ? 0 : (dx1 * sy - sx * dy1) / den;
  const [a, b, c] = [x1 - x0 + g * x1, x3 - x0 + hh * x3, x0];
  const [d, e, f] = [y1 - y0 + g * y1, y3 - y0 + hh * y3, y0];
  return (u, v) => {
    const z = g * u + hh * v + 1;
    return [(a * u + b * v + c) / z, (d * u + e * v + f) / z];
  };
}

/** cv2's cubic kernel, A = -0.75: weights for the 4 taps around a fractional offset t. */
function cubicWeights(t) {
  const A = -0.75;
  const w0 = ((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A;
  const w1 = ((A + 2) * t - (A + 3)) * t * t + 1;
  const w2 = ((A + 2) * (1 - t) - (A + 3)) * (1 - t) * (1 - t) + 1;
  return [w0, w1, w2, 1 - w0 - w1 - w2];
}

function sampleBicubic(src, w, h, x, y, out, offset) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return; // degenerate quad: leave the pixel black
  const fx = Math.floor(x);
  const fy = Math.floor(y);
  const wx = cubicWeights(x - fx);
  const wy = cubicWeights(y - fy);
  const xs = [-1, 0, 1, 2].map((k) => Math.min(Math.max(fx + k, 0), w - 1));
  const ys = [-1, 0, 1, 2].map((k) => Math.min(Math.max(fy + k, 0), h - 1));
  for (let c = 0; c < 4; c++) {
    let value = 0;
    for (let j = 0; j < 4; j++) {
      let row = 0;
      for (let i = 0; i < 4; i++) row += src[(ys[j] * w + xs[i]) * 4 + c] * wx[i];
      value += row * wy[j];
    }
    out[offset + c] = Math.min(255, Math.max(0, Math.round(value)));
  }
}

// ---------------------------------------------------------------------------------------
// Reading order and fragments.

function median(values) {
  const s = [...values].sort((a, b) => a - b);
  if (!s.length) return 0;
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

const extent = (points) => {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
};

/** line_height: median height of line-shaped boxes (at least twice as wide as tall). */
function lineHeight(quads) {
  if (!quads.length) return 1;
  const heights = quads.map((q) => Math.max(distance(q[0], q[3]), 1));
  const lines = heights.filter((hgt, i) => distance(quads[i][0], quads[i][1]) >= 2 * hgt);
  return median(lines.length >= 3 ? lines : heights);
}

/** page_skew: median angle in degrees of boxes at least 3x wider than tall. */
function skew(quads) {
  const angles = quads
    .filter((q) => Math.hypot(q[1][0] - q[0][0], q[1][1] - q[0][1]) >= 3 * Math.max(distance(q[0], q[3]), 1))
    .map((q) => (Math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0]) * 180) / Math.PI);
  return median(angles);
}

/**
 * reading_order without column blocks: rows top to bottom by vertical overlap in deskewed
 * coordinates, each left to right. Small boxes join the nearest row without moving it.
 */
function rows(quads) {
  if (!quads.length) return [];
  const a = (-skew(quads) * Math.PI) / 180;
  const [c, s] = [Math.cos(a), Math.sin(a)];
  const ext = quads.map((q) => extent(q.map(([x, y]) => [x * c - y * s, x * s + y * c])));
  const hmed = lineHeight(quads);
  const cy = ext.map((e) => (e.y0 + e.y1) / 2);
  const rx = ext.map((e) => (e.x0 + e.x1) / 2);
  const hs = quads.map((q) => Math.max(distance(q[0], q[3]), 1));
  const tol = 0.5 * hmed;
  const mean = (row) => row.reduce((sum, i) => sum + cy[i], 0) / row.length;

  const indices = quads.map((_, i) => i);
  const tall = indices.filter((i) => hs[i] >= 0.6 * hmed);
  const full = tall.length ? tall : indices;
  const small = indices.filter((i) => !full.includes(i));
  const out = [];
  for (const i of [...full].sort((p, q) => cy[p] - cy[q])) {
    if (out.length && Math.abs(cy[i] - mean(out[out.length - 1])) <= tol) out[out.length - 1].push(i);
    else out.push([i]);
  }
  const centres = out.map(mean);
  for (const i of small) {
    const d = centres.map((cc) => Math.abs(cy[i] - cc));
    const nearest = d.indexOf(Math.min(...d));
    if (d.length && d[nearest] <= tol + 0.5 * hs[i]) {
      out[nearest].push(i);
    } else {
      out.push([i]);
      centres.push(cy[i]);
    }
  }
  return out
    .map((row, r) => [row, centres[r]])
    .sort((p, q) => p[1] - q[1])
    .map(([row]) => [...row].sort((p, q) => rx[p] - rx[q]));
}

/**
 * _fragments: boxes that are pieces of a neighbouring line (a stacked mark or a tail detected
 * alone): at most 3 characters, under 35 % of the line height, touching a full-height line
 * above or below.
 */
function fragments(quads, texts) {
  const out = new Set();
  if (quads.length < 3) return out;
  const hs = quads.map((q) => distance(q[0], q[3]));
  const hmed = lineHeight(quads);
  const ext = quads.map(extent);
  quads.forEach((_, i) => {
    if (hs[i] >= 0.35 * hmed || [...texts[i].replaceAll(" ", "")].length > 3) return;
    const b = ext[i];
    const touches = quads.some((__, j) => {
      if (j === i || hs[j] < 0.6 * hmed) return false;
      const e = ext[j];
      return e.x0 <= b.x1 && b.x0 <= e.x1 && Math.min(Math.abs(b.y0 - e.y1), Math.abs(e.y0 - b.y1)) <= 0.5 * hmed;
    });
    if (touches) out.add(i);
  });
  return out;
}
