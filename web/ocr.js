// One text line in, text out: rec.onnx on ONNX Runtime Web, greedy CTC over rec_dict.txt.
// Port of runtime/line_engine.py + runtime/thai_norm.py (iapp/OpenThai2.0-OCR-Text),
// the same steps as OCRCore (Swift). `ort` is passed in, so the browser and Node tests
// can each load their own build.

export const HEIGHT = 48;
const MIN_WIDTH = 16;
const MAX_WIDTH = 3200;

// The model's config.json "rec" line filter.
const DROP_BELOW = 0.8;
const JUNK_BELOW = 0.9;
const JUNK_INVALID = 0.05;
const MIN_CHARS = 2;

/**
 * @typedef {{ text: string, confidence: number, prepMs: number, inferMs: number, decodeMs: number }} LineReading
 * @typedef {{ read: (rgba: Uint8Array | Uint8ClampedArray, width: number, height: number) => Promise<LineReading> }} LineRecognizer
 */

/**
 * @param {typeof import("onnxruntime-web")} ort
 * @param {Uint8Array | ArrayBuffer} modelBytes rec.onnx
 * @param {string} dictText rec_dict.txt
 * @returns {Promise<LineRecognizer>}
 */
export async function createRecognizer(ort, modelBytes, dictText) {
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  // CTC classes: 0 = blank, then the dictionary, then the space PaddleOCR appends.
  const classes = ["", ...dictText.split("\n").filter(Boolean), " "];

  return {
    async read(rgba, width, height) {
      const start = performance.now();
      const tensor = toTensor(rgba, width, height);
      const prepared = performance.now();
      const input = new ort.Tensor("float32", tensor.values, [1, 3, HEIGHT, tensor.width]);
      const { probs } = await session.run({ x: input });
      const inferred = performance.now();
      const [, frames, classCount] = probs.dims;
      const { text, confidence } = decode(probs.data, frames, classCount, classes);
      return {
        text,
        confidence,
        prepMs: prepared - start,
        inferMs: inferred - prepared,
        decodeMs: performance.now() - inferred,
      };
    },
  };
}

/** @returns {boolean} whether a reading passes the model's own line filter */
export function isAcceptable(text, confidence) {
  if (!text || confidence < DROP_BELOW) return false;
  if ([...text.replaceAll(" ", "")].length < MIN_CHARS) return false;
  return !(confidence < JUNK_BELOW && invalidShare(text) > JUNK_INVALID);
}

// ---------------------------------------------------------------------------------------
// Preprocessing, "color": "flatten": resize to 48 px high (cv2.INTER_LINEAR) -> BT.601 luma
// -> _flatten -> (x / 255 - 0.5) / 0.5, grey replicated over 3 channels, channel-major.

/** @returns {{ values: Float32Array, width: number }} `[1, 3, 48, width]` */
export function toTensor(rgba, w, h) {
  if (w < 1 || h < 1) throw new Error(`empty crop ${w}x${h}`);
  const lineWidth = Math.min(MAX_WIDTH, Math.max(1, Math.ceil((HEIGHT * w) / h)));
  const small = resizeBilinear(luma(rgba, w, h), w, h, lineWidth, HEIGHT);
  const flat = flatten(small, lineWidth, HEIGHT);

  const width = Math.max(MIN_WIDTH, lineWidth);
  const plane = HEIGHT * width;
  const values = new Float32Array(3 * plane); // 0 = mid-grey after normalising (padding)
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < lineWidth; x++) values[y * width + x] = flat[y * lineWidth + x] / 127.5 - 1;
  }
  values.copyWithin(plane, 0, plane);
  values.copyWithin(2 * plane, 0, plane);
  return { values, width };
}

function luma(rgba, w, h) {
  const out = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = 0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2];
  return out;
}

/** cv2.resize INTER_LINEAR: half-pixel centres, two taps per axis, no antialiasing. */
function resizeBilinear(src, w, h, dw, dh) {
  const taps = (d, scale, size) => {
    const f = (d + 0.5) * scale - 0.5;
    let i = Math.floor(f);
    let t = f - i;
    if (i < 0) [i, t] = [0, 0];
    if (i >= size - 1) [i, t] = [size - 1, 0];
    return [i, Math.min(i + 1, size - 1), t];
  };
  const xTaps = Array.from({ length: dw }, (_, x) => taps(x, w / dw, w));
  const out = new Uint8Array(dw * dh); // Uint8Array stores truncate; round first
  for (let y = 0; y < dh; y++) {
    const [y0, y1, ty] = taps(y, h / dh, h);
    for (let x = 0; x < dw; x++) {
      const [x0, x1, tx] = xTaps[x];
      const top = src[y0 * w + x0] * (1 - tx) + src[y0 * w + x1] * tx;
      const bottom = src[y1 * w + x0] * (1 - tx) + src[y1 * w + x1] * tx;
      out[y * dw + x] = Math.min(255, Math.max(0, Math.round(top * (1 - ty) + bottom * ty)));
    }
  }
  return out;
}

/**
 * _flatten: dark text on white from any line crop. Each column's background is the median
 * of the crop's top and bottom margin rows, smoothed along the line; text is whatever
 * differs from it, light or dark, stretched so the 98th percentile is black.
 */
function flatten(g, w, h) {
  const k = Math.max(2, Math.floor(h / 6));
  const margin = new Uint8Array(2 * k);
  const background = new Uint8Array(w);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < k; y++) {
      margin[y] = g[y * w + x];
      margin[k + y] = g[(h - k + y) * w + x];
    }
    margin.sort();
    background[x] = (margin[k - 1] + margin[k]) >> 1; // np.median, then astype(uint8)
  }
  const smooth = medianFilter(background, Math.max(3, Math.min(h | 1, 255)));

  const diff = new Float32Array(w * h);
  const histogram = new Int32Array(256);
  for (let i = 0; i < w * h; i++) {
    const d = Math.abs(g[i] - smooth[i % w]);
    diff[i] = d;
    histogram[Math.trunc(d)]++;
  }
  const target = 0.98 * w * h;
  let seen = 0;
  let level = 255;
  for (let v = 0; v < 256; v++) {
    seen += histogram[v];
    if (seen >= target) {
      level = v;
      break;
    }
  }
  const scale = 255 / Math.max(level, 24);
  return Uint8Array.from(diff, (d) => 255 - Math.min(d * scale, 255)); // truncates, as astype(uint8)
}

/** cv2.medianBlur along one row: odd window, border replicated. */
function medianFilter(values, window) {
  const radius = window >> 1;
  const last = values.length - 1;
  const buffer = new Uint8Array(window);
  return values.map((_, x) => {
    for (let j = 0; j < window; j++) buffer[j] = values[Math.min(Math.max(x - radius + j, 0), last)];
    buffer.sort();
    return buffer[radius];
  });
}

// ---------------------------------------------------------------------------------------
// Greedy CTC: argmax per frame, repeats collapsed, blanks dropped; confidence is the mean of
// each kept character's first-frame probability (as _ctc_detail).

function decode(p, frames, classCount, classes) {
  let text = "";
  let sum = 0;
  let kept = 0;
  let previous = -1;
  for (let t = 0; t < frames; t++) {
    const row = t * classCount;
    let best = 0;
    for (let c = 1; c < classCount; c++) if (p[row + c] > p[row + best]) best = c;
    if (best !== previous && best !== 0) {
      text += classes[best] ?? "";
      sum += p[row + best];
      kept++;
    }
    previous = best;
  }
  return { text: clean(text), confidence: kept ? sum / kept : 0 };
}

// ---------------------------------------------------------------------------------------
// Text clean-up: fix_scripts(canonical(text)). JS strings iterate by code point, as Python.

// Legacy Thai glyph-variant private-use code points -> standard Thai.
const PUA = new Map([
  [0xf700, 0x0e10], [0xf701, 0x0e34], [0xf702, 0x0e35], [0xf703, 0x0e36], [0xf704, 0x0e37],
  [0xf705, 0x0e48], [0xf706, 0x0e49], [0xf707, 0x0e4a], [0xf708, 0x0e4b], [0xf709, 0x0e4c],
  [0xf70a, 0x0e48], [0xf70b, 0x0e49], [0xf70c, 0x0e4a], [0xf70d, 0x0e4b], [0xf70e, 0x0e4c],
  [0xf70f, 0x0e0d], [0xf710, 0x0e31], [0xf711, 0x0e4d], [0xf712, 0x0e47], [0xf713, 0x0e48],
  [0xf714, 0x0e49], [0xf715, 0x0e4a], [0xf716, 0x0e4b], [0xf717, 0x0e4c], [0xf718, 0x0e38],
  [0xf719, 0x0e39], [0xf71a, 0x0e3a],
]);
const DROPPED = new Set([0x200b, 0x200c, 0x200d, 0x2060, 0x00ad, 0xfeff]); // zero-width, soft hyphen, BOM
const VOWEL_MARKS = new Set("ัิีึืฺุู็");
const TONE_MARKS = new Set("่้๊๋์๎");
const NIKHAHIT = "ํ";
const SARA_AM = /([่-๋]?)ํ([่-๋]?)า/g;
const SPACES = /[ \t\f\v]+/g;
const LONE_W = /(?<![A-Za-z])[Ww](?=\.?[ก-ฮเ-ไ])/g;
const ZERO_IN_THAI = /(?<=[๐-๙])[Oo0]|[Oo0](?=[๐-๙])/g;

/** @returns {string} the runtime's canonical, script-fixed form of a line */
export function clean(text) {
  return canonical(text).replace(LONE_W, "พ").replace(ZERO_IN_THAI, "๐");
}

function canonical(text) {
  let out = "";
  for (const ch of text) {
    const cp = ch.codePointAt(0);
    if (DROPPED.has(cp)) continue;
    out += cp === 0x00a0 ? " " : PUA.has(cp) ? String.fromCodePoint(PUA.get(cp)) : ch;
  }
  out = orderMarks(out.normalize("NFC"))
    .replace(SARA_AM, "$1$2ำ")
    .replaceAll("เเ", "แ")
    .replaceAll("ำำ", "ำ")
    .replace(/\r\n?/g, "\n");
  return out
    .split("\n")
    .map((line) => line.replace(SPACES, " ").trim())
    .join("\n")
    .trim();
}

function markRank(ch) {
  if (VOWEL_MARKS.has(ch)) return 0;
  if (TONE_MARKS.has(ch)) return 1;
  if (ch === NIKHAHIT) return 2;
  return -1;
}

/** Runs of 2+ marks: vowel, then tone, then nikhahit (stable sort), a doubled mark kept once. */
function orderMarks(text) {
  let out = "";
  let run = [];
  const flush = () => {
    const ordered = run.length < 2 ? run : run.toSorted((a, b) => markRank(a) - markRank(b));
    ordered.forEach((ch, i) => {
      if (run.length < 2 || i === 0 || ordered[i - 1] !== ch) out += ch;
    });
    run = [];
  };
  for (const ch of text) {
    if (markRank(ch) >= 0) {
      run.push(ch);
    } else {
      flush();
      out += ch;
    }
  }
  flush();
  return out;
}

const SPELLING_MARKS = new Set("ัิีึืฺุู็่้๊๋์ํ๎");
const isConsonant = (ch) => ch >= "ก" && ch <= "ฮ";
const isLeadingVowel = (ch) => ch >= "เ" && ch <= "ไ";
const isThai = (ch) => ch >= "฀" && ch <= "๿";

/**
 * Share of Thai characters that break Thai spelling: a mark with no consonant under it,
 * or a leading vowel not followed by a consonant.
 */
function invalidShare(text) {
  const s = [...text];
  let thai = 0;
  let bad = 0;
  s.forEach((ch, i) => {
    if (!isThai(ch)) return;
    thai++;
    const prev = i > 0 ? s[i - 1] : " ";
    if (SPELLING_MARKS.has(ch)) {
      const onStackedMark = SPELLING_MARKS.has(prev) && i >= 2 && isConsonant(s[i - 2]);
      if (!isConsonant(prev) && !onStackedMark) bad++;
    } else if (isLeadingVowel(ch) && !(i + 1 < s.length && isConsonant(s[i + 1]))) {
      bad++;
    }
  });
  return thai ? bad / thai : 0;
}
