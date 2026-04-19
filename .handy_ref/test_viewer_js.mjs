// Harness that extracts the <script> body from gauntlet_sprite_viewer.template.html,
// replaces the `boot()`/DOM bits, feeds the ROM in, runs scanRom, then renders
// every sprite into a PNG atlas so we can visually validate the JS decoder.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

const ROOT = '/Users/jonathanrothberg/Games';
const TPL  = path.join(ROOT, 'Generated_games/gauntlet_sprite_viewer.template.html');
const ROM  = path.join(ROOT, 'Gauntlet - The Third Encounter (1990).lnx');
const OUT  = path.join(ROOT, 'Generated_games/gauntlet_sprites/extract_v2/viewer_js_atlas.png');

const html = fs.readFileSync(TPL, 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) throw new Error('no <script> in template');
let js = m[1];

// Strip the `ui = {...}` block, the DOM event hookups, and boot() — we only
// need the pure decoder/scanner functions (decodePackedLine, decodeSpritePacked,
// decodeSpriteLiteral, decodeMultiQuad, stitchQuads, scoreSprite, scanRom,
// mkSprite, tileForward, makePal, BPPS, MIN_H_DEFAULT, etc.).
// Cut from `const ui = ` onward.
const cut = js.indexOf('const ui = {');
if (cut >= 0) js = js.slice(0, cut);

// Expose what we need to the module scope.
js += `
module.exports = {
  decodeSpritePacked, decodeSpriteLiteral, decodeMultiQuad, stitchQuads,
  scanRom, scoreSprite, makePal, tileForward, mkSprite,
  BPPS, MIN_H_DEFAULT, MIN_W_DEFAULT, WTOL_DEFAULT,
};
`;

// Dynamically require the JS as a CommonJS module.
const tmpFile = path.join(ROOT, '.handy_ref', '_viewer_js_extracted.cjs');
fs.writeFileSync(tmpFile, js);
const mod = await import('node:module');
const req = mod.createRequire(import.meta.url);
const api = req(tmpFile);

// Load ROM, strip LNX header.
const full = fs.readFileSync(ROM);
const body = Uint8Array.from(full.subarray(64));
console.log(`cart body: ${body.length} bytes`);

const opts = { minH: api.MIN_H_DEFAULT, minW: api.MIN_W_DEFAULT, wtol: api.WTOL_DEFAULT };

// Per (mode, bpp) breakdown.
for (const mode of ['rle', 'lit']) {
  for (const bpp of api.BPPS) {
    const r = api.tileForward(body, mode, bpp, opts);
    console.log(`  tileForward  ${mode} bpp=${bpp}: ${r.length} sprites`);
  }
}

const t0 = Date.now();
const sprites = api.scanRom(body, opts);
const dt = Date.now() - t0;
console.log(`\nscanRom total: ${sprites.length} sprites  (${dt}ms)`);

// Quadrant-count stats.
const qCounts = [0, 0, 0, 0, 0];
for (const s of sprites) qCounts[s.quadrants.length]++;
console.log(`quadrant counts: 1=${qCounts[1]} 2=${qCounts[2]} 3=${qCounts[3]} 4=${qCounts[4]}`);

// Minimal PNG encoder: produces an indexed 8bpp palette PNG, which keeps the
// output small and easy to diff visually. We'll render to a big RGB buffer
// with 2px padding between tiles.
function viewLines(s) { return s.stitched || s.lines; }

const COLS = 24;
const SCALE = 2;
const PAD = 2;
const pal = (bpp) => api.makePal('vivid', bpp);

// compute cell size (max W/H across all displayed grids)
let maxW = 1, maxH = 1;
for (const s of sprites) {
  const vl = viewLines(s);
  if (vl[0].length > maxW) maxW = vl[0].length;
  if (vl.length > maxH) maxH = vl.length;
}
const CW = maxW * SCALE + PAD * 2;
const CH = maxH * SCALE + PAD * 2;
const rows = Math.ceil(sprites.length / COLS);
const W = CW * COLS;
const H = CH * rows;
const buf = Buffer.alloc(W * H * 3, 15);  // dark background

function putPx(x, y, r, g, b) {
  if (x < 0 || y < 0 || x >= W || y >= H) return;
  const o = (y * W + x) * 3;
  buf[o] = r; buf[o+1] = g; buf[o+2] = b;
}
for (let i = 0; i < sprites.length; i++) {
  const s = sprites[i];
  const vl = viewLines(s);
  const P = pal(s.bpp);
  const r = Math.floor(i / COLS), c = i % COLS;
  const x0 = c * CW + PAD;
  const y0 = r * CH + PAD;
  for (let y = 0; y < vl.length; y++) for (let x = 0; x < vl[0].length; x++) {
    const v = vl[y][x];
    const [rr, gg, bb] = P[v] || [255, 0, 255];
    for (let sy = 0; sy < SCALE; sy++) for (let sx = 0; sx < SCALE; sx++) {
      putPx(x0 + x * SCALE + sx, y0 + y * SCALE + sy, rr, gg, bb);
    }
  }
}

// Write a simple PNG (truecolor, 8bpc) using zlib + CRC.
function writePng(outPath, width, height, rgb) {
  const rowLen = width * 3;
  const raw = Buffer.alloc((rowLen + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (rowLen + 1)] = 0;  // filter None
    rgb.copy(raw, y * (rowLen + 1) + 1, y * rowLen, (y + 1) * rowLen);
  }
  const idat = zlib.deflateSync(raw);

  const crcTable = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    crcTable[n] = c >>> 0;
  }
  function crc32(buf) {
    let c = 0xffffffff;
    for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }
  function chunk(type, data) {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
    const typ = Buffer.from(type, 'ascii');
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(Buffer.concat([typ, data])), 0);
    return Buffer.concat([len, typ, data, crc]);
  }
  const sig = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;     // bit depth
  ihdr[9] = 2;     // color type: RGB
  ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  fs.writeFileSync(outPath, Buffer.concat([sig, chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', Buffer.alloc(0))]));
}
fs.mkdirSync(path.dirname(OUT), { recursive: true });
writePng(OUT, W, H, buf);
console.log(`wrote atlas -> ${OUT}  (${W}x${H})`);
