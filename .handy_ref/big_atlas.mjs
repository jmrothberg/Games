// Render every sprite from the HTML viewer's JS scan into a huge, labelled
// atlas. Each tile is scale 5× with offset + size printed underneath so we
// can scan for specific sprites (scorpion, apple) by eye.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import mod from 'node:module';

const ROOT = '/Users/jonathanrothberg/Games';
const TPL  = path.join(ROOT, 'Generated_games/gauntlet_sprite_viewer.template.html');
const ROM  = path.join(ROOT, 'Gauntlet - The Third Encounter (1990).lnx');
const OUTDIR = path.join(ROOT, 'Generated_games/gauntlet_sprites/extract_v2');

const html = fs.readFileSync(TPL, 'utf8');
let js = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const cut = js.indexOf('const ui = {'); if (cut >= 0) js = js.slice(0, cut);
js += `module.exports = { scanRom, makePal, BPPS, MIN_H_DEFAULT, MIN_W_DEFAULT, WTOL_DEFAULT };`;
const tmpFile = path.join(ROOT, '.handy_ref', '_viewer_js_extracted.cjs');
fs.writeFileSync(tmpFile, js);
const req = mod.createRequire(import.meta.url); const api = req(tmpFile);

const body = Uint8Array.from(fs.readFileSync(ROM).subarray(64));
const opts = { minH: api.MIN_H_DEFAULT, minW: api.MIN_W_DEFAULT, wtol: api.WTOL_DEFAULT };
const sprites = api.scanRom(body, opts);
console.log(`${sprites.length} sprites`);

// Group by (mode, bpp). Render each group as its own atlas.
const groups = {};
for (const s of sprites) {
  const k = `${s.mode}_bpp${s.bpp}`;
  (groups[k] = groups[k] || []).push(s);
}

// Tiny 3x5 bitmap font for per-tile offset labels.
const FONT = {
  '0': ['111','101','101','101','111'],
  '1': ['010','110','010','010','111'],
  '2': ['111','001','111','100','111'],
  '3': ['111','001','111','001','111'],
  '4': ['101','101','111','001','001'],
  '5': ['111','100','111','001','111'],
  '6': ['111','100','111','101','111'],
  '7': ['111','001','010','100','100'],
  '8': ['111','101','111','101','111'],
  '9': ['111','101','111','001','111'],
  'a': ['111','101','111','101','101'],
  'b': ['110','101','110','101','110'],
  'c': ['111','100','100','100','111'],
  'd': ['110','101','101','101','110'],
  'e': ['111','100','111','100','111'],
  'f': ['111','100','111','100','100'],
  'x': ['000','101','010','101','101'],
  '×': ['000','101','010','101','101'],
  '.': ['000','000','000','000','010'],
  ' ': ['000','000','000','000','000'],
};
function drawText(buf, W, H, x, y, txt, rgb) {
  let cx = x;
  for (const ch of txt.toLowerCase()) {
    const g = FONT[ch] || FONT[' '];
    for (let row = 0; row < 5; row++) for (let col = 0; col < 3; col++) {
      if (g[row][col] === '1') {
        const px = cx + col, py = y + row;
        if (px >= 0 && py >= 0 && px < W && py < H) {
          const o = (py * W + px) * 3;
          buf[o] = rgb[0]; buf[o+1] = rgb[1]; buf[o+2] = rgb[2];
        }
      }
    }
    cx += 4;
  }
}

function writePng(outPath, width, height, rgb) {
  const rowLen = width * 3;
  const raw = Buffer.alloc((rowLen + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (rowLen + 1)] = 0;
    rgb.copy(raw, y * (rowLen + 1) + 1, y * rowLen, (y + 1) * rowLen);
  }
  const idat = zlib.deflateSync(raw);
  const crcT = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1); crcT[n] = c >>> 0; }
  const crc32 = b => { let c = 0xffffffff; for (let i = 0; i < b.length; i++) c = crcT[(c ^ b[i]) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
  const chunk = (t, d) => { const L = Buffer.alloc(4); L.writeUInt32BE(d.length, 0); const T = Buffer.from(t, 'ascii'); const C = Buffer.alloc(4); C.writeUInt32BE(crc32(Buffer.concat([T, d])), 0); return Buffer.concat([L, T, d, C]); };
  const sig = Buffer.from([0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a]);
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4); ihdr[8] = 8; ihdr[9] = 2;
  fs.writeFileSync(outPath, Buffer.concat([sig, chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', Buffer.alloc(0))]));
}

const SCALE = 5, PAD = 3, LABEL_H = 8;
const COLS = 12;

function viewLines(s) { return s.stitched || s.lines; }

for (const k of Object.keys(groups).sort()) {
  const items = groups[k].sort((a, b) => a.off - b.off);
  const bpp = items[0].bpp;
  const pal = api.makePal('vivid', bpp);
  const maxW = Math.max(...items.map(s => viewLines(s)[0].length));
  const maxH = Math.max(...items.map(s => viewLines(s).length));
  const CW = maxW * SCALE + PAD * 2;
  const CH = maxH * SCALE + PAD * 2 + LABEL_H;
  const rows = Math.ceil(items.length / COLS);
  const W = CW * COLS, H = CH * rows;
  const buf = Buffer.alloc(W * H * 3, 15);
  for (let i = 0; i < items.length; i++) {
    const s = items[i]; const vl = viewLines(s);
    const r = Math.floor(i / COLS), c = i % COLS;
    const x0 = c * CW + PAD, y0 = r * CH + PAD;
    // fill tile background a touch lighter so transparent shows border
    for (let y = 0; y < maxH * SCALE; y++) for (let x = 0; x < maxW * SCALE; x++) {
      const o = ((y0 + y) * W + (x0 + x)) * 3;
      buf[o] = 25; buf[o+1] = 25; buf[o+2] = 35;
    }
    // draw pixels
    for (let y = 0; y < vl.length; y++) for (let x = 0; x < vl[0].length; x++) {
      const v = vl[y][x];
      const [rr, gg, bb] = pal[v] || [255, 0, 255];
      for (let sy = 0; sy < SCALE; sy++) for (let sx = 0; sx < SCALE; sx++) {
        const px = x0 + x * SCALE + sx, py = y0 + y * SCALE + sy;
        const o = (py * W + px) * 3;
        buf[o] = rr; buf[o+1] = gg; buf[o+2] = bb;
      }
    }
    // label: hex offset + wxh
    const label = `${s.off.toString(16).padStart(5, '0')} ${vl[0].length}x${vl.length}${s.quadrants.length > 1 ? ` q${s.quadrants.length}` : ''}`;
    drawText(buf, W, H, x0, y0 + maxH * SCALE + 1, label, [200, 200, 80]);
  }
  const out = path.join(OUTDIR, `labelled_${k}.png`);
  writePng(out, W, H, buf);
  console.log(`  ${k.padEnd(10)} -> ${out}  (${W}x${H}, ${items.length} sprites)`);
}
