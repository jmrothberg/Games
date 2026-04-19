// Same as test_viewer_js.mjs but force-accept every successful decode,
// ignoring the heuristic scoreSprite filter. Prints counts per (mode, bpp),
// total bytes covered, and dumps a coverage-map PNG.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import mod from 'node:module';

const ROOT = '/Users/jonathanrothberg/Games';
const TPL  = path.join(ROOT, 'Generated_games/gauntlet_sprite_viewer.template.html');
const ROM  = path.join(ROOT, 'Gauntlet - The Third Encounter (1990).lnx');
const OUT  = path.join(ROOT, 'Generated_games/gauntlet_sprites/extract_v2/coverage_no_score.png');

const html = fs.readFileSync(TPL, 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
let js = m[1];
const cut = js.indexOf('const ui = {');
if (cut >= 0) js = js.slice(0, cut);
js += `
module.exports = { decodeMultiQuad, scoreSprite, stitchQuads,
  BPPS, MIN_H_DEFAULT, MIN_W_DEFAULT, WTOL_DEFAULT, makePal };
`;
const tmpFile = path.join(ROOT, '.handy_ref', '_viewer_js_extracted.cjs');
fs.writeFileSync(tmpFile, js);
const req = mod.createRequire(import.meta.url);
const api = req(tmpFile);

const body = Uint8Array.from(fs.readFileSync(ROM).subarray(64));
const opts = { minH: api.MIN_H_DEFAULT, minW: api.MIN_W_DEFAULT, wtol: api.WTOL_DEFAULT };

// Tile forward, but accept EVERY successful multi-quad decode regardless of score.
function tileNoScore(data, mode, bpp, opts) {
  const out = [];
  let pos = 0;
  while (pos < data.length - 2) {
    const r = api.decodeMultiQuad(data, pos, bpp, mode, opts);
    if (r && r.lastTerm === 'sprite') {
      out.push({ mode, bpp, off: pos, end: r.end, q: r.quadrants.length,
                 score: api.scoreSprite(r.quadrants[0], bpp) });
      pos = r.end;
    } else {
      pos += 1;
    }
  }
  return out;
}

const W = 512, H = Math.ceil(body.length / W);
const buf = Buffer.alloc(W * H * 3, 10);
const colors = { rle: [[60,60,90],[200,60,60],[200,160,40],[60,220,120]],
                 lit: [[100,100,160],[60,160,220],[160,220,60],[220,160,220]] };

let totalAll = 0, scoredOut = 0, scoredIn = 0;
for (const mode of ['rle', 'lit']) {
  for (const bpp of api.BPPS) {
    const found = tileNoScore(body, mode, bpp, opts);
    let unscored = 0, scored = 0;
    for (const s of found) {
      if (s.score > 0) scored++; else unscored++;
      const [r,g,b] = colors[mode][bpp - 1];
      for (let p = s.off; p < Math.min(s.end, body.length); p++) {
        const x = p % W, y = Math.floor(p / W);
        const o = (y * W + x) * 3;
        buf[o] = r; buf[o+1] = g; buf[o+2] = b;
      }
    }
    console.log(`  ${mode} bpp=${bpp}: ${found.length} raw  (${scored} scored>0, ${unscored} rejected by score)`);
    totalAll += found.length; scoredIn += scored; scoredOut += unscored;
  }
}
console.log(`\nRAW decodes : ${totalAll}`);
console.log(`scored out  : ${scoredOut}   (score<=0)`);
console.log(`scored in   : ${scoredIn}    (score>0, currently shown in viewer)`);

function writePng(outPath, width, height, rgb) {
  const rowLen = width * 3;
  const raw = Buffer.alloc((rowLen + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (rowLen + 1)] = 0;
    rgb.copy(raw, y * (rowLen + 1) + 1, y * rowLen, (y + 1) * rowLen);
  }
  const idat = zlib.deflateSync(raw);
  const crcT = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    crcT[n] = c >>> 0;
  }
  const crc32 = b => {
    let c = 0xffffffff;
    for (let i = 0; i < b.length; i++) c = crcT[(c ^ b[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  const chunk = (t, d) => {
    const L = Buffer.alloc(4); L.writeUInt32BE(d.length, 0);
    const T = Buffer.from(t, 'ascii');
    const C = Buffer.alloc(4); C.writeUInt32BE(crc32(Buffer.concat([T, d])), 0);
    return Buffer.concat([L, T, d, C]);
  };
  const sig = Buffer.from([0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width,0); ihdr.writeUInt32BE(height,4);
  ihdr[8]=8; ihdr[9]=2;
  fs.writeFileSync(outPath, Buffer.concat([sig, chunk('IHDR',ihdr), chunk('IDAT',idat), chunk('IEND',Buffer.alloc(0))]));
}
writePng(OUT, W, H, buf);
console.log(`coverage map -> ${OUT}`);
