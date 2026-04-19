// Walk the coverage produced by tile_forward, find every "gap" in the sprite
// region ≥ 32 bytes, print:
//   - byte range
//   - a hex preview
//   - for each offset inside the gap, whether any (mode, bpp) decodeMultiQuad
//     accepts it — i.e., whether the tiler could have picked it up with a
//     different entry alignment.
import fs from 'node:fs';
import path from 'node:path';
import mod from 'node:module';

const ROOT = '/Users/jonathanrothberg/Games';
const TPL  = path.join(ROOT, 'Generated_games/gauntlet_sprite_viewer.template.html');
const ROM  = path.join(ROOT, 'Gauntlet - The Third Encounter (1990).lnx');

const html = fs.readFileSync(TPL, 'utf8');
let js = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const cut = js.indexOf('const ui = {'); if (cut >= 0) js = js.slice(0, cut);
js += `module.exports = { decodeMultiQuad, BPPS, MIN_H_DEFAULT, MIN_W_DEFAULT, WTOL_DEFAULT, tileForward, scoreSprite };`;
const tmpFile = path.join(ROOT, '.handy_ref', '_viewer_js_extracted.cjs');
fs.writeFileSync(tmpFile, js);
const req = mod.createRequire(import.meta.url);
const api = req(tmpFile);

const body = Uint8Array.from(fs.readFileSync(ROM).subarray(64));
const opts = { minH: api.MIN_H_DEFAULT, minW: api.MIN_W_DEFAULT, wtol: api.WTOL_DEFAULT };

// Compute coverage.
const cov = new Uint8Array(body.length);
for (const mode of ['rle', 'lit']) for (const bpp of api.BPPS) {
  const found = api.tileForward(body, mode, bpp, opts);
  for (const s of found) for (let p = s.off; p < s.end; p++) cov[p] = 1;
}
const covered = cov.reduce((a,b) => a + b, 0);
console.log(`covered: ${covered} / ${body.length}  (${(100*covered/body.length).toFixed(1)}%)`);

// Find gaps.
const gaps = [];
let i = 0;
while (i < body.length) {
  if (cov[i]) { i++; continue; }
  const start = i;
  while (i < body.length && !cov[i]) i++;
  gaps.push([start, i]);
}
console.log(`${gaps.length} gaps`);

// For each gap in the sprite-dense region (rough heuristic: between first and last
// covered byte), try every offset and see what any (mode, bpp) multi-quad decode
// accepts.
let firstCov = cov.indexOf(1);
let lastCov = body.length;
for (let k = body.length - 1; k >= 0; k--) if (cov[k]) { lastCov = k; break; }
const big = gaps.filter(([a, b]) => b - a >= 32 && a >= firstCov && b <= lastCov)
                .sort((x, y) => (y[1]-y[0]) - (x[1]-x[0]))
                .slice(0, 12);

for (const [a, b] of big) {
  const hits = [];
  for (let o = a; o < Math.min(b, a + 128); o++) {
    for (const mode of ['rle', 'lit']) for (const bpp of api.BPPS) {
      const r = api.decodeMultiQuad(body, o, bpp, mode, opts);
      if (r && r.lastTerm === 'sprite') {
        const sc = api.scoreSprite(r.quadrants[0], bpp);
        if (sc > 0) {
          hits.push({ off: o, bpp, mode, end: r.end, w: r.quadrants[0][0].length, h: r.quadrants[0].length, sc });
        }
      }
    }
  }
  const hex = Array.from(body.subarray(a, Math.min(a + 32, b))).map(b => b.toString(16).padStart(2,'0')).join(' ');
  console.log(`\ngap 0x${a.toString(16).padStart(5,'0')}..0x${b.toString(16).padStart(5,'0')}  (${b - a} B)`);
  console.log(`  bytes: ${hex}${b - a > 32 ? ' …' : ''}`);
  console.log(`  valid starts in first 128 B: ${hits.length}`);
  for (const h of hits.slice(0, 5)) {
    console.log(`    @0x${h.off.toString(16).padStart(5,'0')}  ${h.mode} ${h.bpp}bpp  ${h.w}×${h.h}  ends@0x${h.end.toString(16).padStart(5,'0')}  score=${h.sc.toFixed(1)}`);
  }
}
