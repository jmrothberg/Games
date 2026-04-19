// Brute-force every offset for every (mode, bpp), then run DP to pick the
// non-overlapping subset that maximises total bytes covered. Compare to the
// forward-tiler's coverage.
import fs from 'node:fs';
import path from 'node:path';
import mod from 'node:module';

const ROOT = '/Users/jonathanrothberg/Games';
const TPL  = path.join(ROOT, 'Generated_games/gauntlet_sprite_viewer.template.html');
const ROM  = path.join(ROOT, 'Gauntlet - The Third Encounter (1990).lnx');

const html = fs.readFileSync(TPL, 'utf8');
let js = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const cut = js.indexOf('const ui = {'); if (cut >= 0) js = js.slice(0, cut);
js += `module.exports = { decodeMultiQuad, scoreSprite, BPPS, MIN_H_DEFAULT, MIN_W_DEFAULT, WTOL_DEFAULT };`;
const tmpFile = path.join(ROOT, '.handy_ref', '_viewer_js_extracted.cjs');
fs.writeFileSync(tmpFile, js);
const req = mod.createRequire(import.meta.url);
const api = req(tmpFile);

const body = Uint8Array.from(fs.readFileSync(ROM).subarray(64));
const opts = { minH: api.MIN_H_DEFAULT, minW: api.MIN_W_DEFAULT, wtol: api.WTOL_DEFAULT };

const all = [];  // {off, end, mode, bpp}
console.log('brute-force scanning all offsets...');
const t0 = Date.now();
for (let off = 0; off < body.length - 2; off++) {
  for (const mode of ['rle', 'lit']) for (const bpp of api.BPPS) {
    const r = api.decodeMultiQuad(body, off, bpp, mode, opts);
    if (r && r.lastTerm === 'sprite') {
      const sc = api.scoreSprite(r.quadrants[0], bpp);
      if (sc > 0) all.push({ off, end: r.end, mode, bpp, sc });
    }
  }
}
console.log(`raw valid decodes: ${all.length}  (${Date.now() - t0}ms)`);

// DP for weighted-interval-scheduling (weight = end - off).
// Sort by end ascending. For each interval, find latest non-overlapping prior
// (binary search on end); dp[i] = max(dp[i-1], (end-start) + dp[predecessor]).
all.sort((a, b) => a.end - b.end);
const N = all.length;
const dp = new Int32Array(N + 1);
const take = new Uint8Array(N);
function lastNonOverlap(i) {
  let lo = 0, hi = i - 1, ans = -1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (all[m].end <= all[i].off) { ans = m; lo = m + 1; }
    else hi = m - 1;
  }
  return ans;
}
for (let i = 0; i < N; i++) {
  const w = all[i].end - all[i].off;
  const j = lastNonOverlap(i);
  const withI = w + (j >= 0 ? dp[j + 1] : 0);
  const without = i > 0 ? dp[i] : 0;
  if (withI >= without) { dp[i + 1] = withI; take[i] = 1; }
  else dp[i + 1] = without;
}
// reconstruct
const picked = [];
let i = N - 1;
while (i >= 0) {
  if (take[i]) { picked.push(all[i]); i = lastNonOverlap(i); }
  else i--;
}
picked.reverse();
const totalBytes = picked.reduce((a, p) => a + (p.end - p.off), 0);
console.log(`DP-optimal pick: ${picked.length} sprites  covering ${totalBytes} bytes  (${(100*totalBytes/body.length).toFixed(1)}%)`);

// per (mode,bpp) breakdown
const bk = {};
for (const p of picked) { const k = `${p.mode}-bpp${p.bpp}`; bk[k] = (bk[k] || 0) + 1; }
for (const [k, n] of Object.entries(bk).sort()) console.log(`  ${k}: ${n}`);

// Compare with the forward tile result.
console.log('\nDelta vs forward-tile (sprites): ' + (picked.length - 203));
