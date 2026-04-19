// Search the ROM for SCB-like templates.
// An SCB layout (without RELOAD-1 stretch/tilt):
//   +0  SPRCTL0  (top 4 bits = sprite type/render mode; low nibble = bpp)
//   +1  SPRCTL1  (literal/quad bits)
//   +2  SPRCOLL
//   +3  NEXT_SCB (2 LE)
//   +5  SPRDLINE (2 LE) -> sprite pixel data
//   +7  HPOSSTRT (2 LE)
//   +9  VPOSSTRT (2 LE)
//   +11 SPRHSIZ  (2 LE)  often 0x0100  (1.0 scaled fixed-point)
//   +13 SPRVSIZ  (2 LE)  often 0x0100
//   [+15 STRETCH (2)] [+17 TILT (2)] only if RELOAD bit set
//   +15..+22 PENPAL (8 bytes) - 4 bits per pen, 16 pens
//
// Heuristic: search for 4-byte windows == [00 01 00 01] (SPRHSIZ/VSIZ both 1.0).
// At each hit, treat the preceding bytes as an SCB; check if SPRDLINE -> sprite.
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

function tryDecode(off) {
  for (const mode of ['rle', 'lit']) for (const bpp of api.BPPS) {
    const r = api.decodeMultiQuad(body, off, bpp, mode, opts);
    if (r && r.lastTerm === 'sprite') {
      const sc = api.scoreSprite(r.quadrants[0], bpp);
      if (sc > 0) return { mode, bpp, end: r.end, lines: r.quadrants[0] };
    }
  }
  return null;
}

const BANKS = [0, 0x4000, 0x8000, 0xC000, 0x10000];
const hits = [];
for (let i = 11; i < body.length - 4; i++) {
  // Look for SPRHSIZ=1.0, SPRVSIZ=1.0 i.e. bytes 00 01 00 01
  if (body[i] !== 0x00 || body[i+1] !== 0x01 || body[i+2] !== 0x00 || body[i+3] !== 0x01) continue;
  // SPRDLINE is 6..7 bytes back from this position? In our layout SPRHSIZ is at +11
  // so SPRDLINE is at +5 = position-6 in the 4-byte window terms. Try a few offsets.
  for (const back of [11, 13, 15]) { // try with/without RELOAD shifts
    const sprdLineOff = i - back + 5;
    if (sprdLineOff < 0) continue;
    const v = body[sprdLineOff] | (body[sprdLineOff+1] << 8);
    for (const base of BANKS) {
      const tgt = (v + base) & 0x3FFFF;
      if (tgt >= body.length) continue;
      const dec = tryDecode(tgt);
      if (dec) {
        hits.push({ scbOff: i - back, sprdLineOff, sprdLine: v, base, sprite: tgt,
                    mode: dec.mode, bpp: dec.bpp });
        break;
      }
    }
  }
}

console.log(`SCB-template candidates (HSIZ=VSIZ=1.0 marker): ${hits.length}`);
// Group by base; print top-N per group
const byBase = {};
for (const h of hits) (byBase[h.base] = byBase[h.base] || []).push(h);
for (const [base, arr] of Object.entries(byBase)) {
  console.log(`\nbase=0x${(+base).toString(16)}  hits=${arr.length}`);
  for (const h of arr.slice(0, 12)) {
    console.log(`  scb@0x${h.scbOff.toString(16).padStart(5,'0')}  sprdline=0x${h.sprdLine.toString(16).padStart(4,'0')} -> sprite@0x${h.sprite.toString(16).padStart(5,'0')} (${h.mode} bpp${h.bpp})`);
  }
}
