// Hunt for sprite-pointer tables (SCB SPRDLINE pointer arrays) in the ROM.
// Lynx carts read sprite data through Suzy via the cart port; SPRDLINE in the
// SCB is typically 16-bit. For a 128KB ROM, the high bit comes from a bank
// register, so we try multiple bank bases.
//
// For every bank base ∈ {0, 0x4000, 0x8000, 0xC000, 0x10000}:
//   walk every byte position; treat consecutive u16 LE values as candidate
//   pointers; a pointer is "good" if (value + base) is a known sprite start.
//   A run of ≥ MIN_TABLE_LEN consecutive good pointers = a probable table.
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

// Map of every offset that decodes as a valid sprite (any mode, any bpp).
console.log('mapping sprite starts ...');
const isStart = new Uint8Array(body.length);
let cnt = 0;
for (let off = 0; off < body.length - 2; off++) {
  for (const mode of ['rle', 'lit']) for (const bpp of api.BPPS) {
    const r = api.decodeMultiQuad(body, off, bpp, mode, opts);
    if (r && r.lastTerm === 'sprite') {
      const sc = api.scoreSprite(r.quadrants[0], bpp);
      if (sc > 0) { isStart[off] = 1; cnt++; break; }
    }
    if (isStart[off]) break;
  }
}
console.log(`  ${cnt} unique sprite-start offsets`);

const MIN_TABLE = 4;
const BANKS = [0, 0x4000, 0x8000, 0xC000, 0x10000];
const tables = [];

for (const base of BANKS) {
  for (let i = 0; i < body.length - 1; ) {
    let j = i;
    const ptrs = [];
    const offs = [];
    while (j + 2 <= body.length) {
      const v = body[j] | (body[j + 1] << 8);
      const tgt = (v + base) & 0x3FFFF; // wrap into reasonable cart range
      if (tgt < body.length && isStart[tgt]) { ptrs.push(v); offs.push(tgt); j += 2; }
      else break;
    }
    if (ptrs.length >= MIN_TABLE) { tables.push({ base, off: i, ptrs, offs }); i = j; }
    else i += 1;
  }
}

tables.sort((a, b) => b.ptrs.length - a.ptrs.length);
console.log(`\nfound ${tables.length} candidate tables (len >= ${MIN_TABLE})`);
for (const t of tables.slice(0, 25)) {
  const pp = t.ptrs.slice(0, 6).map(p => '0x' + p.toString(16).padStart(4, '0')).join(' ');
  const oo = t.offs.slice(0, 6).map(o => '0x' + o.toString(16).padStart(5, '0')).join(' ');
  console.log(`  base=0x${t.base.toString(16).padStart(5,'0')}  @0x${t.off.toString(16).padStart(5,'0')}  len=${t.ptrs.length}  ptrs:[${pp}]  -> [${oo}]`);
}
