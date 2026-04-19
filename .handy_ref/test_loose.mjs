// Sanity check: scan with strict and loose modes and compare counts.
import fs from 'node:fs';
import path from 'node:path';
import mod from 'node:module';

const ROOT = '/Users/jonathanrothberg/Games';
const TPL  = path.join(ROOT, 'Generated_games/gauntlet_sprite_viewer.template.html');
const ROM  = path.join(ROOT, 'Gauntlet - The Third Encounter (1990).lnx');

const html = fs.readFileSync(TPL, 'utf8');
let js = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const cut = js.indexOf('const ui = {'); if (cut >= 0) js = js.slice(0, cut);
js += `module.exports = { scanRom, MIN_H_DEFAULT, MIN_W_DEFAULT, WTOL_DEFAULT };`;
const tmpFile = path.join(ROOT, '.handy_ref', '_viewer_js_extracted.cjs');
fs.writeFileSync(tmpFile, js);
const req = mod.createRequire(import.meta.url);
const api = req(tmpFile);

const body = Uint8Array.from(fs.readFileSync(ROM).subarray(64));
const base = { minH: api.MIN_H_DEFAULT, minW: api.MIN_W_DEFAULT, wtol: api.WTOL_DEFAULT };

const strict = api.scanRom(body, { ...base, loose: false });
const loose  = api.scanRom(body, { ...base, loose: true  });
console.log('strict sprites :', strict.length);
console.log('loose  sprites :', loose.length);
console.log('extra in loose :', loose.length - strict.length);

// Coverage analysis
const cov = arr => {
  const seen = new Uint8Array(body.length);
  for (const s of arr) for (let i = s.off; i < s.end; i++) seen[i] = 1;
  let c = 0; for (let i = 0; i < seen.length; i++) if (seen[i]) c++;
  return (c / body.length * 100).toFixed(2) + '%';
};
console.log('strict coverage:', cov(strict));
console.log('loose  coverage:', cov(loose));

// new-in-loose offsets
const stOffs = new Set(strict.map(s => `${s.mode}:${s.bpp}:${s.off}`));
const newOnes = loose.filter(s => !stOffs.has(`${s.mode}:${s.bpp}:${s.off}`));
console.log(`\n${newOnes.length} extra sprites in loose mode (sample, first 20):`);
for (const s of newOnes.slice(0, 20)) {
  console.log(`  ${s.mode} bpp${s.bpp}  off=0x${s.off.toString(16).padStart(5,'0')}  size=${s.lines[0].length}x${s.lines.length}  score=${s.score.toFixed(1)}`);
}
