#!/usr/bin/env python3
"""Render every decodable sprite-start in the ROM into one scrollable atlas."""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys

sys.path.insert(0, str(Path(__file__).parent))
from lynx_sprite_extract import (
    decode_sprite_literal, decode_sprite_packed, score_sprite,
    gray_palette, vivid_palette, render_sprite,
)

ROM = "/Users/jonathanrothberg/Games/Gauntlet - The Third Encounter (1990).lnx"
OUT = Path("/Users/jonathanrothberg/Games/Generated_games/gauntlet_sprites/all_candidates")
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = [("rle", 4), ("rle", 3), ("rle", 2), ("rle", 1),
            ("lit", 4), ("lit", 2), ("lit", 1)]

def best_at(body, off):
    best = None
    for mode, bpp in VARIANTS:
        fn = decode_sprite_literal if mode == "lit" else decode_sprite_packed
        r = fn(body, off, bpp)
        if r is None:
            continue
        lines, end = r
        s = score_sprite(lines, bpp)
        if s <= 0:
            continue
        if best is None or s > best[4]:
            best = (mode, bpp, lines, end, s)
    return best

def main():
    body = Path(ROM).read_bytes()[64:]
    print("scanning...")
    hits = []
    for off in range(len(body) - 4):
        r = best_at(body, off)
        if r is None:
            continue
        mode, bpp, lines, end, s = r
        hits.append((off, mode, bpp, lines, end, s))
    print(f"  {len(hits)} candidate sprite starts")

    # Group non-overlapping (per scan order) keeping highest-scoring
    hits.sort(key=lambda h: h[0])
    kept = []
    last_end = -1
    i = 0
    while i < len(hits):
        # within an overlap window, take the highest score
        j = i
        best = hits[i]
        while j < len(hits) and hits[j][0] < best[4]:
            if hits[j][5] > best[5]:
                best = hits[j]
            j += 1
        kept.append(best)
        # advance past chosen sprite end
        while i < len(hits) and hits[i][0] < best[4]:
            i += 1
    print(f"  kept non-overlapping: {len(kept)}")

    # Render each candidate scaled 3x, with offset label beside it
    SCALE = 3
    LABEL_W = 76
    PAD = 2
    COLS = 10
    cells = []
    for off, mode, bpp, lines, end, s in kept:
        img = render_sprite(lines, vivid_palette(bpp))
        up = img.resize((img.width * SCALE, img.height * SCALE), Image.NEAREST)
        cell_w = max(up.width + LABEL_W, LABEL_W) + PAD * 2
        cell_h = max(up.height, 20) + PAD * 2
        cell = Image.new("RGB", (cell_w, cell_h), (20, 20, 30))
        d = ImageDraw.Draw(cell)
        # label
        d.text((PAD, PAD), f"{off:05x}", fill=(230, 230, 230))
        d.text((PAD, PAD + 10), f"{mode}{bpp} {img.width}x{img.height}",
               fill=(160, 180, 200))
        cell.paste(up, (LABEL_W, PAD))
        cells.append(cell)

    if not cells:
        print("no cells"); return
    maxw = max(c.width for c in cells)
    maxh = max(c.height for c in cells)
    rows = (len(cells) + COLS - 1) // COLS
    atlas = Image.new("RGB", (maxw * COLS, maxh * rows), (8, 8, 12))
    for i, c in enumerate(cells):
        r_, col = divmod(i, COLS)
        atlas.paste(c, (col * maxw, r_ * maxh))
    out = OUT / "ALL_candidates.png"
    atlas.save(out)
    print(f"  wrote {out}  size={atlas.size}")

if __name__ == "__main__":
    main()
