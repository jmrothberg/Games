#!/usr/bin/env python3
"""
Look for sprite-pointer tables in the Gauntlet III Lynx ROM.

Lynx games usually store sprite pixel data in ROM and reference it via an
SCB (in RAM, copied from ROM) whose `data` field is a 16-bit pointer. Many
games keep a table of such pointers together; those tables look like a run of
little-endian 16-bit values, each pointing to a byte that looks like a valid
line-offset (small value, typically 3..20) followed by a plausible sprite.

Strategy:
  1. For each candidate offset in ROM, test "does this look like a sprite
     start?" using our validated decoder (tries rle 4bpp, rle 3bpp, lit 4bpp,
     lit 2bpp — the most common).
  2. Walk the ROM as u16_le. Any 16-bit value that resolves to a "good"
     sprite-start offset (when added to some base) is a possible pointer.
  3. A pointer TABLE is a sequence of >=4 such good pointers at consecutive
     2-byte positions.
  4. Once a table is located, every pointer from it is a high-confidence
     sprite.

Also tries an alternative: treat the u16 as an ABSOLUTE cart offset
(base=0). Gauntlet may embed offsets as bank-relative instead.

Outputs:
  * Console: detected pointer tables with # pointers and sample sprite images
  * PNGs: one per referenced sprite at out_dir/ptr_<table>_<idx>.png
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image
import sys
import struct

sys.path.insert(0, str(Path(__file__).parent))
from lynx_sprite_extract import (
    decode_sprite_literal, decode_sprite_packed, score_sprite,
    gray_palette, vivid_palette, render_sprite,
)

ROM = "/Users/jonathanrothberg/Games/Gauntlet - The Third Encounter (1990).lnx"
OUT = Path("/Users/jonathanrothberg/Games/Generated_games/gauntlet_sprites/tables")

# Decoder variants to try in preference order
VARIANTS = [
    ("rle", 4), ("rle", 3), ("rle", 2),
    ("lit", 4), ("lit", 2), ("lit", 1),
]

MIN_TABLE_LEN = 4  # pointer runs shorter than this are ignored


def try_sprite_at(body, off):
    """Try all variants; return (mode, bpp, lines, end, score) for the best."""
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
    OUT.mkdir(parents=True, exist_ok=True)

    # Pre-compute a "is sprite start?" map for each byte offset.
    print("mapping valid sprite-start offsets (this takes ~60s)...")
    is_start = [None] * len(body)
    for off in range(len(body) - 4):
        r = try_sprite_at(body, off)
        if r is not None:
            is_start[off] = r
    good_offsets = {i for i, v in enumerate(is_start) if v is not None}
    print(f"  {len(good_offsets)} candidate sprite starts")

    # Scan for pointer tables: runs of u16_le at 2-byte steps whose values
    # each resolve to a valid sprite start when interpreted as a cart offset.
    # We try two bases: 0 (absolute cart offset) and any offset that produces
    # a longer run.
    print("scanning for pointer tables...")
    tables = []  # (base, table_offset, pointers[], sprite_offsets[])
    for base in (0,):
        i = 0
        while i < len(body) - 1:
            ptrs = []
            sprite_offs = []
            j = i
            while j + 2 <= len(body):
                v = struct.unpack_from("<H", body, j)[0]
                tgt = v - base
                if 0 <= tgt < len(body) and tgt in good_offsets:
                    ptrs.append(v)
                    sprite_offs.append(tgt)
                    j += 2
                else:
                    break
            if len(ptrs) >= MIN_TABLE_LEN:
                tables.append((base, i, ptrs, sprite_offs))
                i = j
            else:
                i += 1
    # Dedup (same base, table_offset)
    tables.sort(key=lambda t: (-len(t[2]), t[1]))
    print(f"  found {len(tables)} tables with >= {MIN_TABLE_LEN} pointers")

    # Print top tables
    for base, toff, ptrs, offs in tables[:20]:
        print(f"  table @ 0x{toff:05x}  base={base}  len={len(ptrs)}  "
              f"first ptrs: {[hex(p) for p in ptrs[:5]]}  "
              f"first sprites: {[hex(o) for o in offs[:5]]}")

    # Dump sprites from top tables
    for ti, (base, toff, ptrs, offs) in enumerate(tables[:8]):
        sprites = []
        for k, off in enumerate(offs):
            r = is_start[off]
            if r is None:
                continue
            mode, bpp, lines, end, s = r
            sprites.append((k, off, mode, bpp, lines))
        # Render vivid + gray atlases
        if not sprites:
            continue
        pal = vivid_palette(4)
        imgs = []
        for k, off, mode, bpp, lines in sprites:
            pal_here = vivid_palette(bpp)
            imgs.append(render_sprite(lines, pal_here))
        cols = 8
        maxw = max(im.width for im in imgs)
        maxh = max(im.height for im in imgs)
        scale = 3
        rows = (len(imgs) + cols - 1) // cols
        cw = maxw * scale + 4
        ch = maxh * scale + 4
        atlas = Image.new("RGB", (cw * cols, ch * rows), (15, 15, 25))
        for i, im in enumerate(imgs):
            up = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
            r_, c_ = divmod(i, cols)
            atlas.paste(up, (c_ * cw + 2, r_ * ch + 2))
        name = OUT / f"table_{ti:02d}_toff{toff:05x}_n{len(ptrs)}.png"
        atlas.save(name)
        print(f"  wrote {name}")


if __name__ == "__main__":
    main()
