#!/usr/bin/env python3
"""
Lynx sprite extractor for Gauntlet: The Third Encounter (.lnx).

Implements both LITERAL (unpacked) and RLE (packed) Suzy sprite decoders per
the documented line+packet format:
  https://www.chibiakumas.com/6502/atarilynx.php

Brute-force scans the cart body for plausible sprite starts, validates by
per-line consistency, and dumps candidates as PNG atlases for visual review.

Usage:
    python3 lynx_sprite_extract.py [rom.lnx] [out_dir]

No side-effects beyond writing PNGs inside out_dir (default:
Generated_games/gauntlet_sprites/extract).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from PIL import Image

# -------- config --------
DEFAULT_ROM = "/Users/jonathanrothberg/Games/Gauntlet - The Third Encounter (1990).lnx"
DEFAULT_OUT = "/Users/jonathanrothberg/Games/Generated_games/gauntlet_sprites/extract"

BPPS = (1, 2, 3, 4)         # try all Lynx bit-depths
MIN_H, MAX_H = 4, 96        # plausible sprite height in lines
MIN_W, MAX_W = 4, 80        # plausible sprite width in pixels (4 catches apples/keys)
MAX_SCAN_CANDIDATES = 50000 # cap for safety
CHAIN_MAX = 128             # max sprites in one tiled region

# ---------- bit reader ----------
def _bit_iter(byte_seq):
    """MSB-first bit stream over a bytes/list-of-int iterable."""
    for b in byte_seq:
        for i in (7, 6, 5, 4, 3, 2, 1, 0):
            yield (b >> i) & 1


# ---------- RLE (packed) line decoder ----------
def decode_packed_line(line_bytes: bytes, bpp: int, expected_w: int | None):
    """
    Decode one RLE line per Lynx packet rules:
      bit: 1 = LITERAL, 0 = RLE
      next 4 bits: N (count - 1, so real count = N+1 in [1..16])
      LITERAL: (N+1) * bpp bits of pixel data follow
      RLE:     bpp bits of pixel follow, repeated (N+1) times

    Consumes bits until we hit a packet that won't fit in remaining bits.
    Returns list of nibble/bit values (length = width in pixels).
    """
    bits = list(_bit_iter(line_bytes))
    pos = 0
    end = len(bits)
    pixels: list[int] = []
    mask = (1 << bpp) - 1

    # hard cap to prevent runaway on pathological inputs
    max_px = MAX_W * 2

    # Need at least 5 bits for a packet header (type bit + 4 count bits).
    while pos + 5 <= end:
        typ = bits[pos]; pos += 1
        n = 0
        for _ in range(4):
            n = (n << 1) | bits[pos]; pos += 1

        if typ == 0:
            # PACKED: per Handy LineGetPixel, count==0 is an explicit END-OF-LINE
            # marker (this was missing before — it caused us to read garbage as
            # an extra RLE pixel and accept far too many false sprites).
            if n == 0:
                break
            count = n + 1
            if pos + bpp > end:
                return None
            v = 0
            for _k in range(bpp):
                v = (v << 1) | bits[pos]; pos += 1
            pixels.extend([v & mask] * count)
        else:
            # LITERAL packet: count = n+1 pixels follow, bpp bits each.
            count = n + 1
            needed = count * bpp
            if pos + needed > end:
                return None
            for _ in range(count):
                v = 0
                for _k in range(bpp):
                    v = (v << 1) | bits[pos]; pos += 1
                pixels.append(v & mask)

        if len(pixels) > max_px:
            return None
    return pixels


WIDTH_TOLERANCE = 2  # allow per-line width to drift by up to this (trailing-bit slop)


def decode_sprite_packed(data: bytes, offset: int, bpp: int):
    """
    Try to decode a single quadrant of a packed sprite starting at `offset`.
    Returns (lines, end_pos) on success, else None.

    Stop conditions:
      line_offset == 0 -> end of sprite
      line_offset == 1 -> end of quadrant (optional marker)

    Accepts up to WIDTH_TOLERANCE pixel drift between lines (partial trailing
    bits can cause off-by-1/2 in the decoder). Normalizes the output to the
    median width by padding or cropping.
    """
    pos = offset
    lines: list[list[int]] = []
    first_w: int | None = None
    while pos < len(data):
        lo = data[pos]
        if lo == 0:
            pos += 1; break
        if lo == 1:
            pos += 1; break
        if lo < 2 or pos + lo > len(data):
            return None
        line_bytes = bytes(data[pos + 1: pos + lo])
        pos += lo
        px = decode_packed_line(line_bytes, bpp, first_w)
        if px is None or not px:
            return None
        w = len(px)
        if w < MIN_W or w > MAX_W:
            return None
        if first_w is None:
            first_w = w
        elif abs(w - first_w) > WIDTH_TOLERANCE:
            return None
        lines.append(px)
        if len(lines) > MAX_H:
            return None
    if len(lines) < MIN_H:
        return None
    # normalize widths to the median (pad with 0, crop if over)
    target_w = sorted(len(l) for l in lines)[len(lines) // 2]
    norm = []
    for l in lines:
        if len(l) < target_w:
            l = l + [0] * (target_w - len(l))
        elif len(l) > target_w:
            l = l[:target_w]
        norm.append(l)
    return norm, pos


# ---------- multi-quadrant wrapper ----------
# Per Handy's outer sprite loop: a Lynx sprite is 1..4 quadrants; each quadrant's
# pixel stream ends with line-offset byte 0x01 (end-of-quad). The final quadrant
# ends with 0x00 (end-of-sprite). We peek at the terminator to walk up to 4
# quadrants per sprite and treat the whole block as a single sprite record.
def decode_sprite_multiquad(data: bytes, offset: int, bpp: int, mode: str):
    """
    Returns (quadrants, end_pos, last_term) where:
      quadrants = list of 1..4 per-quadrant line arrays
      end_pos   = byte offset just past the terminator
      last_term = 'sprite' (0x00) or 'quad' (0x01) of the final read
    or None on failure.
    """
    quads = []
    pos = offset
    last_term = None
    dec = decode_sprite_packed if mode == "rle" else decode_sprite_literal
    for _ in range(4):
        r = dec(data, pos, bpp)
        if r is None:
            break
        lines, new_pos = r
        # Figure out which terminator the decoder just consumed (decoder already
        # advanced past the 0x00 or 0x01 byte).
        term_byte = data[new_pos - 1] if new_pos > 0 else None
        last_term = 'sprite' if term_byte == 0 else ('quad' if term_byte == 1 else None)
        quads.append(lines)
        pos = new_pos
        if last_term != 'quad':
            break
    if not quads:
        return None
    return quads, pos, last_term


def stitch_quads(quads):
    """Compose up to 4 quadrants around a center using Suzy rotation order:
    Q0=DR, Q1=UR, Q2=UL, Q3=DL. Each quadrant's [0,0] is the center pixel."""
    n = len(quads)
    W = [0, 0, 0, 0]; H = [0, 0, 0, 0]
    for i in range(n):
        H[i] = len(quads[i]); W[i] = len(quads[i][0]) if quads[i] else 0
    right = max(W[0], W[1]); left = max(W[2], W[3])
    down  = max(H[0], H[3]); up   = max(H[1], H[2])
    Wo = left + right; Ho = up + down
    if Wo <= 0 or Ho <= 0:
        return quads[0]
    out = [[0] * Wo for _ in range(Ho)]
    cx, cy = left, up
    # Q0 DR
    if n >= 1:
        for y in range(H[0]):
            for x in range(W[0]):
                out[cy + y][cx + x] = quads[0][y][x]
    # Q1 UR (vflip)
    if n >= 2:
        for y in range(H[1]):
            for x in range(W[1]):
                out[cy - 1 - y][cx + x] = quads[1][y][x]
    # Q2 UL (both flipped)
    if n >= 3:
        for y in range(H[2]):
            for x in range(W[2]):
                out[cy - 1 - y][cx - 1 - x] = quads[2][y][x]
    # Q3 DL (hflip)
    if n >= 4:
        for y in range(H[3]):
            for x in range(W[3]):
                out[cy + y][cx - 1 - x] = quads[3][y][x]
    return out


# ---------- LITERAL (unpacked) decoder ----------
def decode_sprite_literal(data: bytes, offset: int, bpp: int):
    """
    Literal-format sprite: each line is (offset byte)(raw pixel bytes).
    Pixels are packed MSB-first at bpp bits per pixel.
    Line length in bytes = line_offset - 1 (the offset byte itself counted).
    """
    pos = offset
    lines: list[list[int]] = []
    first_len: int | None = None
    mask = (1 << bpp) - 1
    # NEW: bitstream-based extract (was byte-aligned only). Handy reads literal
    # pixels MSB-first across the line bytes, so 3bpp is valid even though it
    # doesn't byte-align. Required for matching the JS decoder.

    while pos < len(data):
        lo = data[pos]
        if lo == 0:
            pos += 1; break
        if lo == 1:
            pos += 1; break
        if lo < 2 or pos + lo > len(data):
            return None
        line_bytes = data[pos + 1: pos + lo]
        pos += lo

        if first_len is None:
            first_len = len(line_bytes)
        elif len(line_bytes) != first_len:
            return None

        total_bits = len(line_bytes) * 8
        px_count = total_bits // bpp
        pixels: list[int] = [0] * px_count
        bit_pos = 0
        for p in range(px_count):
            v = 0
            for _ in range(bpp):
                byte = line_bytes[bit_pos >> 3]
                v = (v << 1) | ((byte >> (7 - (bit_pos & 7))) & 1)
                bit_pos += 1
            pixels[p] = v & mask
        w = len(pixels)
        if w < MIN_W or w > MAX_W:
            return None
        lines.append(pixels)
        if len(lines) > MAX_H:
            return None

    if len(lines) < MIN_H:
        return None
    return lines, pos


# ---------- scoring & palette ----------
def score_sprite(lines: list[list[int]], bpp: int) -> float:
    """Heuristic: prefer mid-sized sprites with color diversity and some zeros."""
    h = len(lines)
    w = len(lines[0])
    flat = [p for row in lines for p in row]
    if not flat:
        return -1
    zeros = flat.count(0)
    unique = len(set(flat))
    ncol = 1 << bpp
    if unique < 2:
        return -1
    trans_ratio = zeros / len(flat)
    if trans_ratio > 0.97:
        return -1
    # NEW structure check: count pixels that share a colour with any 4-neighbour.
    # Real sprites have local cohesion; random ROM bytes do not.
    neigh = 0
    for y in range(h):
        row = lines[y]
        for x in range(w):
            v = row[x]
            if (x > 0 and row[x - 1] == v) or \
               (x < w - 1 and row[x + 1] == v) or \
               (y > 0 and lines[y - 1][x] == v) or \
               (y < h - 1 and lines[y + 1][x] == v):
                neigh += 1
    cohesion = neigh / len(flat)
    if cohesion < 0.35:
        return -1
    shape_score = min(h, 32) + min(w, 32)
    return shape_score + unique * 2 - abs(trans_ratio - 0.35) * 20 + cohesion * 10


def gray_palette(bpp: int):
    ncol = 1 << bpp
    pal = []
    for i in range(ncol):
        if i == 0:
            pal.append((20, 20, 30))  # "transparent" -> dark
        else:
            g = int(32 + (i / max(1, ncol - 1)) * 220)
            pal.append((g, g, g))
    return pal


def vivid_palette(bpp: int):
    """Placeholder 16-color-ish palette for a recognizable preview."""
    base = [
        (20, 20, 30),  # 0 transparent
        (255, 255, 255),
        (200, 40, 40),
        (40, 200, 40),
        (40, 90, 220),
        (230, 200, 60),
        (200, 120, 40),
        (180, 60, 200),
        (80, 80, 80),
        (150, 150, 150),
        (255, 150, 150),
        (150, 255, 150),
        (150, 190, 255),
        (255, 230, 120),
        (220, 160, 100),
        (235, 180, 235),
    ]
    ncol = 1 << bpp
    return base[:ncol] + [(128, 128, 128)] * max(0, ncol - len(base))


def render_sprite(lines, pal):
    h = len(lines); w = len(lines[0])
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y, row in enumerate(lines):
        for x, v in enumerate(row):
            px[x, y] = pal[v] if v < len(pal) else (255, 0, 255)
    return img


def render_atlas(sprites, cols=8, scale=3, pad=2):
    if not sprites:
        return None
    maxw = max(s.width for s in sprites)
    maxh = max(s.height for s in sprites)
    rows = (len(sprites) + cols - 1) // cols
    cell_w = maxw * scale + pad * 2
    cell_h = maxh * scale + pad * 2 + 10  # room for label
    out = Image.new("RGB", (cell_w * cols, cell_h * rows), (15, 15, 25))
    for i, s in enumerate(sprites):
        r, c = divmod(i, cols)
        up = s.resize((s.width * scale, s.height * scale), Image.NEAREST)
        out.paste(up, (c * cell_w + pad, r * cell_h + pad))
    return out


# ---------- forward-tiling scan ----------
# Instead of brute-force-per-byte + greedy dedup (which swallows small
# neighbours like "apple"), we walk the ROM forward. When a valid multi-quad
# sprite decodes at pos, we record it and continue at end_pos. If decode fails,
# slide 1 byte and try again. This naturally tiles contiguous sprite data
# regions while still surviving gaps.
def tile_forward(body: bytes, mode: str, bpp: int):
    out = []
    pos = 0
    while pos < len(body) - 2:
        r = decode_sprite_multiquad(body, pos, bpp, mode)
        accepted = False
        if r is not None:
            quads, end, term = r
            # only accept blocks that ended cleanly on a sprite terminator
            stitched = stitch_quads(quads) if len(quads) > 1 else quads[0]
            s = score_sprite(stitched, bpp)
            if s > 0 and term == 'sprite':
                out.append({
                    'mode': mode, 'bpp': bpp, 'off': pos, 'end': end,
                    'quadrants': quads, 'stitched': stitched, 'score': s,
                })
                pos = end
                accepted = True
        if not accepted:
            pos += 1
    return out


# ---------- main scan ----------
def scan(rom_path: str, out_dir: str):
    rom = Path(rom_path).read_bytes()
    body = rom[64:]  # strip 64-byte LYNX header
    print(f"cart body: {len(body)} bytes")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- primary pass: forward-tiling multi-quadrant scan per (mode, bpp). ---
    # This is the new path that should catch whole sprite regions without
    # dedup-eating small neighbours. Results go straight to `tiled`.
    tiled = []  # list of dicts
    for mode in ("rle", "lit"):
        for bpp in BPPS:
            found = tile_forward(body, mode, bpp)
            print(f"  tile_forward  {mode} bpp={bpp}: {len(found)} sprites")
            tiled.extend(found)
    print(f"tiled total: {len(tiled)} sprites")

    # Scan every offset for both packed and literal decoders, each bpp.
    candidates = []  # (score, mode, bpp, offset, lines, end_pos)

    for bpp in BPPS:
        # LITERAL: fixed-width line, easier to detect; scan all offsets
        for off in range(0, len(body) - 4):
            r = decode_sprite_literal(body, off, bpp)
            if r is None:
                continue
            lines, end = r
            s = score_sprite(lines, bpp)
            if s <= 0:
                continue
            candidates.append((s, "lit", bpp, off, lines, end))

        # PACKED/RLE: scan all offsets
        for off in range(0, len(body) - 4):
            r = decode_sprite_packed(body, off, bpp)
            if r is None:
                continue
            lines, end = r
            s = score_sprite(lines, bpp)
            if s <= 0:
                continue
            candidates.append((s, "rle", bpp, off, lines, end))

    print(f"raw candidates: {len(candidates)}")

    # CHAINED SCAN: starting from every raw candidate, greedily walk forward by
    # re-decoding at end_pos. This expands single finds into full animation
    # strips without losing any by overlap dedup.
    chained = []
    seen_starts = set()
    for seed in candidates:
        _, mode, bpp, off, lines, end = seed
        if (mode, bpp, off) in seen_starts:
            continue
        chain = []
        cur = off
        for _ in range(CHAIN_MAX):
            if mode == "lit":
                r = decode_sprite_literal(body, cur, bpp)
            else:
                r = decode_sprite_packed(body, cur, bpp)
            if r is None:
                break
            ls, e = r
            s = score_sprite(ls, bpp)
            if s <= 0:
                break
            chain.append((s, mode, bpp, cur, ls, e))
            seen_starts.add((mode, bpp, cur))
            if e <= cur:
                break
            cur = e
        chained.extend(chain)

    # Keep unique by (mode,bpp,off)
    uniq = {}
    for c in chained:
        key = (c[1], c[2], c[3])
        if key not in uniq or c[0] > uniq[key][0]:
            uniq[key] = c
    all_cands = list(uniq.values())

    # Per-(mode,bpp) non-overlap dedup: sort by start, greedy pick the
    # *longest* (end-start) at each point; then step forward to end_pos.
    chosen = []
    for mode in ("lit", "rle"):
        for bpp in BPPS:
            grp = [c for c in all_cands if c[1] == mode and c[2] == bpp]
            grp.sort(key=lambda t: (t[3], -(t[5] - t[3])))  # by start, prefer longest
            i = 0
            while i < len(grp):
                # among all candidates starting within the same anchor window,
                # keep the longest (covers most bytes without gaps)
                start_i = grp[i][3]
                best = grp[i]
                j = i + 1
                while j < len(grp) and grp[j][3] < best[5]:
                    if (grp[j][5] - grp[j][3]) > (best[5] - best[3]):
                        best = grp[j]
                    j += 1
                chosen.append(best)
                # advance past the chosen sprite's end
                while i < len(grp) and grp[i][3] < best[5]:
                    i += 1

    chosen.sort(key=lambda t: (t[1], t[2], t[3]))
    if len(chosen) > MAX_SCAN_CANDIDATES:
        chosen = chosen[:MAX_SCAN_CANDIDATES]
    print(f"chained unique candidates kept: {len(chosen)}  (from {len(all_cands)} raw unique)")

    # Group into atlases per (mode,bpp)
    groups: dict[tuple[str, int], list] = {}
    for c in chosen:
        groups.setdefault((c[1], c[2]), []).append(c)

    summary = []
    for (mode, bpp), items in sorted(groups.items()):
        items.sort(key=lambda t: t[3])  # by offset
        pal_gray = gray_palette(bpp)
        pal_viv = vivid_palette(bpp)
        imgs_gray = [render_sprite(it[4], pal_gray) for it in items]
        imgs_viv  = [render_sprite(it[4], pal_viv)  for it in items]
        a_g = render_atlas(imgs_gray, cols=16, scale=3)
        a_v = render_atlas(imgs_viv,  cols=16, scale=3)
        if a_g is not None:
            p1 = out / f"atlas_{mode}_bpp{bpp}_gray.png"
            a_g.save(p1)
            p2 = out / f"atlas_{mode}_bpp{bpp}_vivid.png"
            a_v.save(p2)
            summary.append((mode, bpp, len(items), str(p1), str(p2)))
            # save EVERY sprite individually (gray + vivid) sorted by offset,
            # so you can browse them in order the ROM stores them.
            ordered = sorted(items, key=lambda t: t[3])
            for i, it in enumerate(ordered):
                g = render_sprite(it[4], pal_gray)
                v = render_sprite(it[4], pal_viv)
                base = f"{mode}_bpp{bpp}_{i:03d}_off{it[3]:06x}_w{g.width}h{g.height}"
                g.resize((g.width * 4, g.height * 4), Image.NEAREST).save(out / (base + "_gray.png"))
                v.resize((v.width * 4, v.height * 4), Image.NEAREST).save(out / (base + "_vivid.png"))

    print("\n=== summary ===")
    for mode, bpp, n, g, v in summary:
        print(f"  {mode:3s} bpp={bpp}  sprites={n:4d}  atlas: {g}")

    # Coverage map: visualize which ROM bytes are claimed by a decoded sprite.
    # Row = 512 bytes, brightness = claimed; color channel by mode/bpp.
    W = 512
    H = (len(body) + W - 1) // W
    cov = Image.new("RGB", (W, H), (10, 10, 15))
    pxc = cov.load()
    palette_by_mbpp = {}
    import colorsys
    modes = [("lit", b) for b in BPPS] + [("rle", b) for b in BPPS]
    for i, k in enumerate(modes):
        hue = (i / len(modes))
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.85, 0.95)]
        palette_by_mbpp[k] = (r, g, b)
    for c in chosen:
        _, mode, bpp, off, lines, end = c
        col = palette_by_mbpp[(mode, bpp)]
        for p in range(off, min(end, len(body))):
            x, y = p % W, p // W
            pxc[x, y] = col
    cov.resize((W, H * 3), Image.NEAREST).save(out / "_coverage_map.png")
    claimed = sum(c[5] - c[3] for c in chosen)
    print(f"ROM coverage: {claimed} / {len(body)} bytes ({100*claimed/len(body):.1f}%)")

    # --- tiled atlases: what the new forward-tiling scan found. ---
    # Group by (mode, bpp); render stitched (multi-quad composed) pixels.
    tgroups: dict[tuple[str, int], list] = {}
    for t in tiled:
        tgroups.setdefault((t['mode'], t['bpp']), []).append(t)
    for (mode, bpp), items in sorted(tgroups.items()):
        items.sort(key=lambda t: t['off'])
        pal_viv = vivid_palette(bpp)
        imgs = [render_sprite(it['stitched'], pal_viv) for it in items]
        a = render_atlas(imgs, cols=16, scale=3)
        if a is not None:
            p = out / f"atlas_tiled_{mode}_bpp{bpp}.png"
            a.save(p)
            print(f"  tiled atlas {mode} bpp={bpp}: {len(items)} sprites -> {p}")

    # One mega-atlas containing every tiled sprite for fast visual sweep.
    all_imgs = []
    for t in sorted(tiled, key=lambda t: (t['bpp'], t['mode'], t['off'])):
        all_imgs.append(render_sprite(t['stitched'], vivid_palette(t['bpp'])))
    mega = render_atlas(all_imgs, cols=24, scale=2)
    if mega is not None:
        mega.save(out / "ALL_tiled.png")
        print(f"  mega atlas -> {out/'ALL_tiled.png'}  ({len(all_imgs)} sprites)")

    # Tiled coverage map (different palette than the legacy chosen map).
    tcov = Image.new("RGB", (W, H), (5, 5, 10))
    tpx = tcov.load()
    for t in tiled:
        col = palette_by_mbpp[(t['mode'], t['bpp'])]
        for p in range(t['off'], min(t['end'], len(body))):
            x, y = p % W, p // W
            tpx[x, y] = col
    tcov.resize((W, H * 3), Image.NEAREST).save(out / "_coverage_map_tiled.png")
    claimed_t = sum(t['end'] - t['off'] for t in tiled)
    print(f"TILED ROM coverage: {claimed_t} / {len(body)} bytes ({100*claimed_t/len(body):.1f}%)")


if __name__ == "__main__":
    rom = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROM
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    scan(rom, out)
