# JMR's Games Collection

![Games](https://img.shields.io/badge/Games-HTML5-blue)
![Mobile](https://img.shields.io/badge/Mobile-iOS%20%2B%20Android-green)

A collection of HTML5 games by Jonathan M. Rothberg - playable on desktop and mobile (iPhone/iPad).

---

## HTML5 Games

These games work directly in your browser on iPhone, iPad, or desktop. No installation needed!

### JMR's Asteroids
**File:** `asteroids.html`

Classic asteroid-shooting action with touch controls.

- **Controls:** Tap left/right to rotate, tap thrust to move, tap fire to shoot
- **Features:**
  - "Gene Machine" ship label
  - Retina display support
  - Touch-optimized for mobile

**Play:** `https://raw.githack.com/jmrothberg/Games/main/asteroids.html`

---

### JMR's Pac-Man
**File:** `pacman.html` (Version 7)

Classic maze chase with ghosts!

- **Controls:**
  - D-pad buttons on screen
  - Swipe in any direction
- **Features:**
  - 4 colored ghosts with AI
  - Power pellets to eat ghosts
  - Touch and swipe controls

**Play:** `https://raw.githack.com/jmrothberg/Games/main/pacman.html`

---

### JMR's Iron Dome
**File:** `missile.html` (Version 9)

Defend Israeli cities from incoming attacks! Inspired by Missile Command.

- **Controls:**
  - Drag to aim crosshair
  - Tap/release to launch interceptor
- **Features:**
  - 6 cities to defend (Tel Aviv, Haifa, Beersheba, Eilat, Netanya, Ashkelon)
  - Multiple enemy types:
    - Missiles (some are MIRVs that split!)
    - Drones with buzzing sound (Wave 2+)
    - Paragliders that drop terrorists (Wave 3+)
    - Terrorists that run toward cities
  - Wave names: "Intifada" (Wave 2), "Al-Aqsa Flood" (Wave 3+)
  - QR code score verification for contests
  - Sound effects

**Play:** `https://raw.githack.com/jmrothberg/Games/main/missile.html`

**Score Verification:** `verify.html` - Scan QR codes to verify contest scores

---

### JMR's Space Invaders
**File:** `invaders.html` (Version 1)

Classic alien invasion arcade action!

- **Controls:**
  - Tap left side to move left
  - Tap right side to move right
  - Tap center to shoot
  - Keyboard: Arrow keys + Space
- **Features:**
  - 5 rows of different enemy types (monsters and stick figures)
  - Destructible barriers for cover
  - Enemies speed up as you destroy them
  - Wave progression
  - Sound effects
  - Commented code for adding custom images

**Play:** `https://raw.githack.com/jmrothberg/Games/main/invaders.html`

---

### JMR's Mr. Do!
**File:** `mrdo.html` (Version 1)

Classic dig-and-collect arcade action! Based on the 1982 Universal classic.

- **Controls:**
  - D-pad to move and dig
  - Center button to throw power ball
  - Keyboard: Arrow keys + Space
- **Features:**
  - Dig tunnels through dirt
  - Collect cherries (8 in a row = 500 bonus!)
  - Throw bouncing power ball at enemies
  - Push apples to crush enemies
  - Red blob enemies that chase you through tunnels
  - Digger enemies that create their own paths
  - Alphamonsters carry E-X-T-R-A letters for extra life
  - Multiple ways to complete each level
  - Sound effects

**Play:** `https://raw.githack.com/jmrothberg/Games/main/mrdo.html`

---

### JMR's 3D Tank Battle
**File:** `3D_Tank_Battle.html`

3D wireframe tank battle with enemy AI!

- **Controls:**
  - D-pad: Move forward/back, rotate left/right
  - Center button: Fire
  - Q↑/Z↓: Adjust barrel angle
  - Swipe up/down: Move
  - Swipe left/right: Rotate
  - Tap: Fire
  - H: Help
- **Features:**
  - 3D wireframe graphics
  - Scope view with range finder
  - Minimap with enemy/power-up tracking
  - 3 difficulty levels: Woke (Easy), Medium, Based (Hard)
  - 6 power-up types (Health, Cooldown, Speed, Minigun, Rocket, Cannon)
  - Enemy AI that pursues and attacks

**Play:** `https://raw.githack.com/jmrothberg/Games/main/3D_Tank_Battle.html`

---

### JMR's Gauntlet: The Third Encounter
**File:** `gauntlet_third_encounter.html`

Single-file HTML/JS recreation of the 1990 Atari Lynx classic — top-down action
RPG in portrait orientation (320×480), no external assets, all sprites baked
from 2D pixel-index arrays at load.

- **Controls (keyboard):**
  - Arrows / WASD — move
  - J — melee attack
  - K — shoot missile
  - I — toggle stats ↔ inventory HUD
  - 1–4 / U — use inventory slot
  - X — drop inventory slot
  - C — toggle **cheat mode** (pins life = 9999, missiles = 99, blocks damage and HP decay)
  - H — help overlay (controls + tile legend, pauses the game)
  - Space — confirm / start / continue
- **Controls (touch):** on-screen D-pad + MELEE / SHOOT / INV / START
- **Features:**
  - All 8 Lynx character classes — Warrior, Valkyrie, Wizard, Archer, Android, Samurai, Pirate, Punk — with per-class stats and unique 16×16 + portrait sprites
  - All core monsters: Scorpion, Skeleton, Ghost (phases through walls), Grunt, Demon (fireballs), Sorcerer (teleports + magic bolts), Lobber (arcing rocks), Death (HP drain), plus **IT** — the 32×32 final boss
  - Enemy generators — pulsing framed tiles that spawn their monster type until destroyed
  - Star Gem quest — one fragment hidden per level 2–9; collecting all 8 changes the ending text
  - Procedurally-generated dungeons with a solvability guarantee — the generator simulates the player collecting keys and opening doors, planting additional keys in already-reachable rooms until the exit is provably reachable
  - Biome rotation (grass / dirt) per level
  - HP decay tick (the series's "Warrior needs food, badly" hallmark) + low-HP red pulse
  - Inventory: potions bomb nearby enemies + heal; food heals instantly
  - Portrait HUD: class portrait blinks, flashes red on damage, dedicated GEMS row
  - localStorage high score on title
  - Procedural Web Audio sound effects (no sound files)
  - Two-frame walk + attack pose per class × 4 facings
  - Letterboxed integer-multiple scaling — renders crisp at any size
- **About the "portals":** there is **one** transport tile — the pulsing blue/yellow exit swirl, now labelled **EXIT** in-game. The red/bronze framed pulsing tile is a **generator** (spawns enemies, attack it to destroy). Wooden panels are **locked doors** (need a key — walk up to one while holding a key to open). Press H in-game for the full tile legend.

**Play:** [https://jmrothberg.github.io/Games/gauntlet_third_encounter.html](https://jmrothberg.github.io/Games/gauntlet_third_encounter.html)

---

### JMR's Chess (Human / Search / LLM)
**File:** `Generated_games/chess.html` + `Generated_games/chess_server.py`

Full-rules chess with independent per-side player choice — any of:

- **Human** — click-to-move, with a promotion overlay. Works anywhere the page loads.
- **Search** — built-in minimax + alpha-beta with iterative deepening, MVV-LVA move ordering, and quiescence. Depth 1–7. Works anywhere the page loads.
- **LLM** — a PyTorch chess transformer loaded from `Generated_games/Chess_LLM_models copy/*.pth`. **Requires running `chess_server.py` locally** — the LLM dropdown is disabled when the bridge isn't reachable.

> **Note on the web link:** opening `chess.html` via the Quick Links table (GitHub / raw.githack) gives you **Human + Search only**. To use an LLM you must clone the repo, place `.pth` checkpoints in `Chess_LLM_models copy/`, and run `chess_server.py` on your own machine — model files are 1.7–4.7 GB and can't be hosted or loaded in a browser.

Pick any combination on the two dropdowns (White Player / Black Player). Human-vs-LLM, Search-vs-LLM, and full self-play (LLM-vs-LLM, Search-vs-Search) all work from a single **New Game** click.

**Running the LLM bridge** (required only if you want the LLM option; Human and Search work as a plain HTML page):

```
python3 "Generated_games/chess_server.py"
# then open http://localhost:5858/chess.html
```

The bridge is stdlib-only (`http.server`) — no pip installs beyond PyTorch itself. It serves `chess.html`, lists the `.pth` files in `Generated_games/Chess_LLM_models copy/`, and on each LLM turn returns the top-20 UCI candidate moves from the selected model. Models are lazy-loaded and cached.

**LLM stats row** — under the status line, the page shows per-side:

- `last rank` — rank (1 = top pick) of the legal move the LLM played this turn
- `avg` / `max` — running average and worst rank across the game
- `fallbacks` — how many times none of the top 20 were legal and the built-in search (depth 2) had to rescue the move

---

## Play Games Generated by CodeRunner

| Game | Play | Model |
|------|------|-------|
| **Vector Tanks** | [Play](https://jmrothberg.github.io/Games/Generated_games/vector_oneshot_great.1.0.html) | GLM-5.1 |
| **Vader's One Shot** | [Play](https://jmrothberg.github.io/Games/Generated_games/vaders_oneshot_great_MM2_7_8bit.html) | MiniMax-M2.7-8bit |

---

## Quick Links

| Game | Platform | Link |
|------|----------|------|
| Asteroids | Mobile/Desktop | [Play](https://raw.githack.com/jmrothberg/Games/main/asteroids.html) |
| Pac-Man | Mobile/Desktop | [Play](https://raw.githack.com/jmrothberg/Games/main/pacman.html) |
| Iron Dome | Mobile/Desktop | [Play](https://raw.githack.com/jmrothberg/Games/main/missile.html) |
| Space Invaders | Mobile/Desktop | [Play](https://raw.githack.com/jmrothberg/Games/main/invaders.html) |
| Mr. Do! | Mobile/Desktop | [Play](https://raw.githack.com/jmrothberg/Games/main/mrdo.html) |
| 3D Tank Battle | Mobile/Desktop | [Play](https://raw.githack.com/jmrothberg/Games/main/3D_Tank_Battle.html) |
| Gauntlet: The Third Encounter | Mobile/Desktop | [Play](https://jmrothberg.github.io/Games/gauntlet_third_encounter.html) |
| Vector Tanks | Desktop | [Play](https://jmrothberg.github.io/Games/Generated_games/vector_tanks.html) |
| Chess (Human + Search only — LLM needs local bridge) | Desktop | [Play](https://raw.githack.com/jmrothberg/Games/main/Generated_games/chess.html) |

---

## Author

Created by JMR, 2024–2026

## License

MIT License
