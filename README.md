# JMR's Games Collection

![Games](https://img.shields.io/badge/Games-HTML5%20%2B%20Python-blue)
![Mobile](https://img.shields.io/badge/Mobile-iOS%20%2B%20Android-green)

A collection of games by Jonathan M. Rothberg - playable on desktop and mobile (iPhone/iPad).

---

## Mobile HTML5 Games

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

## Desktop Python Games

### 3D Tank Battle (Python - Multiplayer)
**Files:** `3D_Network_Tank_Feb_4_25.py`, `3D_Network_Tank_July_21_24_rockets.py`

The original Python version with networked multiplayer support.

- **Requirements:** `pip install pygame python-nmap`
- **Controls:**
  - Arrow Keys: Move/rotate tank
  - Q/Z: Adjust barrel angle
  - Spacebar: Fire
- **Features:**
  - 3D graphics with Pygame
  - Networked multiplayer (host/join)
  - Multiple weapons (Cannon, Minigun, Rockets)
  - Difficulty levels: Woke (Easy), Medium, Based (Hard)

See `READMEDOOMNETWORK.MD` for more details.

---

### Doom-style Game
**File:** `doom3_Feb_6_7pm.py`

A Doom-inspired first-person game.

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
| Tank Battle (Python) | Desktop | Download & run with Python |

---

## Author

Created by JMR, 2024-2025

## License

MIT License
