# Gauntlet: The Third Encounter - Complete Game Reference

**Platform:** Atari Lynx (1990)  
**Developer:** Epyx  
**Publisher:** Atari Corporation  
**Orientation:** Vertical (portrait mode)  
**Players:** 1-4 (via ComLynx cable)

---

## Story

An enchanted Star Gem fell from the stars into an abandoned castle. Loathsome
creatures transported it into the depths of the fortress, gaining magical
strength from it. Unless the gem is removed, the evil ones will grow strong
enough to venture forth and destroy the world. You must battle through 40 levels
of the castle to retrieve the Star Gem.

---

## Characters (8 Classes)

Each character has unique stats for Speed, Strength, and Missiles. Stats never
fall below initial values during gameplay.

| Character    | Speed | Strength | Missiles |
|-------------|-------|----------|----------|
| Android     | 24    | 7        | 9        |
| Valkyrie    | 40    | 3        | 5        |
| Gunfighter  | 32    | 4        | 7        |
| Nerd        | 26    | 2        | 4        |
| Pirate      | 28    | 5        | 5        |
| Punkrocker  | 29    | 5        | 6        |
| Samurai     | 34    | 6        | 5        |
| Wizard      | 28    | 3        | 8        |

**In multiplayer, each character can only be chosen by one player.**

---

## Stats

- **Life** - Starts at 20,000. Decreases from enemy attacks, touching enemies,
  and slowly over time even when idle. Reaches 0 = death. Increased by food,
  some scrolls, and some potions.
- **Speed** - How fast you can move. Increased by Speed scrolls or red potions.
- **Strength** - Reduces damage taken; determines carrying capacity. Increased
  by Strength scrolls or blue potions. Potion effect is temporary (one level);
  scroll effect is permanent.
- **Missiles** - Damage dealt by ranged attacks. Increased by Shots scrolls or
  green potions.

---

## Controls (Original Lynx)

- **Joypad** - Move character in 8 directions
- **A button** - Fire missiles / attack
- **B button (hold)** - View Inventory window
- **B + Left/Right** - Scroll through inventory items
- **B + Up** - USE the displayed inventory item
- **B + Down** - DROP the displayed inventory item
- **OPTION 1 + PAUSE** - Return to title screen
- **OPTION 2 + PAUSE** - Flip screen 180 degrees
- **PAUSE** - Pause/unpause

---

## Screen Layout

The screen is divided into two sections:

### Action Window (top)
Scrolling overhead map showing your character, objects, enemies, and the
surrounding area. Other players in the same area also appear here.

### Statistics / Radar / Inventory Window (bottom)
Divided into two sub-sections:

**Default view (Stats/Radar):**
- **Left panel** - Detailed close-up picture of what you are approaching or
  the last significant thing encountered (nearby enemy, item, etc.)
- **Right panel** - Life, Speed, Strength, Missiles, Level, Score

**Inventory view (hold B):**
- **Left panel** - Detailed picture of the currently selected inventory item
- **Right panel** - Life rating + scrollable inventory list with USE/DROP

---

## Items

Items are picked up by walking over them and added to your inventory.

### Automatically Used Items

| Item | Effect |
|------|--------|
| **Gold** | Currency for purchasing at computer terminals. Automatically deducted on purchase. |
| **Keys** | Open wooden doors. Walk up to a closed door and a key is consumed automatically. |
| **Card Keys** | Open laser doors. Same behavior as keys but for tech doors. |

### Inventory Items (Must USE from Inventory)

| Item | Effect |
|------|--------|
| **Green Apple** | +250 life points |
| **Red Apple** | +500 life points |
| **Potions (various)** | See Potions table below |
| **Scrolls (various)** | See Scrolls table below |

### Potions

Potion effects are **temporary** (last until next level) unless noted.

| Potion | Effect |
|--------|--------|
| **Red Potion** | Increases Speed rating (temporary, one level) |
| **Blue Potion** | Increases Strength rating (temporary, one level) |
| **Green Potion** | Increases Missiles rating (temporary, one level) |
| **Life Potion** | +1000 life points (permanent) |
| **Poison** | Reduces life rating (permanent - be careful!) |

### Scrolls

Scroll effects are generally **permanent** unless noted with a duration.

| Scroll | Effect |
|--------|--------|
| **Revive** | Brings character back to life. Must be used during death spin animation. In multiplayer, can revive a dead comrade. |
| **Invis** | Invisible for 25 seconds |
| **Farsee** | View the entire level map. Use joypad to pan. Press A to return to normal view. |
| **Blast** | Kills all enemies currently visible in the Action Window. Does not harm other players. |
| **Heal** | +2500 life points |
| **Shots** | +2 Missiles rating (permanent, rest of game) |
| **Speed** | +4 Speed rating (permanent, rest of game) |
| **Strong** | +2 Strength rating (permanent, rest of game) |
| **Repel** | Enemies avoid you for 25 seconds |
| **Power** | Boosts Missiles, Speed, and Strength for 30 seconds |

---

## Computer Terminals

Special interactive objects found on some levels. You **cannot** pick them up.

- Some terminals display **vital messages** (hints, lore, warnings).
- Some terminals act as **shops** where you can purchase potions, scrolls, and
  food using gold from your inventory.
- Purchasable items appear **in the vicinity** of the terminal. Walk over an
  item near a terminal to buy it (gold is auto-deducted).
- If you don't have enough gold, you can't purchase.

---

## Enemies

| Enemy | Points | Notes |
|-------|--------|-------|
| **Slime** | 0 | Divides when hit in the open. Lure into hallways/doorways where they can't split, then attack. |
| **Spider** | 1 | Basic enemy |
| **Ladybug** | 4 | Slightly tougher |
| **Ghost** | 10 | Ethereal, may pass through walls |
| **Cactus** | 15 | Stationary hazard |
| **Frog** | 15 | |
| **Monk** | 15 | |
| **Scorpion** | 20 | |
| **Land Shark** | 40 | Dangerous predator |
| **Boulder** | 255 | Very tough obstacle |

**Enemy behavior:**
- Some enemies will leave you alone if you don't approach or attack them first.
- Enemies who touch your character sap life points (your character flashes).
- If a creature touches you, the monster disappears (contact = mutual damage).
- Generators/spawn points create new enemies over time. Destroy them to stop spawning.

---

## Level Structure

- **40 levels** in total
- Levels are presented in a **fixed order** with fixed layouts (not random)
- Some levels lead through **dimension doors** into strange universes
- **Illusory walls** - some walls can be walked through (they only look solid)
- **Doors** - require keys to open (walk adjacent to auto-use key)
- **Laser doors** - require card keys
- **Exit** - warp pad to advance to the next level
- Final level: find the **Star Gem** and discover its secret

### Level Skip (Cheat)
At the start of level 1, before any player moves, press OPTION 1 to choose
starting level: 1, 5, 10, 15, or 20.

---

## Scoring

| Action | Points |
|--------|--------|
| Slime | 0 |
| Spider | 1 |
| Ladybug | 4 |
| Ghost | 10 |
| Cactus | 15 |
| Frog | 15 |
| Monk | 15 |
| Scorpion | 20 |
| Land Shark | 40 |
| Opening a Door | 100 |
| Boulder | 255 |

---

## Strategy Tips (from the manual)

1. **Select your character wisely** based on your play style.
2. **Don't shoot food, potions, or scrolls** - you can destroy them.
3. **Some walls are not solid** - look for hidden passages.
4. **Some enemies are passive** until provoked.
5. **Attack from a distance** - melee contact costs life points.
6. **Slimes divide in open areas** - funnel them into corridors first.
7. **Drop unwanted items to create barriers** against enemies.
8. **You can attack and view inventory simultaneously.**
9. **Revive scroll timing** - have it ready before you die; must use during
   death spin animation.
10. **Loot dead allies** in multiplayer - they won't need their items anymore.

---

## Differences from Original Gauntlet (1985)

- **8 character classes** instead of 4 (original had Warrior, Valkyrie,
  Wizard, Elf)
- **Inventory management system** - items are carried and must be actively used
- **Computer terminals** for purchasing supplies
- **Scrolls** with varied magical effects (Invis, Farsee, Blast, etc.)
- **Vertical/portrait orientation** on the Lynx
- **Fixed level layouts** (not procedurally generated like the arcade)
- **Radar/detail window** showing close-ups of nearby objects
- **Gold economy** for buying from terminals
- **Illusory walls** as a puzzle element

---

*Source: Official Atari Lynx manual via AtariAge, IGN review, Wikipedia*
