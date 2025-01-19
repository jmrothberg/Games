# 3D Tank Battle Game

A Python-based 3D tank battle game featuring both single-player and multiplayer modes, powered by Pygame.

## Features

- **Multiple Game Modes**
  - Single-player against AI tanks
  - Two-player networked multiplayer (host/join)
  - Automatic network discovery for finding local games

- **3D Graphics**
  - Rendered using Pygame's 2D engine with 3D perspective calculations
  - Dynamic camera system with scope view and minimap
  - Real-time shadows and explosions

- **Gameplay Elements**
  - Multiple weapon systems (Cannon, Minigun, Rockets)
  - Power-ups (Health, Speed, Cooldown)
  - Destructible environment
  - Sound effects for shooting and impacts

- **Difficulty Levels**
  - Woke (Easy)
  - Medium
  - Based (Hard)

## Requirements
pip install pygame
pip install python-nmap


## Controls

- **Arrow Keys**: Move tank forward/backward and rotate
- **Q/Z**: Adjust barrel angle up/down
- **Spacebar**: Fire weapon
- **H**: Toggle help screen
- **R**: Reset game (when game over)
- **X**: Exit game (when game over)

## Network Setup

For multiplayer mode:
1. First player selects "Host" and shares their IP address
2. Second player selects "Join" and enters the host's IP address
3. Game automatically synchronizes between players

## Difficulty Settings

Select difficulty at game start:
- **W**: Woke (Easy) - More obstacles, weaker enemies, faster player
- **M**: Medium - Balanced gameplay
- **B**: Based (Hard) - Fewer obstacles, stronger enemies, slower player

## Power-ups

- **Health** (Orange): Restore tank health
- **Speed** (Cyan): Increase movement speed
- **Cooldown** (Magenta): Reduce weapon cooldown
- **Weapons**:
  - Minigun (Purple): Rapid-fire, low damage
  - Cannon (White): Standard weapon, balanced damage
  - Rocket (Pink): Slow-fire, high damage

## Development

Created by JMR, July 2024
- Initial release: June 27, 2024
- Latest update: July 21, 2024

## Known Issues

- Network reconnection may be unstable in some situations
- Sound effects require specific WAV files to be present

## Future Enhancements

- Additional weapon types
- More power-up varieties
- Enhanced AI behavior
- Improved network stability
- Additional game modes

## License

This project is open source and available under the MIT License.
