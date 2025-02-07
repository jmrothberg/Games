# JMR's DOOM-like Game

A Python-based 3D first-person shooter inspired by the classic DOOM, featuring both single-player and multiplayer modes.

## Features

- **3D Raycasting Engine**: Custom-built 3D rendering system with textured walls and sprite-based enemies
- **Multiplayer Support**: Host or join games over local network with real-time player synchronization
- **Dynamic Combat**: Multiple weapons, power-ups, and enemy types
- **Procedural Level Generation**: Randomly generated maze-like levels for unique gameplay experiences
- **Advanced AI**: Enemies use pathfinding to track and engage players
- **Modern Graphics**: Texture mapping, dynamic lighting, and sprite scaling
- **Interactive HUD**: Real-time health, ammo, and score tracking

## Controls

- **Arrow Keys**: Move and turn
- **Space**: Shoot
- **Left Mouse Button**: Change weapon
- **H**: Show help menu

## Game Modes

- **Single Player**: Fight against AI enemies in a procedurally generated maze
- **Host Game**: Create a multiplayer session for others to join
- **Join Game**: Connect to an existing multiplayer session

## Technical Details

- Built with Python and Pygame
- Network implementation using sockets for multiplayer
- A* pathfinding for enemy AI
- Ray-casting for 3D rendering
- Vector-based collision detection
- Pickle-based network serialization

## Requirements

- Python 3.x
- Pygame
- NumPy
- Pathfinding library

## Installation

1. Clone the repository
2. Install required packages: `pip install pygame numpy pathfinding`
3. Run the game: `python doom3_Feb_6_7pm.py`

## Multiplayer Setup

- Host: Select "Host Game" and share your IP address
- Client: Select "Join Game" and enter the host's IP address
- Default port: 65432

## Credits

Created by JMR with assistance from Claude AI
Graphics and sound effects from various open-source resources

## License

Open source - Feel free to modify and distribute with attribution
