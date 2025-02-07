#JMR Doom June 22 2023 with the help of Claude Sonnet 3.5. Had to hand do the angle calculations, LLM could not get them right for looking at the monsters.
#Make the monsters appear further away but smaller, they only appear when they are close to the player.
# August 31, 2024 added graphics for powerups, made monsters look more 3D, added weapons, and opened up maze, adjusted hit area and aim
# February 7, 2025 added multiplayer, added score, and game over screen
import pygame
import math
import random
import os
from pygame.math import Vector3
import numpy as np
import pathfinding
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import socket
import pickle
import threading
import time
import subprocess
import nmap
import ipaddress
import re
import queue
from typing import Optional, List, Dict, Any

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Screen settings
WIDTH = 1200
HEIGHT = 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("JMR's DOOM-like Game")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)    
MAGENTA = (255, 0, 255)
DARK_GRAY = (50, 50, 50)
BROWN = (139, 69, 19)
ORANGE = (255, 165, 0)  
PURPLE = (128, 0, 128)  

colors = [RED, YELLOW, CYAN, MAGENTA,ORANGE, PURPLE]  
epsilon = 0.00001

# Player settings
player_x = 100
player_y = 100

player_speed = 3
player_health = 100
player_score = 0

# Map settings
MAP_SIZE = 16
TILE_SIZE = 64

# Weapon settings
weapons = ['Pistol', 'Shotgun', 'Plasma Gun']
current_weapon = 0

# Raycasting settings
FOV = math.pi / 3
HALF_FOV = FOV / 2
CASTED_RAYS = 240
STEP_ANGLE = FOV / CASTED_RAYS
MAX_DEPTH = 1200

MAX_MONSTERS = 10
MAX_POWERUPS = 10

# Minimap settings
MINIMAP_SCALE = 0.2

# New 3D rendering constants
SCREEN_DIST = 416
SCALE = 150
H_WIDTH = WIDTH // 2
H_HEIGHT = HEIGHT // 2

# Sound effects
shoot_sound = pygame.mixer.Sound('mixkit-arcade-game-explosion-2759.wav')
hit_sound = pygame.mixer.Sound('mixkit-explosive-impact-from-afar-2758.wav')
#monster graphics from png file make a list of monster images
monster_images = []
monster_images.append(pygame.image.load('bird1_transparent.png'))
monster_images.append(pygame.image.load('bird2_transparent.png'))
monster_images.append(pygame.image.load('bird3_transparent.png'))

# New global variables
MONSTER_SHOOT_COOLDOWN = 100  # Frames between monster shots
BULLET_SPEED = 10  # Speed of bullets
BULLET_DAMAGE = 10  # Damage dealt by bullets

monster_damage_value = 0
power_up_timer = 0

# Add these lines near the top of your file, after initializing pygame
WALL_TEXTURE = pygame.image.load('doom_wall_texture.png')
WALL_TEXTURE = pygame.transform.scale(WALL_TEXTURE, (64, 64))

# Load weapon images
def load_and_resize_weapon(filename):
    image = pygame.image.load(filename).convert_alpha()
    return pygame.transform.scale(image, (image.get_width() // 4, image.get_height() // 4))

pistol_image = load_and_resize_weapon('pistol.png')
machinegun_image = load_and_resize_weapon('machinegun.png')

# Add near other global variables:
wall_slice_cache = {}

# Network settings
is_server = False
client_socket = None
server_socket = None
HOST = None
PORT = 65432
DEFAULT_HOST = None

# Get local IP address (same as tank game)
ip_command_output = subprocess.check_output('ifconfig', shell=True).decode()
inet_patterns = re.compile(r'inet (\d+\.\d+\.\d+\.\d+)')
for line in ip_command_output.split('\n'):
    if 'inet ' in line and '127.0.0.1' not in line:
        ip_address = re.findall(inet_patterns, line)[0]
        break
print(f'Real IP Address: {ip_address}')
DEFAULT_HOST = ip_address
HOST = ip_address

player_server_image = pygame.image.load('player_server.png')
player_client_image = pygame.image.load('player_client.png')

enemy_player = None  # Will store the other player in multiplayer
enemy_damage_value = 0


class Player:
    def __init__(self, position, direction, is_server=False):
        self.pos = Vector3(position)
        self.direction = Vector3(direction).normalize()
        self.type = "player"
        self.health = 100
        self.speed = 200
        self.size = 30
        # Use is_server to determine color and image
        self.colors = BLUE if is_server else YELLOW
        self.image = player_server_image if is_server else player_client_image
        self.weapon = "Pistol"
        self.weapon_spread = 0.01
        self.weapon_damage = 10
        self.weapon_cooldown_time = 10
        self.weapon_crosshair_color = GREEN
        self.weapon_crosshair_size = 10
        self.ammo = 100
        self.cooldown = 0
        self.score = 0
        self.angle = math.atan2(self.direction.z, self.direction.x)
        self.is_server = is_server

    def __getstate__(self):
        state = self.__dict__.copy()
        # Convert Vector3 objects to tuples so they are pickleable.
        state['pos'] = (self.pos.x, self.pos.y, self.pos.z)
        state['direction'] = (self.direction.x, self.direction.y, self.direction.z)
        # Remove the pygame.Surface (image) which is not pickleable.
        if 'image' in state:
            del state['image']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.pos = Vector3(state['pos'])
        self.direction = Vector3(state['direction'])
        # Reinitialize the image based on is_server flag
        self.image = player_server_image if self.is_server else player_client_image

    def move(self, forward, dt):
        movement = self.direction * (self.speed * dt if forward else -self.speed * dt)
        new_pos = self.pos + movement
        if not is_collision(new_pos.x, new_pos.z):
            self.pos = new_pos
    
    def rotate(self, angle):
        self.angle += angle
        self.angle %= 2 * math.pi
        self.direction = Vector3(
            math.cos(self.angle),
            0,
            math.sin(self.angle)
        ).normalize()

    def shoot(self):
        if self.cooldown == 0 and self.ammo > 0:
            shoot_sound.play()
            self.cooldown = self.weapon_cooldown_time
            self.ammo -= 1
            return True
        return False

    def update(self):
        if self.cooldown > 0:
            self.cooldown -= 1

class Monster:
    def __init__(self, x, y, z, monster_type):
        self.pos = Vector3(x, y, z)
        self.type = monster_type
        self.image = monster_images[0] if monster_type == 'bird1' else monster_images[1] if monster_type == 'bird2' else monster_images[2]
        self.health = 100 if monster_type == 'bird1' else 200 if monster_type == 'bird2' else 300
        self.speed = .5 if monster_type == 'bird1' else 1 if monster_type == 'bird2' else 1.5 
        self.size = 200 if monster_type == 'bird1' else 300 if monster_type == 'bird2' else 400
        self.scaled_image = pygame.transform.scale(self.image, (self.size, self.size))  
        self.color = RED if monster_type == 'bird1' else BROWN if monster_type == 'bird2' else YELLOW
        self.shoot_cooldown = 0
        self.shoot_probability = 0.05  # 5% chance to shoot per frame when in line of sight
        # Add these new attributes
        self.screen_x = 0
        self.screen_y = 0
        self.screen_width = 0
        self.screen_height = 0

    def shoot(self, player):
        if self.shoot_cooldown == 0:
            dx = player.pos.x - self.pos.x
            dz = player.pos.z - self.pos.z
            distance = math.sqrt(dx*dx + dz*dz)
            
            # Calculate the probability of hitting based on distance
            hit_probability = max(0.1, 1 - (distance / 1000))  # 10% minimum chance, decreases with distance
            shoot_sound.play()
            if random.random() < hit_probability:
                damage = random.randint(5, 15)  # Random damage between 5 and 15
                player.health -= damage
                hit_sound.play()
            
            self.shoot_cooldown = MONSTER_SHOOT_COOLDOWN

    def update(self):
        # Update monster's cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

    def __getstate__(self):
        state = self.__dict__.copy()
        state['pos'] = (self.pos.x, self.pos.y, self.pos.z)
        # Don't pickle pygame surfaces
        state.pop('image', None)
        state.pop('scaled_image', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.pos = Vector3(state['pos'])
        # Restore images
        self.image = monster_images[0] if self.type == 'bird1' else monster_images[1] if self.type == 'bird2' else monster_images[2]

# Updated PowerUp class for 3D rendering
class PowerUp:
    def __init__(self, x, y, z, power_type):
        self.pos = Vector3(x, y, z)
        self.type = power_type
        self.size = 200  # This should be similar to the monster size
        self.power = random.choice(['health', 'ammo'])
        self.color = GREEN if self.power == 'health' else BLUE
        if self.power == 'health':
            self.healing = random.randint(10, 50)
            self.ammo = 0
        else:
            self.ammo = random.randint(10, 50)
            self.healing = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state['pos'] = (self.pos.x, self.pos.y, self.pos.z)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.pos = Vector3(state['pos'])

# Function to convert world coordinates to screen coordinates
def world_to_screen(world_pos):
    dx = world_pos.x - player.pos.x
    dz = world_pos.z - player.pos.z
    distance = math.sqrt(dx*dx + dz*dz)
    
    angle = math.atan2(dz, dx) - player.angle
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    
    if abs(angle) > HALF_FOV:
        return None
    
    screen_x = (angle / HALF_FOV + 1) * WIDTH / 2
    screen_y = HEIGHT / 2 - HEIGHT / (2 * distance)
    
    return (int(screen_x), int(screen_y))


def generate_complex_maze(width, height):
    """Generate a more complex maze using a recursive backtracking algorithm with added openings."""
    maze = [[1 for _ in range(width)] for _ in range(height)]
    
    def carve_path(x, y):
        maze[y][x] = 0
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx * 2, y + dy * 2
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 1:
                maze[y + dy][x + dx] = 0
                carve_path(nx, ny)
    
    carve_path(1, 1)
    
    # Add random openings in long passages
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if maze[y][x] == 1:
                # Check for long horizontal walls
                if maze[y][x-1] == maze[y][x+1] == 0 and random.random() < 0.2:
                    maze[y][x] = 0
                # Check for long vertical walls
                elif maze[y-1][x] == maze[y+1][x] == 0 and random.random() < 0.2:
                    maze[y][x] = 0
    
    # Ensure no enclosed areas
    for y in range(1, height - 1, 2):
        for x in range(1, width - 1, 2):
            if maze[y][x] == 1:
                if random.random() < 0.5:
                    maze[y][x] = 0
                else:
                    maze[y][random.choice([x-1, x+1])] = 0
    
    return maze


def is_collision(x,z ):
    col = int(x / TILE_SIZE)
    row = int(z / TILE_SIZE)
    is_collision = 0 <= row < MAP_SIZE and 0 <= col < MAP_SIZE and map_data[row][col] == 1
    #print(f"Collision at x,y: {x}, {y}, row, col: {row}, {col}, is_collision: {is_collision}")
    return is_collision


def handle_movement(dt):
    keys = pygame.key.get_pressed()
    turn_speed = math.radians(60)  # Turn speed (120° per second)
    if keys[pygame.K_LEFT]:
        player.rotate(-turn_speed * dt)
    if keys[pygame.K_RIGHT]:
        player.rotate(turn_speed * dt)
    
    # Optionally, if you want to change the movement speed, define a move speed (pixels/second)
    # and pass it into the player's move method.
    if keys[pygame.K_UP]:
        player.move(True, dt)
    if keys[pygame.K_DOWN]:
        player.move(False, dt)


def handle_shooting():
    global monsters, player_score, current_weapon, player, enemy_player,enemy_damage_value
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if player.shoot():  # If shot fired successfully
                    enemy_hit_detected = False
                    monster_damage_value = 0  # We'll set this if we hit something
                    
                    # Process monster hits (identical for both server and client)
                    for monster in monsters[:]:
                        if (abs(monster.screen_x + monster.screen_width/2 - WIDTH/2) < 50 and
                            abs(monster.screen_y + monster.screen_height/2 - HEIGHT/2) < 50):
                            monster_damage_value = player.weapon_damage
                            monster.health -= monster_damage_value
                            enemy_hit_detected = True
                            hit_sound.play()
                            pygame.draw.circle(screen, RED, (WIDTH//2, HEIGHT//2), 20, 2)
                            pygame.display.flip()
                            pygame.time.wait(50)
                            print(f"Monster hit, health: {monster.health}")
                            if monster.health <= 0:
                                monsters.remove(monster)
                                player.score += 100  # Award score when a monster is killed
                            
                    # Process enemy player hits in multiplayer (if applicable)
                    if mode != "singleplayer" and enemy_player:
                        dx = enemy_player.pos.x - player.pos.x
                        dz = enemy_player.pos.z - player.pos.z
                        angle_to_enemy = math.atan2(dz, dx)
                        relative_angle = (angle_to_enemy - player.angle + math.pi) % (2 * math.pi) - math.pi
                        
                        if abs(relative_angle) < math.radians(10):
                            distance = math.sqrt(dx*dx + dz*dz)
                            if distance < 30:  # Hit detection radius
                                enemy_damage_value = player.weapon_damage
                                print(f"Enemy player hit, damage: {enemy_damage_value}")
                                enemy_hit_detected = True
                                pygame.draw.circle(screen, ORANGE, (WIDTH//2, HEIGHT//2), 20, 2)
                                pygame.display.flip()
                                pygame.time.wait(50)
                    
                    if not enemy_hit_detected:
                        pygame.draw.circle(screen, WHITE, (WIDTH//2, HEIGHT//2), 20, 2)
                        pygame.display.flip()
                        pygame.time.wait(50)

            if event.key == pygame.K_h:
                show_help()
                
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button: change weapon
                current_weapon = (current_weapon + 1) % len(weapons)
                player.weapon = weapons[current_weapon]
                if player.weapon == 'Pistol':
                    player.weapon_spread = 0.01
                    player.weapon_damage = 10
                    player.weapon_cooldown_time = 10
                    player.weapon_crosshair_color = GREEN
                    player.weapon_crosshair_size = 10
                elif player.weapon == 'Shotgun':
                    player.weapon_spread = 0.05
                    player.weapon_damage = 20
                    player.weapon_cooldown_time = 30
                    player.weapon_crosshair_color = YELLOW
                    player.weapon_crosshair_size = 15
                elif player.weapon == 'Plasma Gun':
                    player.weapon_spread = 0.1
                    player.weapon_damage = 30
                    player.weapon_cooldown_time = 50
                    player.weapon_crosshair_color = CYAN
                    player.weapon_crosshair_size = 20          
    return True


def simple_pathfind(start, end, max_iterations=100):
    path = []
    current = Vector3(start.x, start.y, start.z)
    visited = set()
    iterations = 0
    while (current - end).length_squared() > TILE_SIZE**2 and iterations < max_iterations:
        if tuple(current) in visited:
            break
        visited.add(tuple(current))
        if random.random() < 0.1:  # 10% chance to make a random move
            random_direction = Vector3(random.uniform(-1, 1), 0, random.uniform(-1, 1)).normalize()
            next_pos = current + random_direction * TILE_SIZE
        else:
            direction = (end - current).normalize()
            next_pos = current + direction * TILE_SIZE
            if not is_collision(next_pos.x, next_pos.z):
                path.append(next_pos)
                current = next_pos
            else:
                for angle in [math.pi/4, -math.pi/4, math.pi/2, -math.pi/2]:
                    rotated_direction = Vector3(
                        direction.x * math.cos(angle) - direction.z * math.sin(angle),
                        0,
                        direction.x * math.sin(angle) + direction.z * math.cos(angle)
                    )
                    next_pos = current + rotated_direction * TILE_SIZE
                    if not is_collision(next_pos.x, next_pos.z):
                        path.append(next_pos)
                        current = next_pos
                        break
                else:
                    break
        iterations += 1
    return path


# Update move_monsters function to use pathfinding
def move_monsters():
    for monster in monsters:
        monster.update()
        
        # Skip if this monster is actually a player representation
        if hasattr(monster, 'is_player'):
            continue
        
        # In multiplayer, only server controls monster movement
        if mode != "singleplayer" and not is_server:
            continue
            
        # Find closest target between both players
        targets = [player]
        if mode != "singleplayer" and enemy_player:
            targets.append(enemy_player)
        
        nearest_target = min(targets, 
            key=lambda t: math.sqrt((t.pos.x - monster.pos.x)**2 + (t.pos.z - monster.pos.z)**2))
        
        dx = nearest_target.pos.x - monster.pos.x
        dz = nearest_target.pos.z - monster.pos.z
        distance = math.sqrt(dx*dx + dz*dz)
        
        if 20 < distance < 1000:
            path = simple_pathfind(monster.pos, nearest_target.pos)
            if path:
                next_pos = path[0]
                move_vector = (next_pos - monster.pos).normalize() * monster.speed
                monster.pos += move_vector
            
            # Only server controls monster shooting in multiplayer
            if mode == "singleplayer" or is_server:
                monster_angle = math.atan2(dz, dx)
                wall_distance = cast_single_ray(monster_angle - math.pi/2)
                
                if distance < wall_distance:
                    if random.random() < monster.shoot_probability:
                        monster.shoot(nearest_target)
                        print(f"Monster shot at {nearest_target.health} health target, Distance: {distance}")  # Debug print


def check_power_ups():
    global power_ups
    # Check collision for every power-up
    for power_up in power_ups[:]:  # Use slice copy to safely modify during iteration
        if (player.pos.x - power_up.pos.x) ** 2 + (player.pos.z - power_up.pos.z) ** 2 < 30 * 30:
            if power_up.power == 'health':
                player.health += power_up.healing
            elif power_up.power == 'ammo':
                player.ammo += power_up.ammo
            power_ups.remove(power_up)

def find_safe_spawn_position():
    while True:
        x = random.randint(1, MAP_SIZE - 2) * TILE_SIZE + TILE_SIZE // 2
        z = random.randint(1, MAP_SIZE - 2) * TILE_SIZE + TILE_SIZE // 2
        if not is_collision(x, z):
            return x, z

def spawn_power_ups(num_power_ups):
    # Only server spawns powerups
    if is_server:
        if len(power_ups) < num_power_ups and random.random() < 0.02:
            x, z = find_safe_spawn_position()
            y = 0
            power_type = random.choice(['health', 'ammo'])
            power_ups.append(PowerUp(x, y, z, power_type))

def spawn_monster(num_monsters):
    if len(monsters) < num_monsters and random.random() < 0.02:
        x, z = find_safe_spawn_position()
        y = 0
        monster_type = random.choice(['bird1', 'bird2', 'bird2'])
        monsters.append(Monster(x, y, z, monster_type))

def draw_hud():
    font = pygame.font.Font(None, 36)
    # Show player's health with appropriate label based on server/client status
    my_label = "Server Health: " if is_server else "Client Health: "
    my_health_text = font.render(f"{my_label}{player.health}", True, GREEN)
    
    # Show enemy player's health with appropriate label in RED
    if enemy_player:
        enemy_label = "Client Health: " if is_server else "Server Health: "
        enemy_health_text = font.render(f"{enemy_label}{enemy_player.health}", True, RED)
        screen.blit(enemy_health_text, (WIDTH - 300, 210))
    
    score_text = font.render(f"Score: {player.score}", True, WHITE)
    weapon_text = font.render(f"Weapon: {player.weapon}", True, WHITE)
    help_text = font.render("Press H for help", True, WHITE)
    ammo_text = font.render(f"Ammo: {player.ammo}", True, BLUE)
    
    screen.blit(my_health_text, (WIDTH - 300, 50))
    screen.blit(score_text, (WIDTH - 300, 90))
    screen.blit(weapon_text, (WIDTH - 300, 130))
    screen.blit(ammo_text, (WIDTH - 300, 170))
    screen.blit(help_text, (WIDTH - 300, 10))


def show_help():
    help_text = [
        "Game Controls:",
        "Arrow keys: Move",
        "Left/Right arrows: Turn",
        "Left mouse button: Change weapon",
        "Space: Shoot",
        "H: Show/hide help",
        "",
        "Objective:",
        "Defeat all monsters",
        "Collect power-ups",
        "Achieve high score",
        "",
        "Press any key to continue"
    ]

    screen.fill(BLACK)
    font = pygame.font.Font(None, 30)
    for i, line in enumerate(help_text):
        text = font.render(line, True, WHITE)
        screen.blit(text, (50, 50 + i * 30))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                waiting = False
             

def draw_map():
    screen.blit(create_minimap_bg(), (0, 0))
    for monster in monsters:
        # Use monster.colors for Player objects, otherwise use monster.color
        color = monster.colors if hasattr(monster, 'colors') else monster.color
        pygame.draw.circle(screen, color, 
                         (int(monster.pos.x * MINIMAP_SCALE), int(monster.pos.z * MINIMAP_SCALE)), 3)
    for power_up in power_ups:
        pygame.draw.circle(screen, power_up.color, 
                         (int(power_up.pos.x * MINIMAP_SCALE), int(power_up.pos.z * MINIMAP_SCALE)), 3)
    # Always draw the current player based on server/client status
    player_color = BLUE if player.is_server else YELLOW
    pygame.draw.circle(screen, player_color, (int(player.pos.x * MINIMAP_SCALE), int(player.pos.z * MINIMAP_SCALE)), 5)

    # Draw enemy player on minimap, also with consistent color
    if enemy_player:
        enemy_color = BLUE if enemy_player.is_server else YELLOW  # Consistent coloring
        pygame.draw.circle(screen, enemy_color, (int(enemy_player.pos.x * MINIMAP_SCALE), int(enemy_player.pos.z * MINIMAP_SCALE)), 5)

    # Change line color to be opposite of player_color
    line_color = YELLOW if player.is_server else BLUE  # Invert the color
    pygame.draw.line(screen, line_color,
                    (player.pos.x * MINIMAP_SCALE, player.pos.z * MINIMAP_SCALE),
                    ((player.pos.x + player.direction.x * 20) * MINIMAP_SCALE, (player.pos.z + player.direction.z * 20) * MINIMAP_SCALE),
                    2)

def cast_single_ray(angle):
    sin_a = -np.sin(angle)
    cos_a = np.cos(angle)
    
    for depth in range(1, MAX_DEPTH, 3):  # Step by 3 instead of 1
        target_x = player.pos.x + sin_a * depth
        target_z = player.pos.z + cos_a * depth
        col = int(target_x / TILE_SIZE)
        row = int(target_z / TILE_SIZE)
        
        if 0 <= row < MAP_SIZE and 0 <= col < MAP_SIZE:
            if map_data[row][col] == 1:
                return depth
    return MAX_DEPTH


def draw_walls():
    player_angle = player.angle - math.pi/2
    player_angle %= 2 * math.pi
    start_angle = (player_angle - HALF_FOV)
    
    start_angle %= 2 * math.pi
    #print(f"Player angle: {player.angle}, start_angle: {start_angle}")
    for ray in range(CASTED_RAYS):
        angle = start_angle + ray * STEP_ANGLE
        cos_a = math.cos(angle - player_angle)
        depth = cast_single_ray(angle)
        
        # Correct depth
        correct_depth = depth * cos_a
        
        # Calculate wall height
        wall_height = min(int((TILE_SIZE * HEIGHT) / correct_depth), HEIGHT * 2)
        
        # Calculate texture column
        hit_x = player.pos.x + depth * math.cos(angle)
        hit_y = player.pos.z + depth * math.sin(angle)
        texture_x = int((hit_x + hit_y) * TILE_SIZE % TILE_SIZE)
        
        # Draw textured wall slice
        wall_top = (HEIGHT - wall_height) // 2
        wall_bottom = wall_top + wall_height
        
        # Calculate texture column as before:
        texture_x = int((hit_x + hit_y) * TILE_SIZE % TILE_SIZE)
        key = (texture_x, wall_height)
        if key in wall_slice_cache:
            base_slice = wall_slice_cache[key]
        else:
            base_slice = pygame.transform.scale(
                WALL_TEXTURE.subsurface((texture_x, 0, 1, TILE_SIZE)),
                (WIDTH // CASTED_RAYS + 1, wall_height)
            )
            wall_slice_cache[key] = base_slice
        wall_slice = base_slice.copy()  # work on a copy so cache stays intact

        shade = max(0, min(255, 255 - int(depth * 255 / MAX_DEPTH)))
        wall_slice.fill((shade, shade, shade), special_flags=pygame.BLEND_MULT)
        screen.blit(wall_slice, (ray * WIDTH // CASTED_RAYS, wall_top))
        
    start_angle += STEP_ANGLE


def draw_floor():
    screen.blit(floor_surface, (0, HEIGHT//2))

def create_floor_surface():
    surface = pygame.Surface((WIDTH, HEIGHT//2))
    for y in range(surface.get_height()):
        brightness = 1 - (y / surface.get_height())
        color = (int(255 * brightness), int(255 * brightness), int(255 * brightness))
        pygame.draw.line(surface, color, (0, y), (WIDTH, y))
    return surface

floor_surface = create_floor_surface()  # Pre-render the floor gradient once.

def draw_weapon():
    weapon_image = pistol_image if player.weapon == 'Pistol' else machinegun_image
    weapon_rect = weapon_image.get_rect()
    
    # Position the weapon with its top at the bottom of the crosshair
    weapon_rect.midtop = (WIDTH // 2, HEIGHT // 2 + 40)  # Move down by 40 pixels
    
    screen.blit(weapon_image, weapon_rect)
    
    crosshair_color = player.weapon_crosshair_color
    crosshair_size = player.weapon_crosshair_size   

    if player.cooldown > 0:
        crosshair_color = RED
    pygame.draw.line(screen, crosshair_color, (H_WIDTH - crosshair_size, H_HEIGHT), (H_WIDTH + crosshair_size, H_HEIGHT))
    pygame.draw.line(screen, crosshair_color, (H_WIDTH, H_HEIGHT - crosshair_size), (H_WIDTH, H_HEIGHT + crosshair_size))

# Add this near the top of your file with other image loads
powerup_image = pygame.image.load('powerup.png')

def tint_image(image, color):
    tinted = image.copy()
    tinted.fill(color, special_flags=pygame.BLEND_RGB_MULT)
    return tinted

def draw_power_ups():
    if power_ups:
        for power_up in power_ups:
            dx = power_up.pos.x - player.pos.x
            dz = power_up.pos.z - player.pos.z
            distance = math.sqrt(dx*dx + dz*dz)
            distance = max(distance, .1)
            power_up_angle = math.atan2(dz, dx)
            power_up_angle %= 2 * math.pi
            relative_angle = power_up_angle - player.angle
            rel_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
            walldistance = cast_single_ray(power_up_angle - math.pi/2)

            if distance <= walldistance and abs(rel_angle) < HALF_FOV:
                proj_height = SCREEN_DIST / (distance * math.cos(rel_angle))
                power_up_height = int(proj_height * power_up.size / TILE_SIZE)
                power_up_height *= 5
                power_up_width = power_up_height
                
                screen_x = int(((rel_angle / HALF_FOV) * 0.5 + 0.5) * WIDTH - power_up_width / 2)
                screen_y = int((HEIGHT - power_up_height) / 2)
                
                # Use powerup.png with color tinting
                scaled_image = pygame.transform.scale(powerup_image, (power_up_width, power_up_height))
                
                # Apply power-up specific color tint
                colored_image = scaled_image.copy()
                colored_image.fill(power_up.color, special_flags=pygame.BLEND_RGBA_MULT)
                
                # Apply distance shading
                shade = max(0, min(255, 255 - int(distance * 255 / MAX_DEPTH)))
                colored_image.fill((shade, shade, shade), special_flags=pygame.BLEND_MULT)
                
                screen.blit(colored_image, (screen_x, screen_y))

def draw_monsters():
    """Draw monsters and enemy player."""
    # Draw enemy player if in multiplayer
    if enemy_player:
        dx = enemy_player.pos.x - player.pos.x
        dz = enemy_player.pos.z - player.pos.z
        distance = math.sqrt(dx*dx + dz*dz)
        distance = max(distance, .1)
        angle = math.atan2(dz, dx)
        angle %= 2 * math.pi
        relative_angle = angle - player.angle
        rel_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
        walldistance = cast_single_ray(angle - math.pi/2)

        if distance <= walldistance and abs(rel_angle) < HALF_FOV:
            proj_height = SCREEN_DIST / (distance * math.cos(rel_angle))
            player_height = int(proj_height * 300 / TILE_SIZE) * 5
            player_width = player_height
            
            screen_x = int(((rel_angle / HALF_FOV) * 0.5 + 0.5) * WIDTH - player_width / 2)
            screen_y = int((HEIGHT - player_height) / 2) + player_height // 4
            
            image_to_use = player_server_image if enemy_player.is_server else player_client_image
            scaled_image = pygame.transform.scale(image_to_use, (player_width, player_height))
            
            shade = max(0, min(255, 255 - int(distance * 255 / MAX_DEPTH)))
            scaled_image.fill((shade, shade, shade), special_flags=pygame.BLEND_MULT)
            screen.blit(scaled_image, (screen_x, screen_y))

    # Draw monsters without constant circles
    if monsters:
        for monster in monsters:
            dx = monster.pos.x - player.pos.x
            dz = monster.pos.z - player.pos.z
            distance = math.sqrt(dx*dx + dz*dz)
            distance = max(distance, .1)
            monster_angle = math.atan2(dz, dx)
            monster_angle %= 2 * math.pi
            relative_angle = monster_angle - player.angle
            rel_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
            walldistance = cast_single_ray(monster_angle - math.pi/2)

            if distance <= walldistance and abs(rel_angle) < HALF_FOV:
                proj_height = SCREEN_DIST / (distance * math.cos(rel_angle))
                monster_height = int(proj_height * monster.size / TILE_SIZE)
                monster_height *= 5
                monster_width = monster_height
                
                screen_x = int(((rel_angle / HALF_FOV) * 0.5 + 0.5) * WIDTH - monster_width / 2)
                screen_y = int((HEIGHT - monster_height) / 2) + monster_height // 4
                
                scaled_image = pygame.transform.scale(monster.image, (monster_width, monster_height))
                
                shade = max(0, min(255, 255 - int(distance * 255 / MAX_DEPTH)))
                scaled_image.fill((shade, shade, shade), special_flags=pygame.BLEND_MULT)
                
                # Store for hit detection
                monster.screen_x = screen_x
                monster.screen_y = screen_y
                monster.screen_width = monster_width
                monster.screen_height = monster_height
                
                screen.blit(scaled_image, (screen_x, screen_y))


def create_minimap_bg():
    minimap_width = int(MAP_SIZE * TILE_SIZE * MINIMAP_SCALE)
    minimap_height = int(MAP_SIZE * TILE_SIZE * MINIMAP_SCALE)
    surface = pygame.Surface((minimap_width, minimap_height))
    surface.fill(BLACK)
    for row in range(MAP_SIZE):
        for col in range(MAP_SIZE):
            if map_data[row][col] == 1:
                square = (col * TILE_SIZE * MINIMAP_SCALE,
                          row * TILE_SIZE * MINIMAP_SCALE,
                          TILE_SIZE * MINIMAP_SCALE,
                          TILE_SIZE * MINIMAP_SCALE)
                pygame.draw.rect(surface, WHITE, square)
    return surface


def server_listen():
    """Improved server listening with on-screen status"""
    global client_socket, server_socket
    if server_socket:
        server_socket.close()
    if client_socket:
        client_socket.close()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    
    # Set the socket to non-blocking mode
    server_socket.setblocking(False)
    
    timeout = 360
    start_time = time.time()
    
    # Setup fonts for display
    font = pygame.font.Font(None, 74)
    small_font = pygame.font.Font(None, 36)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        
        # Clear screen and show waiting status
        screen.fill(BLACK)
        title = font.render("Hosting Game", True, RED)
        host_info = small_font.render(f"Hosting on {HOST}:{PORT}", True, GREEN)
        waiting_text = small_font.render("Waiting for player to join...", True, YELLOW)
        escape_text = small_font.render("Press ESC to cancel", True, WHITE)
        
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 100))
        screen.blit(host_info, (WIDTH//2 - host_info.get_width()//2, 250))
        screen.blit(waiting_text, (WIDTH//2 - waiting_text.get_width()//2, 300))
        screen.blit(escape_text, (WIDTH//2 - escape_text.get_width()//2, 400))
        
        pygame.display.flip()
        
        try:
            client_socket, addr = server_socket.accept()
            print(f"Connected by {addr}")
            return True
        except BlockingIOError:
            pass
        
        if time.time() - start_time > timeout:
            print("Timeout: No connection after 6 minutes")
            return False
        
        pygame.time.delay(100)


def client_connect(max_attempts=9, retry_delay=15):
    """Improved client connection from Tank game"""
    global client_socket
    for attempt in range(max_attempts):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"Attempting to connect to server at {HOST}:{PORT} (attempt {attempt + 1}/{max_attempts})...")
            client_socket.connect((HOST, PORT))
            print(f"Connected to server at {HOST}:{PORT}")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                time.sleep(retry_delay)
    return False


def handle_network_disconnect():
    global client_socket, server_socket
    print("Network disconnection detected")
    if client_socket:
        client_socket.close()
    if server_socket:
        server_socket.close()
    client_socket = None
    server_socket = None


def network_send(data):
    """Improved network send function from Tank game"""
    try:
        if client_socket:
            serialized_data = pickle.dumps(data)
            data_size = len(serialized_data)
            client_socket.settimeout(.1)  # .1 seconds timeout
            client_socket.send(data_size.to_bytes(4, byteorder='big'))
            client_socket.send(serialized_data)
    except socket.timeout:
        print("Send operation timed out")
        return False    
    except (ConnectionResetError, BrokenPipeError):
        print("Connection lost. Attempting to reconnect...")
        handle_network_disconnect()
    except Exception as e:
        print(f"Error sending data: {e}")
        handle_network_disconnect()


def network_receive():
    """Improved network receive function from Tank game"""
    try:
        if client_socket:
            client_socket.settimeout(1)
            size_bytes = client_socket.recv(4)
            if not size_bytes:
                raise RuntimeError("Connection closed by remote host")
            size = int.from_bytes(size_bytes, byteorder='big')
            data = b""
            while len(data) < size:
                chunk = client_socket.recv(size - len(data))
                if not chunk:
                    raise RuntimeError("Connection closed by remote host")
                data += chunk
            return pickle.loads(data)
    except socket.timeout:
        print("Socket timeout occurred")
        return None
    except (ConnectionResetError, BrokenPipeError):
        print("Connection lost. Attempting to reconnect...")
        handle_network_disconnect()
        return None
    except Exception as e:
        print(f"Error receiving data: {e}")
        handle_network_disconnect()
        return None


def show_game_over(is_winner):
    """Display game over screen."""
    screen.fill(BLACK)
    font = pygame.font.Font(None, 74)
    if is_winner == "win":
        text = font.render('You Win!', True, GREEN)
    else:
        text = font.render('Game Over - You Lost!', True, RED)
    
    score_text = font.render(f'Final Score: {player.score}', True, WHITE)
    
    screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - 50))
    screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, HEIGHT//2 + 50))
    pygame.display.flip()
    
    # Wait a few seconds before closing
    pygame.time.wait(6000)


def choose_game_mode():
    """Display menu and return selected game mode."""
    screen.fill(BLACK)
    font = pygame.font.Font(None, 74)
    small_font = pygame.font.Font(None, 36)  # Added for connection info
    
    menu_items = [
        ("Single Player", "singleplayer"),
        ("Host Game", "hostgame"),
        ("Join Game", "joingame"),
        ("Quit", "quit")
    ]
    
    selected = 0
    
    while True:
        screen.fill(BLACK)
        
        # Draw title
        title = font.render("DOOM-like Game", True, RED)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 100))
        
        # Draw menu items
        for i, (text, _) in enumerate(menu_items):
            color = YELLOW if i == selected else WHITE
            item = font.render(text, True, color)
            screen.blit(item, (WIDTH//2 - item.get_width()//2, 300 + i * 100))
            
            # Add connection info when "Host Game" is selected
            if i == 1 and selected == 1:  # When "Host Game" is selected
                host_info = small_font.render(f"Hosting on {HOST}", True, GREEN)
                port_info = small_font.render(f"Port: {PORT}", True, GREEN)
                screen.blit(host_info, (WIDTH//2 - host_info.get_width()//2, 300 + i * 100 + 50))
                screen.blit(port_info, (WIDTH//2 - port_info.get_width()//2, 300 + i * 100 + 80))
        
        pygame.display.flip()
        
        # Handle input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(menu_items)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(menu_items)
                elif event.key == pygame.K_RETURN:
                    return menu_items[selected][1]


def pygame_input(prompt, default_text=""):
    """Get text input from user using pygame."""
    screen.fill(BLACK)
    font = pygame.font.Font(None, 36)
    text = default_text
    input_active = True
    
    while input_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return text
                elif event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                else:
                    text += event.unicode
        
        screen.fill(BLACK)
        # Draw prompt
        prompt_surface = font.render(prompt, True, WHITE)
        screen.blit(prompt_surface, (10, HEIGHT//2 - 50))
        # Draw input text
        txt_surface = font.render(text, True, WHITE)
        screen.blit(txt_surface, (10, HEIGHT//2))
        pygame.display.flip()
    return text


def reset_game_multiplayer(max_retries=9, retry_delay=2):
    """Improved multiplayer initialization from Tank game"""
    global player, enemy_player, monsters, power_ups, map_data, minimap_bg, running

    for attempt in range(max_retries):
        try:
            # Reset game variables
            monsters = []
            power_ups = []

            if is_server:
                # Server generates the initial game state
                map_data = generate_complex_maze(48, 48)
                initial_data = {
                    "map_data": map_data,
                    "initial_data": True
                }
                print("Sending initial game data to client")
                network_send(initial_data)
                
                # Wait for client acknowledgment
                ack_received = False
                for i in range(5):
                    ack = network_receive()
                    if ack and "initial_data_ack" in ack:
                        ack_received = True
                        print("Client acknowledgment received")
                        break
                    print("Waiting for client acknowledgment try:", i)
                    time.sleep(1)
                
                if not ack_received:
                    raise Exception("Failed to receive client acknowledgment")
            else:
                # Client receives the initial game state
                received_data = None
                for i in range(5):
                    received_data = network_receive()
                    if received_data and "map_data" in received_data:
                        print("Received initial game data from server")
                        map_data = received_data["map_data"]
                        network_send({"initial_data_ack": True})
                        print("Acknowledgment sent to server")
                        break
                    print("Waiting for initial game data try:", i)
                    time.sleep(1)
                
                if not received_data:
                    raise Exception("Failed to receive initial game data")

            # Initialize players with proper positions
            x, z = find_safe_spawn_position()
            player = Player((x, 0, z), (0, 0, -1), is_server)  # Pass is_server flag
            
            # Exchange player positions and create enemy_player
            network_send({"player": player})
            received_data = network_receive()
            if received_data and "player" in received_data:
                # Create enemy_player with opposite is_server value
                enemy_data = received_data["player"]
                enemy_player = Player(enemy_data.pos, enemy_data.direction, not is_server)  # Important: opposite is_server value
                print(f"Enemy player initialized at position: {enemy_player.pos}")
                print(f"Enemy player is_server value: {enemy_player.is_server}")
            else:
                raise Exception("Failed to receive enemy player data")

            return True

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Game initialization failed.")
                return False

    return False


def main():
    global player, monsters, map_data, power_ups, is_server, client_socket, HOST, minimap_bg, mode, running, enemy_player, enemy_damage_value

    # Choose game mode
    mode = choose_game_mode()
    if mode == "quit":
        return

    # Initialize based on mode
    if mode == "singleplayer":
        map_data = generate_complex_maze(48, 48)
        minimap_bg = create_minimap_bg()
        
        # Initialize game state
        y = 0
        x, z = find_safe_spawn_position()
        power_ups = [PowerUp(x, y, z, 'health')]
        x, z = find_safe_spawn_position()
        monsters = [Monster(x, y, z, 'bird1')]
        x, z = find_safe_spawn_position()
        player = Player((x, y, z), (0, 0, -1), is_server)
    
    # Handle multiplayer setup
    elif mode == "hostgame":
        is_server = True
        print(f"Hosting game on {HOST}")
        if not server_listen():
            print("Failed to start server")
            return
        reset_game_multiplayer()
        
    elif mode == "joingame":
        is_server = False
        host_ip = pygame_input("Enter host IP:", DEFAULT_HOST)
        if host_ip == "quit":
            return
        HOST = host_ip
        print(f"Joining game at {HOST}")
        if not client_connect():
            print("Failed to connect to server")
            return
        reset_game_multiplayer()

    clock = pygame.time.Clock()
    running = True

    while running:
        dt = clock.tick(90) / 1000.0
        screen.fill(BLACK)
        handle_movement(dt)
        
        if mode == "singleplayer":
            move_monsters()
            spawn_monster(MAX_MONSTERS)
            spawn_power_ups(MAX_POWERUPS)
        else:  # Multiplayer mode
            try:
                # Process power-up collisions
                check_power_ups()
                
                # Build current state with only essential information
                current_state = {
                    'player': {
                        'pos': (player.pos.x, player.pos.y, player.pos.z),
                        'direction': (player.direction.x, player.direction.y, player.direction.z),
                        'health': player.health
                    },
                    'power_ups': power_ups,  # Both need to sync these
                    'monsters': monsters,     # Both need to sync these
                    'damage': enemy_damage_value
                }

                # Send current state
                network_send(current_state)

                # Reset damage after sending so we don't apply it repeatedly
                enemy_damage_value = 0

                # Receive and process enemy state
                received_state = network_receive()
                if received_state:
                    if 'player' in received_state:
                        # Update enemy player from the received player state
                        player_state = received_state['player']
                        enemy_player.pos = Vector3(player_state['pos'][0], player_state['pos'][1], player_state['pos'][2])
                        enemy_player.direction = Vector3(player_state['direction'][0], player_state['direction'][1], player_state['direction'][2])
                        enemy_player.health = player_state['health']

                    if enemy_player.health <= 0:
                        #your score goes up
                        player.score += 1000
                        show_game_over("win")
                        running = False

                    if player.health <= 0:
                        show_game_over("lose")
                        running = False

                    if 'power_ups' in received_state and random.random() < 0.1:
                        power_ups = received_state['power_ups']
                    if 'monsters' in received_state and random.random() < 0.1:
                        monsters = received_state['monsters']

                    # Process damage received for the player
                    if 'damage' in received_state and received_state['damage'] > 0:
                        player.health -= received_state['damage']
                        hit_sound.play()
                        print(f"You took damage! Health now: {player.health}")
                        received_state['damage'] = 0
                        
                # Server also handles monsters
                if is_server:
                    spawn_monster(MAX_MONSTERS)
                    spawn_power_ups(MAX_POWERUPS)
                    move_monsters()

            except Exception as e:
                print(f"Network error in main loop: {e}")
                handle_network_disconnect()

        # For single player, check powerups normally (client will update locally too)
        if mode == "singleplayer":
            check_power_ups()
            if player.health <= 0:
                show_game_over("lose")
                running = False

        draw_floor()
        draw_walls()
        draw_monsters()
        draw_power_ups()
        draw_map()
        draw_weapon()
        draw_hud()
        
        handle_shooting()
        player.update()
        
        pygame.display.flip()

    # Cleanup
    if client_socket:
        client_socket.close()
    if server_socket:
        server_socket.close()

# Modified game entry point
if __name__ == "__main__":
    # Map settings
    MAP_SIZE = 24
    power_ups = []
    monsters = []
    map_data = None
    
    # Move the mode-specific initialization into main()
    print("Initial setup complete")
    main()