#JMR TANK battle game June 27, 2024
# July 11th, working two player mode
# July 13th, scanning for for local IP addresses
# July 17th added sound effects
# July 20th fixed sync issues with acknowledging initial data, added color to Tank and PowerUp classes added miniguns and rockets
#pip3 install python-nmap
import pygame
import math
import random
from pygame.math import Vector3
import socket
import pickle
import threading
import time
import subprocess
import nmap
import ipaddress
import re
import queue

# Initialize Pygame
pygame.init()

WIDTH, HEIGHT = 1200, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("JMR's 3D Tank Battle")

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255) 
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
PINK = (255, 192, 203)
# what other colors work well for games
PURPLE = (128, 0, 128)
BROWN = (165, 42, 42)
GREY = (128, 128, 128)

# Game constants, enemy constants only used in single player mode
TERRAIN_SIZE = 1000
NUM_OBSTACLES = 30
NUM_ENEMY_TANKS = 3
PLAYER_TANK_STRENGTH = 2000
BULLET_SPEED = 5
CANNON_SPEED = 5
ENEMY_TANK_SPEED = 1  # Adjust this value to control enemy tank speed
PLAYER_TANK_SPEED = 2  # Adjust this value to control player tank speed
ENEMY_TANK_STRENGTH = 2000  # Adjust this value to control enemy tank strength
PLAYER_COOL_DOWN = 30
ENEMY_COOL_DOWN = 30

try:
    shoot_sound = pygame.mixer.Sound('mixkit-arcade-game-explosion-2759.wav')
    impact_sound = pygame.mixer.Sound('mixkit-explosive-impact-from-afar-2758.wav')
except:
    shoot_sound = None
    impact_sound    = None

# New global variables for networking
is_server = False
client_socket = None
server_socket = None

#powerstuff
power_ups = []

# Execute shell command to get IP addresses                                   
ip_command_output = subprocess.check_output('ifconfig', shell=True).decode()  
                                                                                
# Extract the first non-loopback IP address 
inet_patterns = re.compile(r'inet (\d+\.\d+\.\d+\.\d+)')                                 
for line in ip_command_output.split('\n'):                                    
      if 'inet ' in line and '127.0.0.1' not in line:                           
          ip_address = re.findall(inet_patterns, line)[0]                                 
          break                                                                                                                                           
print(f'Real IP Address: {ip_address}')   

DEFAULT_HOST = ip_address

HOST = ip_address
PORT = 65432        # Port to use (make sure it's open in your firewall)

class Tank:
    def __init__(self, position, direction, is_player=False):
        self.position = Vector3(position)
        self.direction = Vector3(direction).normalize()
        self.is_player = is_player
        if is_player:
            self.health = PLAYER_TANK_STRENGTH
            self.cool_time = PLAYER_COOL_DOWN
            self.speed = PLAYER_TANK_SPEED
            self.color = BLUE if mode == "single" else BLUE if is_server else YELLOW
        else:
            self.health = ENEMY_TANK_STRENGTH
            self.cool_time = ENEMY_COOL_DOWN
            self.speed = ENEMY_TANK_SPEED
            self.color = RED
        
        self.weapon_type = "cannon"
        self.cooldown = 0
        self.score = 0
        self.barrel_length = 15
        self.barrel_angle = 0
        self.adjusted_vertical_angle = 0    # New attribute to store the adjusted vertical angle for shooting and rendering

    def __getstate__(self):
        # This method is called when pickling the object
        state = self.__dict__.copy()
        # Convert Vector3 objects to tuples for pickling
        state['position'] = tuple(self.position)
        state['direction'] = tuple(self.direction)
        return state
    
    def __setstate__(self, state):
        # This method is called when unpickling the object
        self.__dict__.update(state)
        # Convert tuples back to Vector3 objects
        self.position = Vector3(state['position'])
        self.direction = Vector3(state['direction'])

    def move(self, forward, obstacles, enemy_tanks=[]):
        speed = self.speed
        # Only move in the direction the tank is facing
        movement = self.direction * (speed if forward else -speed)
        new_position = self.position + movement
        if 0 <= new_position.x < TERRAIN_SIZE and 0 <= new_position.z < TERRAIN_SIZE:
            if not check_collision(new_position, obstacles, [tank for tank in enemy_tanks if tank != self]):
                self.position = new_position
            
    def rotate(self, angle):
        # Convert angle to radians
        angle_rad = math.radians(angle)
        # Rotate the direction vector around the y-axis
        old_x = self.direction.x
        old_z = self.direction.z
        # Apply rotation
        self.direction.x = old_x * math.cos(angle_rad) - old_z * math.sin(angle_rad)
        self.direction.z = old_x * math.sin(angle_rad) + old_z * math.cos(angle_rad)
        # Normalize the direction vector
        self.direction = self.direction.normalize()

    def fire(self):
       if self.cooldown > 0:
           return None
       
       self.cooldown = self.cool_time
       
       # Calculate the bullet direction in world space
       up_vector = Vector3(0, 1, 0)
       right_vector = self.direction.cross(up_vector).normalize()
       
       # Rotate around the right vector (which is perpendicular to both up and forward)
       bullet_direction = self.direction.rotate(self.barrel_angle, right_vector)
       
       barrel_angle_rad = math.radians(self.barrel_angle)
       barrel_length = 15
       vertical_offset = barrel_length * math.sin(barrel_angle_rad) + 5 # Approximate height of the turret from the tank's base
   
       start = self.position + self.direction * 15 + Vector3(0, vertical_offset, 0)
       end = start + bullet_direction * 1000
       if self.is_player and shoot_sound:
            shoot_sound.play()
       return Bullet(start, end, self.is_player, self.weapon_type)

    def barrel_angle_adjust(self, angle):
        # Only adjust the vertical angle of the barrel
        self.barrel_angle = max(-35, min(35, self.barrel_angle + angle))
        self.adjusted_vertical_angle = self.barrel_angle  

    def update(self):
        if self.cooldown > 0:
            self.cooldown -= 1


class Bullet:
    def __init__(self, start, end, is_player_bullet, type = "cannon"):
        self.start = Vector3(start)
        self.end = Vector3(end)
        self.position = Vector3(start)
        self.direction = (end - start).normalize()
        self.distance_traveled = 0
        self.is_player_bullet = is_player_bullet  # New attribute to distinguish player bullets
        self.type = type
        self.damage = 20 if self.type == "minigun" else 1000 if self.type == "rocket" else 100
        self.speed =  2 * CANNON_SPEED if self == "minigun" else .5 * CANNON_SPEED if self.type == "rocket" else 5
        self.color = PURPLE if self.type == "minigun" else PINK if self.type == "rocket" else WHITE

    def __getstate__(self):
        state = self.__dict__.copy()
        state['start'] = tuple(self.start)
        state['end'] = tuple(self.end)
        state['position'] = tuple(self.position)
        state['direction'] = tuple(self.direction)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.start = Vector3(state['start'])
        self.end = Vector3(state['end'])
        self.position = Vector3(state['position'])
        self.direction = Vector3(state['direction'])

    def update(self):
        self.position += self.direction * self.speed
        self.distance_traveled += self.speed
        return self.distance_traveled < 500 if self.type == "minigun" else self.distance_traveled < 1500 if self.type == "rocket" else self.distance_traveled < 1000


class Explosion:
    def __init__(self, position, tank, radius=10):
        self.position = position
        self.radius = 0
        self.max_radius = radius #add so when tank killed we do much bigger explosion
        self.growth_rate = 1
        self.tank = tank
        if impact_sound:
            impact_sound.play()

    def update(self):
        global enemy_tanks
        #remove tank from enemy_tanks list
        
        self.radius += self.growth_rate
        if self.radius >= self.max_radius and self.tank.health <= 0:
            if self.tank in enemy_tanks:
                enemy_tanks.remove(self.tank)
        return self.radius < self.max_radius


class PowerUp:
    def __init__(self, position, type):
        self.position = position
        self.type = type  # "health", "cooldown", or "speed", "minigun", "cannon", "rocket"
        self.creation_time = time.time()
        self.duration = 30  # Power-up lasts for 60 seconds
        Health = random.randint(0, 500)
        Cooldown = random.randint(0, 10)
        Speed = random.uniform(0, 0.5)
        self.value = Health if type == "health" else Cooldown if type == "cooldown" else Speed 
        self.color = ORANGE if type == "health" else MAGENTA if type == "cooldown" else CYAN if type == "speed" else PINK if type == "rocket" else PURPLE if type == "minigun" else WHITE 

    def __getstate__(self):
        state = self.__dict__.copy()
        state['position'] = tuple(self.position)
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.position = Vector3(state['position'])                          

    def is_expired(self):
        return time.time() - self.creation_time > self.duration


def show_help_reset(screen):
    global ENEMY_TANK_STRENGTH, NUM_ENEMY_TANKS, ENEMY_TANK_SPEED, PLAYER_COOL_DOWN, NUM_OBSTACLES, PLAYER_TANK_STRENGTH, PLAYER_TANK_SPEED, BULLET_SPEED
    help_text = [
        "Controls:",
        "Arrow Up/Down: Move forward/backward",
        "Arrow Left/Right: Rotate tank",
        "Q/Z: Raise/Lower barrel",
        "Spacebar: Fire",
        "H: Toggle help screen",
        "",
        "Objective:",
        "Destroy all enemy tanks",
        "Avoid obstacles and enemy fire",
        "",
        "Set difficulty level:",
        "(W) Woke, (M) Medium, (B) Based",
        "", 
        "Hit Return to continue"
    ]
    
    font = pygame.font.Font(None, 30)
    for i, line in enumerate(help_text):
        text = font.render(line, True, WHITE)
        screen.blit(text, (WIDTH/2 -250, 150 + i * 30))
    pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    NUM_OBSTACLES = 30
                    ENEMY_TANK_STRENGTH = 1000
                    NUM_ENEMY_TANKS = 3
                    ENEMY_TANK_SPEED = 0.5
                    PLAYER_TANK_SPEED = 3
                    PLAYER_COOL_DOWN = 10
                    PLAYER_TANK_STRENGTH = 5000
                    BULLET_SPEED = 5

                elif event.key == pygame.K_m: 
                    NUM_OBSTACLES = 20
                    ENEMY_TANK_STRENGTH = 2000
                    NUM_ENEMY_TANKS = 5
                    ENEMY_TANK_SPEED = 1
                    PLAYER_TANK_SPEED = 2
                    PLAYER_COOL_DOWN = 30
                    PLAYER_TANK_STRENGTH = 2000
                    BULLET_SPEED = 5
                    
                elif event.key == pygame.K_b: 
                    NUM_OBSTACLES = 10
                    ENEMY_TANK_STRENGTH = 5000
                    NUM_ENEMY_TANKS = 10
                    ENEMY_TANK_SPEED = 2
                    PLAYER_TANK_SPEED = 1
                    PLAYER_COOL_DOWN = 60
                    PLAYER_TANK_STRENGTH = 500
                    BULLET_SPEED = 5

                print (f"P_T_S: {PLAYER_TANK_STRENGTH}, N_O: {NUM_OBSTACLES}, N_E_T: {NUM_ENEMY_TANKS}, P_T_SP: {PLAYER_TANK_SPEED}, C_D: {PLAYER_COOL_DOWN}")
                print (f"ENEMY_TANK_STRENGTH: {ENEMY_TANK_STRENGTH}, NUM_ENEMY_TANKS: {NUM_ENEMY_TANKS}, ENEMY_TANK_SPEED: {ENEMY_TANK_SPEED}") 
                return True, False


def pygame_input(message, default, port=65432):
    font = pygame.font.Font(None, 36)
    input_text = default
    input_rect = pygame.Rect(WIDTH/2 - 150, 350, 400, 40)
    color = WHITE
    host_ip = default
    servers = []
    progress_queue = queue.Queue()
    stop_event = threading.Event()
    scan_thread = threading.Thread(target=scan_network, args=(host_ip, port, servers, progress_queue, stop_event))
    scan_thread.start()

    text_surface = font.render(message, True, WHITE)
    screen.blit(text_surface, (WIDTH/2 - 150, 300))
    pygame.display.flip()

    selected_index = -1
    scanning = True
    progress = 0
    scanned_hosts = 0
    total_hosts = 0
    elapsed_time = 0
    eta = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()
                scan_thread.join()
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    print(f"Input: {input_text}")
                    stop_event.set()
                    scan_thread.join()
                    return input_text
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif event.key == pygame.K_UP and servers:
                    selected_index = (selected_index - 1) % len(servers)
                    input_text = servers[selected_index]
                    color = GREEN
                elif event.key == pygame.K_DOWN and servers:
                    selected_index = (selected_index + 1) % len(servers)
                    input_text = servers[selected_index]
                    color = GREEN
                else:
                    input_text += event.unicode

        # Check for updates from the scanning thread
        while not progress_queue.empty():
            update_type, data = progress_queue.get()
            if update_type == 'progress':
                progress, scanned_hosts, total_hosts, elapsed_time, eta = data
            elif update_type == 'server':
                if data not in servers:
                    servers.append(data)
            elif update_type == 'done':
                scanning = False

        screen.fill(BLACK)
        screen.blit(text_surface, (WIDTH/2 - 150, 300))
        pygame.draw.rect(screen, WHITE, input_rect, 2)
        text_surface = font.render(input_text, True, color)
        screen.blit(text_surface, (input_rect.x + 5, input_rect.y + 5))

        for i, server in enumerate(servers):
            color = GREEN if i == selected_index else WHITE
            server_surface = font.render(server, True, color)
            screen.blit(server_surface, (WIDTH/2 - 150, 400 + i * 30))

        if scanning:
            progress_text = f"Scanning: {progress:.1f}% | {scanned_hosts}/{total_hosts} hosts"
            time_text = f"Elapsed: {elapsed_time:.1f}s | ETA: {eta:.1f}s"
            progress_surface = font.render(progress_text, True, WHITE)
            time_surface = font.render(time_text, True, WHITE)
            screen.blit(progress_surface, (WIDTH/2 - 150, 400 + len(servers) * 30))
            screen.blit(time_surface, (WIDTH/2 - 150, 430 + len(servers) * 30))
        elif len(servers) == 0:
            no_server_surface = font.render("No servers found", True, WHITE)
            screen.blit(no_server_surface, (WIDTH/2 - 150, 400))

        pygame.display.flip()


def create_terrain():
    terrain = []
    for x in range(0, TERRAIN_SIZE, 50):
        for z in range(0, TERRAIN_SIZE, 50):
            terrain.append((Vector3(x, 0, z), Vector3(x+50, 0, z)))
            terrain.append((Vector3(x, 0, z), Vector3(x, 0, z+50)))
    return terrain


def create_obstacles():
    obstacles = []
    for _ in range(NUM_OBSTACLES):
        size = random.uniform(10, 30)
        shape = random.choice(["cube", "pyramid"]) # can Add "sphere" to the list
        position = Vector3(random.uniform(0, TERRAIN_SIZE), 0, random.uniform(0, TERRAIN_SIZE))
        obstacles.append((position, size, shape))
    return obstacles


def check_collision(position, obstacles, enemy_tanks=[]):
    for obstacle in obstacles:
        if (position - obstacle[0]).length() < obstacle[1] + 1:
            return True
    if enemy_tanks:
        #adjuse for height of bullet
        
        for tank in enemy_tanks:
            if (position - tank.position).length() < 20:
                return True
    return False


def project_3d_to_2d(point, camera_pos, camera_dir, fov=60):
    relative_pos = point - camera_pos
    forward = camera_dir.normalize()
    right = Vector3(forward.z, 0, -forward.x).normalize()
    up = Vector3(0, 1, 0)
    x = relative_pos.dot(right)
    y = relative_pos.dot(up)
    z = relative_pos.dot(forward)
    if z <= 0:
        return None
    f = WIDTH / (2 * math.tan(math.radians(fov / 2)))
    sx = WIDTH / 2 + x * f / z
    sy = HEIGHT / 2 - y * f / z
    return (sx, sy)


def render_3d_line(screen, start, end, camera_pos, camera_dir, color, fov=60):
    start_2d = project_3d_to_2d(start, camera_pos, camera_dir, fov)
    end_2d = project_3d_to_2d(end, camera_pos, camera_dir, fov)
    if start_2d and end_2d:
        pygame.draw.line(screen, color, start_2d, end_2d)


def render_3d_cube(screen, center, size, camera_pos, camera_dir, color, fov=60, direction=None):
    if isinstance(size, (int, float)):
        size = Vector3(size, size, size)
    half_size = size / 2
    vertices = [
        Vector3(center.x - half_size.x, center.y - half_size.y, center.z - half_size.z),
        Vector3(center.x + half_size.x, center.y - half_size.y, center.z - half_size.z),
        Vector3(center.x + half_size.x, center.y + half_size.y, center.z - half_size.z),
        Vector3(center.x - half_size.x, center.y + half_size.y, center.z - half_size.z),
        Vector3(center.x - half_size.x, center.y - half_size.y, center.z + half_size.z),
        Vector3(center.x + half_size.x, center.y - half_size.y, center.z + half_size.z),
        Vector3(center.x + half_size.x, center.y + half_size.y, center.z + half_size.z),
        Vector3(center.x - half_size.x, center.y + half_size.y, center.z + half_size.z),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for edge in edges:
        # Rotate the cube to face the direction
        if direction:
            start = vertices[edge[0]] - center
            end = vertices[edge[1]] - center
            # Check if the direction vector is not close to zero
            if direction.length_squared() > 1e-6:
                axis = direction.cross(Vector3(0, 0, 1))
                if axis.length_squared() > 1e-6:
                    angle = direction.angle_to(Vector3(0, 0, 1))
                    start = start.rotate(-angle, axis)
                    end = end.rotate(-angle, axis)
            start += center
            end += center
        else:
            start = vertices[edge[0]]
            end = vertices[edge[1]]
        render_3d_line(screen, start, end, camera_pos, camera_dir, color, fov)


def render_3d_pyramid(screen, base_center, size, camera_pos, camera_dir, color, fov=60):
    half_size = size / 2
    base_vertices = [
        Vector3(base_center.x - half_size, base_center.y, base_center.z - half_size),
        Vector3(base_center.x + half_size, base_center.y, base_center.z - half_size),
        Vector3(base_center.x + half_size, base_center.y, base_center.z + half_size),
        Vector3(base_center.x - half_size, base_center.y, base_center.z + half_size),
    ]
    apex = Vector3(base_center.x, base_center.y + size, base_center.z)
    for i in range(4):
        render_3d_line(screen, base_vertices[i], base_vertices[(i+1)%4], camera_pos, camera_dir, color, fov)
        render_3d_line(screen, base_vertices[i], apex, camera_pos, camera_dir, color, fov)


def render_3d_cylinder(screen, start, end, radius, camera_pos, camera_dir, color, fov=60):
   pass


def render_3d_obstacle(screen, obstacle, camera_pos, camera_dir, fov=60):
    position, size, shape = obstacle
    if shape == "cube":
        render_3d_cube(screen, position, size, camera_pos, camera_dir, GREEN, fov)
    elif shape == "sphere":
        render_3d_cube(screen, position, size, camera_pos, camera_dir, GREEN, fov)
        #render_3d_sphere(screen, position, size, camera_pos, camera_dir, GREEN, fov)
    elif shape == "pyramid":
        render_3d_pyramid(screen, position, size, camera_pos, camera_dir, GREEN, fov)


def render_explosion(screen, explosion, camera_pos, camera_dir):
    center = project_3d_to_2d(explosion.position, camera_pos, camera_dir)
    if center:
        #if explosion.radious is even then orange else red
        if explosion.radius % 2 == 0:
            pygame.draw.circle(screen, (255, 165, 0), center, int(explosion.radius * 200 / explosion.tank.position.distance_to(camera_pos)))
        else:
            pygame.draw.circle(screen, RED, center, int(explosion.radius * 200 / explosion.position.distance_to(camera_pos)))


def render_3d_tank(screen, tank, camera_pos, camera_dir, color, fov=60):
    body_size = Vector3(8, 5, 10)  # Width, Height, Length
    body_pos = tank.position + Vector3(0, body_size.y / 2, 0)
    render_3d_cube(screen, body_pos, body_size, camera_pos, camera_dir, color, fov, tank.direction)
    # Turret
    turret_pos = tank.position + Vector3(0, body_size.y + 1.5, 0)
    turret_size = Vector3(6, 3, 6)
    #make turrent the color of the weapon system
    colort = PINK if tank.weapon_type == "rocket" else PURPLE if tank.weapon_type == "minigun" else WHITE
    render_3d_cube(screen, turret_pos, turret_size, camera_pos, camera_dir, colort, fov, tank.direction)
    # Main gun (apply vertical angle to the tank's direction)
    up_vector = Vector3(0, 1, 0)
    right_vector = tank.direction.cross(up_vector).normalize()
    barrel_direction = tank.direction.rotate(tank.barrel_angle, right_vector)
    barrel_start = turret_pos + Vector3(0, 0, 0)
    barrel_end = barrel_start + barrel_direction * tank.barrel_length
    #make barrel a line not cylinder
    render_3d_line(screen, barrel_start, barrel_end, camera_pos, camera_dir, colort, fov)

    # Treads
    tread_length, tread_width, tread_height = 15, 3, 2
    tread_offset = 4  # Half of the tank's width
    left_tread = tank.position + tank.direction.cross(Vector3(0, 1, 0)) * tread_offset
    right_tread = tank.position + tank.direction.cross(Vector3(0, -1, 0)) * tread_offset
    tread_size = Vector3(tread_width, tread_height, tread_length)
    render_3d_cube(screen, left_tread, tread_size, camera_pos, camera_dir, color, fov, tank.direction)
    render_3d_cube(screen, right_tread, tread_size, camera_pos, camera_dir, color, fov, tank.direction)


def render_hud(screen, player_tank):
    font = pygame.font.Font(None, 30)
    color = PURPLE if player_tank.weapon_type == "minigun" else PINK if player_tank.weapon_type == "rocket" else WHITE
    weapon_text = font.render(f"Weapon: {player_tank.weapon_type}", True, color)
    health_text = font.render(f"Health: {player_tank.health:.0f}", True, ORANGE)
    score_text = font.render(f"Score: {player_tank.score:.0f}", True, WHITE)
    cool_text = font.render(f"Cool Time: {player_tank.cool_time:.0f}", True, MAGENTA)
    speed_text = font.render(f"Speed: {player_tank.speed:.1f}", True, CYAN)
    screen.blit(weapon_text, (WIDTH - weapon_text.get_width() - 10, HEIGHT - 200))
    screen.blit(health_text, (WIDTH - health_text.get_width() - 10, HEIGHT - 160))
    screen.blit(cool_text, (WIDTH - cool_text.get_width() - 10, HEIGHT - 120))
    screen.blit(speed_text, (WIDTH - speed_text.get_width() - 10, HEIGHT - 80))
    screen.blit(score_text, (WIDTH - score_text.get_width() - 10, HEIGHT - 40))  


def render_scope_view(screen, player_tank, obstacles, enemy_tanks):
    scope_size = 350
    scope_surface = pygame.Surface((scope_size, scope_size))
    scope_surface.fill(BLACK)
   
    barrel_angle_rad = math.radians(player_tank.barrel_angle)
    barrel_length = 15  # Adjust this value based on the length of the tank's barrel
    turret_height = 5  # Approximate height of the turret from the tank's base
    
    # Calculate the scope as we did the bullet direction in world space
    up_vector = Vector3(0, 1, 0)
    #right_vector = player_tank.direction.cross(up_vector).normalize()

    vertical_offset = barrel_length * math.sin(barrel_angle_rad)  + -5
    
    scope_camera_pos = player_tank.position - player_tank.direction * 10 + Vector3(10, vertical_offset, 5)

    scope_camera_dir = player_tank.direction     #.rotate(player_tank.barrel_angle, right_vector) #same as bullet direction does not work
    scope_camera_dir = scope_camera_dir# .normalize()
    fov = 20  

    # Render obstacles
    for obstacle in obstacles:
        render_3d_obstacle(scope_surface, obstacle, scope_camera_pos, scope_camera_dir, fov)
    
    # Render enemy tanks
    visible_tanks = []
    for tank in enemy_tanks:
        tank_pos_2d = project_3d_to_2d(tank.position, scope_camera_pos, scope_camera_dir, fov)
        if tank_pos_2d:
            color = tank.color
            render_3d_tank(scope_surface, tank, scope_camera_pos, scope_camera_dir, color, fov)
            range_to_enemy = (tank.position - player_tank.position).length()
            visible_tanks.append((tank_pos_2d[1], f"Range: {range_to_enemy:.1f}m"))
    
    # Render power-ups
    for poweer_up_list in [power_ups,enemy_power_ups]:
        for power_up in poweer_up_list:
            render_power_up(scope_surface, power_up, scope_camera_pos, scope_camera_dir, fov)
   
    # Render bullets
    for bullet_list in [player_bullets, enemy_bullets]:
            for bullet in bullet_list:
                render_3d_line(scope_surface, bullet.position, bullet.position + bullet.direction * 5, scope_camera_pos, scope_camera_dir, bullet.color, fov)

    # Render explosions
    for explosion in explosions:
        render_explosion(scope_surface, explosion, scope_camera_pos, scope_camera_dir)

    # Display ranges for visible tanks
    font = pygame.font.Font(None, 20)
    for i, (y, range_text) in enumerate(sorted(visible_tanks)):
        text = font.render(range_text, True, WHITE)
        scope_surface.blit(text, (10, 10 + i * 20))
    # Draw barrel angle indicator
    bar_height = 100
    bar_width = 10
    bar_x = scope_size - bar_width - 10
    bar_y = (scope_size - bar_height) // 2
    pygame.draw.rect(scope_surface, WHITE, (bar_x, bar_y, bar_width, bar_height), 1)
    indicator_y = bar_y + bar_height // 2 - (player_tank.barrel_angle / 90) * (bar_height // 2)
    # Add a line to show zero degrees on the barrel angle indidcator
    pygame.draw.line(scope_surface, WHITE, (bar_x, bar_y + bar_height // 2), (bar_x + bar_width, bar_y + bar_height // 2), 1)
    pygame.draw.rect(scope_surface, GREEN, (bar_x, indicator_y, bar_width, 1))
    # Draw a crosshair
    pygame.draw.circle(scope_surface, GREEN, (scope_size // 2, scope_size // 2), scope_size // 2, 1)
    pygame.draw.line(scope_surface, GREEN, (scope_size // 2, 0), (scope_size // 2, scope_size), 1)
    pygame.draw.line(scope_surface, GREEN, (0, scope_size // 2), (scope_size, scope_size // 2), 1)
    screen.blit(scope_surface, (10, 10))


def render_minimap(screen, player_tank, enemy_tanks, obstacles):
    global mode
    # Define map size and position
    map_size = 350
    map_pos = (WIDTH - map_size - 10, 10)  # Top right corner
    # Draw map border
    pygame.draw.rect(screen, WHITE, (*map_pos, map_size, map_size), 1)
    
    # Draw obstacles
    for obstacle in obstacles:
        position, size, shape = obstacle
        x = map_pos[0] + position.x * map_size / TERRAIN_SIZE
        y = map_pos[1] + (TERRAIN_SIZE - position.z) * map_size / TERRAIN_SIZE  # Invert z-axis
        if shape == "cube":
            pygame.draw.rect(screen, GREEN, (x - size/5, y - size/5, size/5, size/5))
        elif shape == "sphere":
            pygame.draw.circle(screen, GREEN, (int(x), int(y)), int(size/5))
        elif shape == "pyramid":
            pygame.draw.polygon(screen, GREEN, [(x, y - size/10), (x - size/10, y + size/10), (x + size/10, y + size/10)])
    
    # Draw enemy tanks
    for tank in enemy_tanks:
        x = map_pos[0] + tank.position.x * map_size / TERRAIN_SIZE
        y = map_pos[1] + (TERRAIN_SIZE - tank.position.z) * map_size / TERRAIN_SIZE  # Invert z-axis
        color = tank.color
        pygame.draw.circle(screen, color, (int(x), int(y)), 2)
    
    # Draw Power-ups
    for all_powerups in [power_ups,enemy_power_ups]:
        for power_up in all_powerups:
            x = map_pos[0] + power_up.position.x * map_size / TERRAIN_SIZE
            y = map_pos[1] + (TERRAIN_SIZE - power_up.position.z) * map_size / TERRAIN_SIZE  # Invert z-axis
            pygame.draw.circle(screen, power_up.color, (int(x), int(y)), 2)
            
    # Draw player tank
    x = map_pos[0] + player_tank.position.x * map_size / TERRAIN_SIZE
    y = map_pos[1] + (TERRAIN_SIZE - player_tank.position.z) * map_size / TERRAIN_SIZE  # Invert z-axis
    # Draw an arrow to show the direction of the player tank
    end_point = player_tank.position + player_tank.direction * 25
    end_x = map_pos[0] + end_point.x * map_size / TERRAIN_SIZE
    end_y = map_pos[1] + (TERRAIN_SIZE - end_point.z) * map_size / TERRAIN_SIZE  # Invert z-axis
    player_color = player_tank.color
    pygame.draw.line(screen, player_color, (int(x), int(y)), (int(end_x), int(end_y)), 2)                 
    pygame.draw.circle(screen, player_color, (int(x), int(y)), 3)


def reset_game():
    global player_tank, enemy_tanks, terrain, obstacles, player_bullets,enemy_bullets, explosions,enemy_power_ups,power_ups, last_power_up  
    explosions = []
    enemy_tanks = []
    terrain = create_terrain()
    obstacles = create_obstacles()
    player_bullets = []
    enemy_bullets = []
    power_ups = []
    enemy_power_ups = []
    last_power_up = Vector3(-1, -1, -1)

    while True:
        player_tank = Tank(Vector3(random.uniform(0, TERRAIN_SIZE), 0, random.uniform(0, TERRAIN_SIZE)), 
                              Vector3(random.uniform(-1, 1), 0, random.uniform(-1, 1)).normalize(), True)
        if not check_collision(player_tank.position, obstacles, enemy_tanks):
            break
    # Loop to check that enemy tank is not in an obstacle or another tank
    for _ in range(NUM_ENEMY_TANKS):
        while True:
            enemy_tank = Tank(Vector3(random.uniform(0, TERRAIN_SIZE), 0, random.uniform(0, TERRAIN_SIZE)), 
                              Vector3(random.uniform(-1, 1), 0, random.uniform(-1, 1)).normalize(), False)
            if not check_collision(enemy_tank.position, obstacles, [player_tank]):
                break
        enemy_tanks.append(enemy_tank)
    print("Game initialized")
    print(f"Player tank health: {player_tank.health}")
    print(f"Number of enemy tanks: {len(enemy_tanks)}") 


def create_power_up():
    """Create a new power-up at a random position"""
    position = Vector3(random.uniform(0, TERRAIN_SIZE), 0, random.uniform(0, TERRAIN_SIZE))
    type = random.choice(["health", "cooldown", "speed", "minigun", "cannon", "rocket"])
    #type = random.choice(["minigun", "cannon", "rocket"])
    return PowerUp(position, type)

def update_power_ups():
    """Remove expired power-ups and create new ones"""
    global power_ups
    power_ups = [p for p in power_ups if not p.is_expired()]
    if len(power_ups) < 6 and random.random() < 0.005:    # 1% chance to create a new power-up, max 3 at a time
        power_ups.append(create_power_up())

def apply_power_up(tank, power_up):
    """Apply the effect of a power-up to a tank"""
    if power_up.type == "health":
        tank.health = min(tank.health + power_up.value, 5000) # Health cannot go above the maximum
    elif power_up.type == "cooldown":
        tank.cool_time = max(tank.cool_time - power_up.value, 10) # Cooldown cannot go below 10
    elif power_up.type == "speed":
        tank.speed = min(tank.speed + power_up.value, 5) # Speed cannot go above 5
    elif power_up.type == "minigun":
        tank.weapon_type = "minigun"
        tank.cool_time = 5
    elif power_up.type == "cannon":
        tank.weapon_type = "cannon"
        tank.cool_time = 30
    elif power_up.type == "rocket":
        tank.weapon_type = "rocket"
        tank.cool_time = 100
    
def check_power_up_collision(tank):
    """Check if a tank has collided with a power-up"""
    global power_ups, enemy_power_ups, last_power_up
    for power_list in[power_ups,enemy_power_ups]:
        for power_up in power_list:
            if (tank.position - power_up.position).length() < 10:
                if last_power_up != power_up.position:
                    apply_power_up(tank, power_up)
                    print(f"Tank {tank.color} picked up a {power_up.type} power-up: +{power_up.value:.2f}")
                    last_power_up = power_up.position
                    power_list.remove(power_up)
                    if power_list is power_ups:
                        print("Removed from power_ups list")
                    else:
                        print("Removed from enemy_power_ups list")
                    break

def render_power_up(screen, power_up, camera_pos, camera_dir, fov):
    """Render a power-up on the screen"""
    color = power_up.color
    render_3d_cube(screen, power_up.position, 5, camera_pos, camera_dir, color, fov)

#Network Code
def scan_network(host_ip, port, servers, progress_queue, stop_event):
    """
    Scan the network for open ports and update progress.
    """
    nm = nmap.PortScanner()
    network = f"{'.'.join(host_ip.split('.')[:3])}.0/24"
    
    total_hosts = 254  # A /24 network has 254 usable host addresses
    scanned_hosts = 0
    start_time = time.time()
    
    for i in range(1, 255):
        if stop_event.is_set():
            break
        
        host = f"{'.'.join(host_ip.split('.')[:3])}.{i}"
        scanned_hosts += 1
        progress = (scanned_hosts / total_hosts) * 100
        elapsed_time = time.time() - start_time
        eta = (elapsed_time / scanned_hosts) * (total_hosts - scanned_hosts) if scanned_hosts > 0 else 0
        
        # Update progress
        progress_queue.put(('progress', (progress, scanned_hosts, total_hosts, elapsed_time, eta)))
        
        try:
            result = nm.scan(host, str(port), arguments='-sT -T4')
            if host in result['scan'] and result['scan'][host]['tcp'][port]['state'] == 'open':
                servers.append(host)
                # Notify about new server
                progress_queue.put(('server', host))
        except:
            pass  # Skip hosts that can't be scanned
    
    # Notify scan completion
    progress_queue.put(('done', None))

def network_send(data):
    global client_socket
    #print (f"Sending data: {data}")
    try:
        if client_socket is None:
            raise RuntimeError("Socket not initialized")
        serialized_data = pickle.dumps(data)
        data_size = len(serialized_data)
        # Set a timeout for send operations
        client_socket.settimeout(.1)  # .1 seconds timeout
        client_socket.send(data_size.to_bytes(4, byteorder='big'))
        client_socket.send(serialized_data)
        #print(f"Data sent: {data}")
    except socket.timeout:
        print("Send operation timed out")
        return False    
    except (ConnectionResetError, BrokenPipeError):
        print("Connection lost. Attempting to reconnect...")
        handle_network_disconnect()
    except Exception as e:
        print(f"Error sending data: {e}")
        handle_network_disconnect()

def network_send_game_state(player_tank, bullets, power_ups, last_power_up):
    game_state = {
        "player_tank": player_tank,
        "bullets": bullets,
        "power_ups": power_ups,
        "last_power_up": last_power_up
    }
    network_send(game_state)

def network_receive():
    global client_socket
    try:
        client_socket.settimeout(1)  # Set timeout to .1 seconds
        if client_socket is None:
            raise RuntimeError("Socket not initialized")
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
        #print(f"Data received:") # {pickle.loads(data)}")
        return pickle.loads(data)
    except socket.timeout:
        print("Socket timeout occurred")
        #handle_network_disconnect
        return None
    except (ConnectionResetError, BrokenPipeError):
        print("Connection lost. Attempting to reconnect...")
        handle_network_disconnect()
        return None
    except Exception as e:
        print(f"Error receiving data: {e}")
        handle_network_disconnect()
        return None

def server_listen():
    global client_socket, server_socket
    # Close existing sockets if they exist
    if server_socket:
        server_socket.close()
    if client_socket:
        client_socket.close()
    # Create a new socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    
    print(f"Server listening on {HOST}:{PORT}")
    print("Waiting for 2nd player to join...")
    
    # Prepare text for display
    test0 = f"Server is listening on {HOST}:{PORT}"
    text = "Waiting for 2nd player to join..."
    font = pygame.font.Font(None, 36)
    text_surface = font.render(text, True, WHITE)
    text_surface0 = font.render(test0, True, WHITE)
    
    # Display text on screen
    screen.blit(text_surface0, (WIDTH/2 - 150, 400))
    screen.blit(text_surface, (WIDTH/2 - 150, 450))
    pygame.display.flip()
    
    # Set the timeout to 6 minutes (360 seconds)
    timeout = 360
    start_time = time.time()
    
    # Set the socket to non-blocking mode
    server_socket.setblocking(False)
    
    while True:
        # Check for pygame events (like quit)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            #check for escape key
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return False
        
        # Try to accept a connection
        try:
            client_socket, addr = server_socket.accept()
            print(f"Connected by {addr}")
            return True
        except BlockingIOError:
            # No connection yet, continue waiting
            pass
        
        # Check if timeout has been reached
        if time.time() - start_time > timeout:
            print("Timeout: No connection after 3 minutes")
            return False
        
        # Update the countdown on screen
        remaining_time = int(timeout - (time.time() - start_time))
        countdown_text = f"Time remaining: {remaining_time} seconds"
        countdown_surface = font.render(countdown_text, True, WHITE)
        
        # Redraw the screen
        screen.fill(BLACK)  # Assuming BLACK is defined as the background color
        screen.blit(text_surface0, (WIDTH/2 - 150, 300))
        screen.blit(text_surface, (WIDTH/2 - 150, 350))
        screen.blit(countdown_surface, (WIDTH/2 - 150, 400))
        pygame.display.flip()
        
        # Small delay to prevent excessive CPU usage
        pygame.time.delay(100)    

def client_connect(max_attempts=9, retry_delay=15):
    global client_socket
    for attempt in range(max_attempts):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"Attempting to connect to Game server at {HOST}:{PORT} (attempt {attempt + 1}/{max_attempts})...")
            client_socket.connect((HOST, PORT))
            print(f"Connected to Game server at {HOST}:{PORT}")
            text = pygame.font.Font(None, 36).render(f"Connected to Game server at {HOST}:{PORT}", True, WHITE)
            screen.blit(text, (WIDTH/2 - 150, 550+50*attempt+1))
            pygame.display.flip()
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            text = pygame.font.Font(None, 36).render(f"Attempt {attempt + 1} failed: {e}", True, WHITE)
            screen.blit(text, (WIDTH/2 - 150, 550+50*attempt+1))
            pygame.display.flip()
            if attempt < max_attempts - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Failed to connect to game server after maximum attempts")
                text = pygame.font.Font(None, 36).render("Failed to connect to game server after maximum attempts", True, WHITE)
                screen.blit(text, (WIDTH/2 - 150, 550+50*attempt+1))
                pygame.display.flip()
    return False

def handle_network_disconnect():
    global client_socket, server_socket, game_over, game_over_message, is_server
    if client_socket:
        client_socket.close()
        client_socket = None
    if server_socket:
        server_socket.close()
        server_socket = None
    # Attempt to reconnect
    max_attempts = 5
    for attempt in range(max_attempts):
        print(f"Attempting to reconnect (attempt {attempt + 1}/{max_attempts})...")
        if is_server:
            if server_listen():
                print("Server Reconnected successfully")
                return
        else:
            if client_connect():
                print("Client Reconnected successfully")
                return
        #time.sleep(2)  # Wait for 2 seconds before next attempt
    game_over = True
    game_over_message = "Network connection lost and unable to reconnect"


def choose_game_mode():
    global HOST
    screen.fill(BLACK)
    font = pygame.font.Font(None, 36)
    # Render menu options
    intro_text = font.render("Chose Game Mode, if you host share your IP with other player", True, WHITE)
    single_player_text = font.render("1. Single Player", True, WHITE)
    two_player_host_text = font.render(f"2. Two Player Host @{HOST}", True, WHITE)
    two_player_join_text = font.render("3. Two Player (Join)", True, WHITE)
    # Position menu options
    screen.blit(intro_text, (WIDTH/2 - 300, 100))
    screen.blit(single_player_text, (WIDTH/2 - 150,150))
    screen.blit(two_player_host_text, (WIDTH/2 - 150, 200))
    screen.blit(two_player_join_text, (WIDTH/2 - 150, 250))
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "single"
                elif event.key == pygame.K_2:
                    return "host"
                elif event.key == pygame.K_3:
                    # Get IP address input
                    HOST = pygame_input(f"Enter IP address (default: {DEFAULT_HOST})", DEFAULT_HOST)
                    if HOST == "quit":
                        return "quit"
                return ("join")

def reset_game_two_player(max_retries=9, retry_delay=2):
    global player_tank, enemy_tanks, terrain, obstacles, player_bullets, enemy_bullets, explosions, is_server, client_socket, power_ups, enemy_power_ups, last_power_up
    for attempt in range(max_retries):
        try:
            # Reset game variables
            explosions = []
            player_bullets = []
            enemy_bullets = []
            enemy_tanks = []
            power_ups = []
            enemy_power_ups = []
            last_power_up = Vector3(-1, -1, -1)

            if is_server:
                # Server generates the initial game state
                terrain = create_terrain()
                obstacles = create_obstacles()
                # Send initial game data to client
                initial_data = {
                    "terrain": terrain,
                    "obstacles": obstacles,
                    "initial_data": True
                }
                print("Sending initial game data to client")
                network_send(initial_data)
                # Wait for client acknowledgment with timeout
                ack_received = False
                for i in range(5):  # Try 5 times
                    ack = network_receive()
                    if ack and "initial_data_ack" in ack:
                        ack_received = True
                        print("Client acknowledgment received")
                        break
                    print("Waiting for client acknowledgment try:", i)
                    time.sleep(1)  # Wait 1 second before trying again
                    
                if not ack_received:
                    raise Exception("Failed to receive client acknowledgment")
                
            else:
                # Client receives the initial game state
                received_data = None
                for i in range(5):  # Try 5 times
                    received_data = network_receive()
                    if received_data and "terrain" in received_data and "obstacles" in received_data and "initial_data" in received_data:
                        print("Received initial game data from server")
                        terrain = received_data["terrain"]
                        obstacles = received_data["obstacles"]
                       
                        # Send acknowledgment
                        network_send({"initial_data_ack": True})
                        print("Acknowledgment sent to server")
                        break
                    print("Waiting for initial game data from server try:", i)
                    time.sleep(1)  # Wait 1 second before trying again
                
                if not received_data:
                    raise Exception("Failed to receive complete initial game data")

            print("Two-player game initialized!")
            # Place tanks after obstacles
            max_placement_attempts = 100
            for _ in range(max_placement_attempts):
                player_tank = Tank(Vector3(random.uniform(0, TERRAIN_SIZE), 0, random.uniform(0, TERRAIN_SIZE)), 
                                   Vector3(random.uniform(-1, 1), 0, random.uniform(-1, 1)).normalize(), mode)
                if not check_collision(player_tank.position, obstacles, enemy_tanks):
                    print(f"Player tank position: {player_tank.position}")
                    break
            else:
                raise Exception("Failed to place player tank without collision")

            # Exchange tank positions
            network_send({"player_tank": player_tank})
            
            enemy_tank_received = False
            for i in range(5):  # Try 5 times
                received_data = network_receive()
                if received_data and "player_tank" in received_data:
                    enemy_tanks = [received_data["player_tank"]]
                    print(f"Enemy tank position: {enemy_tanks[0].position}")
                    enemy_tank_received = True
                    break
                print("Waiting for enemy tank data try:", i)
                time.sleep(1)  # Wait 1 second before trying again
            
            if not enemy_tank_received:
                raise Exception("Failed to receive enemy tank data")
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
    global player_tank, enemy_tanks, terrain, obstacles, player_bullets,enemy_bullets, explosions, is_server, client_socket, mode,power_ups,enemy_power_ups,initial_data,last_power_up
    initial_data = False
    show_help_reset(screen)
    mode = choose_game_mode()
    if mode == "quit":
        return
    elif mode == "single":
        reset_game()
    elif mode == "host":
        is_server = True
        if server_listen():
            if not reset_game_two_player():
                print("Failed to initialize two-player game")
                return
        else:
            print("Failed to establish server connection")
            return
        print("Connected to client")
    elif mode == "join":
        is_server = False
        if client_connect():
            if not reset_game_two_player():
                print("Failed to initialize two-player game")
                return
        else:
            print("Failed to connect to server")
            return
        print("Connected to server")
    clock = pygame.time.Clock()
    fov = 60
    running = True
    show_help_screen = False
    game_over = False
    old_temp = Vector3(-1, -1, -1)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        # Handle player input
        player_input = {
            "up": keys[pygame.K_UP],
            "down": keys[pygame.K_DOWN],
            "left": keys[pygame.K_LEFT],
            "right": keys[pygame.K_RIGHT],
            "q": keys[pygame.K_q],
            "z": keys[pygame.K_z],
            "space": keys[pygame.K_SPACE],
            "quit": keys[pygame.K_ESCAPE],
            "help": keys[pygame.K_h],
            "reset": keys[pygame.K_r],
            "exit": keys[pygame.K_x],
        }
        # Apply player input
        if player_input["up"]:
            player_tank.move(True, obstacles, enemy_tanks=enemy_tanks)
        if player_input["down"]:
            player_tank.move(False, obstacles,enemy_tanks=enemy_tanks)
        if player_input["left"]:
            player_tank.rotate(1)
        if player_input["right"]:
            player_tank.rotate(-1)
        if player_input["q"]:
            player_tank.barrel_angle_adjust(1)
        if player_input["z"]:
            player_tank.barrel_angle_adjust(-1)
        if player_input["space"]:
            new_bullet = player_tank.fire()
            if new_bullet:
                player_bullets.append(new_bullet)
        if player_input["quit"]:
            game_over_message = "You Exited the game."
            game_over = True
        if player_input["help"]:
            running, game_over =  show_help_reset(screen)
        if player_input["reset"] and game_over and not explosions:
            screen.fill(BLACK)
            pygame.display.flip()
            return main()
        if player_input["exit"] and game_over and not explosions:
            running = False

        if not game_over or explosions:
            # In the main game loop, for two-player mode:
            if mode != "single":
                try:
                    network_send_game_state(player_tank, player_bullets, power_ups, last_power_up)
                    received_state = network_receive()
                    if received_state:
                        if "player_tank" in received_state:
                            enemy_tanks[0] = received_state["player_tank"]
                        if "bullets" in received_state:
                            enemy_bullets = received_state["bullets"]
                        if "power_ups" in received_state:
                            enemy_power_ups = received_state["power_ups"]
                        if "last_power_up" in received_state:
                            temp = received_state["last_power_up"]
                            if temp != old_temp:
                                print (f"my last power-up: {last_power_up}")    
                                print (f"Removing last power-up from power_ups: {temp}")
                                last_power_up = temp
                                power_ups = [powerup for powerup in power_ups if powerup.position != temp]
                                old_temp = temp
                    else:
                        print("No state received")
                except Exception as e:
                    print(f"Error in MAIN LOOP: {e}")
                    handle_network_disconnect()                    
            # Update enemy tanks
            if mode == "single":
                for enemy_tank in enemy_tanks:
                    # Aim at player tank
                    to_player = player_tank.position - enemy_tank.position
                    target_direction = Vector3(to_player.x, 0, to_player.z).normalize()
                    angle_diff = enemy_tank.direction.angle_to(target_direction)
                    rotation = min(2, max(-2, angle_diff))
                    enemy_tank.rotate(rotation)
                    if angle_diff < 5:
                        new_bullet = enemy_tank.fire()
                        if new_bullet:
                            enemy_bullets.append(new_bullet)
                    elif to_player.length() < 100 and angle_diff < 20:
                        new_bullet = enemy_tank.fire()
                        if new_bullet:
                            enemy_bullets.append(new_bullet)
                    elif random.random() < 0.01 and angle_diff < 90:
                        new_bullet = enemy_tank.fire()
                        if new_bullet:
                            enemy_bullets.append(new_bullet)
                    enemy_tank.move(True, obstacles, [player_tank] + enemy_tanks)

            # Update bullets and check collisions
            for bullet_list in [player_bullets, enemy_bullets]:
                for bullet in bullet_list[:]:  # Create a copy of the list to iterate over
                    if not bullet.update():
                        if bullet in bullet_list:
                            bullet_list.remove(bullet)
                        continue

                    for tank in [player_tank] + enemy_tanks:
                        if (bullet.position - tank.position).length() < 7 and bullet.is_player_bullet != tank.is_player:
                            tank.health -= bullet.damage
                            tank.speed = max (tank.speed * 0.9, 1)
                            tank.cool_time = min (tank.cool_time * 1.1, 100)
                            if bullet in bullet_list:
                                bullet_list.remove(bullet)
                            if tank.health <= 0:
                                explosions.append(Explosion(tank.position, tank, 60))
                            else:
                                explosions.append(Explosion(tank.position, tank))
                            if mode == "single" and not tank.is_player:
                                player_tank.score += bullet.damage
                            
                            if mode == "host" and tank.is_player == "join" or mode == "join" and tank.is_player == "host":
                                player_tank.score += bullet.damage
                            break

                    for obstacle in obstacles:
                        if (bullet.position - obstacle[0]).length() < obstacle[1]:
                            if bullet in bullet_list:
                                bullet_list.remove(bullet)
                            break

                    for power_list in [power_ups,enemy_power_ups]:
                        for power_up in power_list:
                            if (bullet.position - power_up.position).length() < 7:
                                if bullet in bullet_list:
                                    bullet_list.remove(bullet)
                                    #remove powerup
                                    power_list.remove(power_up)
                                break
            
            update_power_ups()
            check_power_up_collision(player_tank)
            
            # Update and render explosions
            explosions = [exp for exp in explosions if exp.update()]

            # Render
            screen.fill(BLACK)
            
            # The camera needs to not just be behind the tank and above but angled so that it is always looking at the tank from straighy behind
            camera_pos = player_tank.position - player_tank.direction * 100 + Vector3(0, 40, 0)
            camera_dir = player_tank.direction.copy()

            # Render terrain
            for line in terrain:
                render_3d_line(screen, line[0], line[1], camera_pos, camera_dir, GREEN)

            # Render obstacles
            for obstacle in obstacles:
                render_3d_obstacle(screen, obstacle, camera_pos, camera_dir, fov)

            # Render Power-ups
            for power_up_list in [power_ups,enemy_power_ups]:
                for power_up in power_up_list:
                    render_power_up(screen, power_up, camera_pos, camera_dir, fov)
            
            # Render tanks
            tanks_to_render = [player_tank] + enemy_tanks
            for tank in tanks_to_render:
                color = tank.color
                if game_over and tank.health <= 0:
                    color = RED
                render_3d_tank(screen, tank, camera_pos, camera_dir, color, fov)
        
            # Render bullets
            for bullet_list in [player_bullets, enemy_bullets]:
                    for bullet in bullet_list:
                        render_3d_line(screen, bullet.position, bullet.position + bullet.direction * 5, camera_pos, camera_dir, bullet.color)

            # Render explosions
            for explosion in explosions:
                render_explosion(screen, explosion, camera_pos, camera_dir)

            render_hud(screen, player_tank)
            render_scope_view(screen, player_tank, obstacles, enemy_tanks)
            render_minimap(screen, player_tank, enemy_tanks, obstacles)
            pygame.display.flip()
            
            # Update tanks gun cooldown
            player_tank.update()
            for enemy_tank in enemy_tanks:
                enemy_tank.update()

            # Check game over conditions
            if mode == "single":
                if player_tank.health <= 0:
                    game_over = True
                    game_over_message = "Game Over! You lost."
                elif not enemy_tanks:
                    game_over = True
                    game_over_message = "Congratulations! You won!"
            else:  # Two-player mode
                if player_tank.health <= 0:
                    game_over = True
                    game_over_message = "Game Over! You lost."
                #check if enemy tank has not been removed from list or you get an error
                elif not enemy_tanks or enemy_tanks[0].health <= 0:
                    game_over = True
                    game_over_message = "Congratulations! You won!"

        if game_over and not explosions:
            font = pygame.font.Font(None, 64)
            text = font.render(game_over_message, True, WHITE)
            text_rect = text.get_rect(center=(WIDTH/2, HEIGHT/2 + 25 ))
            screen.blit(text, text_rect)
            font = pygame.font.Font(None, 32)
            continue_text = font.render("Press R to Restart or X to eXit", True, WHITE)
            continue_rect = continue_text.get_rect(center=(WIDTH/2, HEIGHT/2 + 75 ))
            screen.blit(continue_text, continue_rect)
        
        pygame.display.flip()
        clock.tick(60)

    print("Game loop ended")
    print(f"Final player health: {player_tank.health}")
    print(f"Remaining enemy tanks: {len(enemy_tanks)}")

    # Close network connections
    if client_socket:
        client_socket.close()
    if server_socket:
        server_socket.close()

    # Keep the window open until the user closes it
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

if __name__ == "__main__":
    main()