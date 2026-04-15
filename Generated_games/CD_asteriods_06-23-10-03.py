import pygame
import math
import random

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Asteroids")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Player ship
class Ship:
    def __init__(self):
        self.pos = [WIDTH // 2, HEIGHT // 2]
        self.angle = 0
        self.speed = [0, 0]
        self.radius = 15

    def rotate(self, direction):
        self.angle += direction * 5

    def thrust(self):
        angle_rad = math.radians(self.angle)
        self.speed[0] += math.cos(angle_rad) * 0.1
        self.speed[1] -= math.sin(angle_rad) * 0.1

    def update(self):
        self.pos[0] = (self.pos[0] + self.speed[0]) % WIDTH
        self.pos[1] = (self.pos[1] + self.speed[1]) % HEIGHT

    def draw(self):
        angle_rad = math.radians(self.angle)
        points = [
            (self.pos[0] + self.radius * math.cos(angle_rad),
             self.pos[1] - self.radius * math.sin(angle_rad)),
            (self.pos[0] + self.radius * math.cos(angle_rad + 2.6),
             self.pos[1] - self.radius * math.sin(angle_rad + 2.6)),
            (self.pos[0] + self.radius * math.cos(angle_rad - 2.6),
             self.pos[1] - self.radius * math.sin(angle_rad - 2.6))
        ]
        pygame.draw.polygon(screen, WHITE, points, 1)

# Bullet
class Bullet:
    def __init__(self, pos, angle):
        self.pos = list(pos)
        self.speed = [math.cos(math.radians(angle)) * 5,
                      -math.sin(math.radians(angle)) * 5]
        self.lifetime = 60

    def update(self):
        self.pos[0] = (self.pos[0] + self.speed[0]) % WIDTH
        self.pos[1] = (self.pos[1] + self.speed[1]) % HEIGHT
        self.lifetime -= 1

    def draw(self):
        pygame.draw.circle(screen, WHITE, (int(self.pos[0]), int(self.pos[1])), 2)

# Asteroid
class Asteroid:
    def __init__(self, pos=None, size=3):
        if pos is None:
            side = random.choice(['top', 'bottom', 'left', 'right'])
            if side == 'top':
                pos = [random.randint(0, WIDTH), 0]
            elif side == 'bottom':
                pos = [random.randint(0, WIDTH), HEIGHT]
            elif side == 'left':
                pos = [0, random.randint(0, HEIGHT)]
            else:
                pos = [WIDTH, random.randint(0, HEIGHT)]
        self.pos = list(pos)
        self.size = size
        self.speed = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.radius = 20 * size
        self.points = self.generate_points()

    def generate_points(self):
        points = []
        for i in range(8):
            angle = i * 45 + random.uniform(-10, 10)
            distance = self.radius + random.uniform(-5, 5)
            x = math.cos(math.radians(angle)) * distance
            y = math.sin(math.radians(angle)) * distance
            points.append((x, y))
        return points

    def update(self):
        self.pos[0] = (self.pos[0] + self.speed[0]) % WIDTH
        self.pos[1] = (self.pos[1] + self.speed[1]) % HEIGHT

    def draw(self):
        draw_points = [(self.pos[0] + x, self.pos[1] + y) for x, y in self.points]
        pygame.draw.polygon(screen, WHITE, draw_points, 1)

# Game state
ship = Ship()
bullets = []
asteroids = [Asteroid() for _ in range(4)]
score = 0
lives = 3
level = 1
game_over = False

# Font
font = pygame.font.Font(None, 36)

# Game loop
clock = pygame.time.Clock()
running = True

def draw_text(text, pos):
    surface = font.render(text, True, WHITE)
    screen.blit(surface, pos)

def show_instructions():
    instructions = [
        "Instructions:",
        "Arrow keys: Rotate ship",
        "Up arrow: Thrust",
        "Space: Fire",
        "H: Show/hide instructions",
        "Q: Quit game",
        "",
        "Press any key to continue"
    ]
    screen.fill(BLACK)
    for i, line in enumerate(instructions):
        draw_text(line, (50, 50 + i * 40))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                waiting = False
    return True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                bullets.append(Bullet(ship.pos, ship.angle))
            elif event.key == pygame.K_h:
                if not show_instructions():
                    running = False
            elif event.key == pygame.K_q:
                running = False

    if not game_over:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            ship.rotate(1)
        if keys[pygame.K_RIGHT]:
            ship.rotate(-1)
        if keys[pygame.K_UP]:
            ship.thrust()

        ship.update()
        for bullet in bullets:
            bullet.update()
        bullets = [b for b in bullets if b.lifetime > 0]

        for asteroid in asteroids:
            asteroid.update()

        # Collision detection
        for bullet in bullets[:]:
            for asteroid in asteroids[:]:
                if math.hypot(bullet.pos[0] - asteroid.pos[0],
                              bullet.pos[1] - asteroid.pos[1]) < asteroid.radius:
                    bullets.remove(bullet)
                    asteroids.remove(asteroid)
                    score += (4 - asteroid.size) * 100
                    if asteroid.size > 1:
                        for _ in range(2):
                            asteroids.append(Asteroid(asteroid.pos, asteroid.size - 1))
                    break

        if not asteroids:
            level += 1
            for _ in range(level + 3):
                asteroids.append(Asteroid())

        for asteroid in asteroids:
            if math.hypot(ship.pos[0] - asteroid.pos[0],
                          ship.pos[1] - asteroid.pos[1]) < asteroid.radius + ship.radius:
                lives -= 1
                if lives == 0:
                    game_over = True
                else:
                    ship = Ship()
                    break

    # Drawing
    screen.fill(BLACK)
    
    if not game_over:
        ship.draw()
        for bullet in bullets:
            bullet.draw()
        for asteroid in asteroids:
            asteroid.draw()

        draw_text(f"Score: {score}", (10, 10))
        draw_text(f"Lives: {lives}", (10, 50))
        draw_text(f"Level: {level}", (10, 90))
    else:
        draw_text("GAME OVER", (WIDTH // 2 - 70, HEIGHT // 2 - 18))
        draw_text(f"Final Score: {score}", (WIDTH // 2 - 70, HEIGHT // 2 + 18))
        draw_text("Press R to restart or Q to quit", (WIDTH // 2 - 150, HEIGHT // 2 + 54))

        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            ship = Ship()
            bullets = []
            asteroids = [Asteroid() for _ in range(4)]
            score = 0
            lives = 3
            level = 1
            game_over = False

    pygame.display.flip()
    clock.tick(60)

pygame.quit()