
import pygame
import random
import time
import nltk
from nltk.corpus import words

# Download the list of English words if it's not already downloaded
nltk.download('words', quiet=True)

# Get the list of English words
english_words = set(word.lower() for word in words.words() if word.isalpha())

# Filter the list to only include 5-letter words
five_letter_words = [word for word in english_words if len(word) == 5]

def play_game():
    # Initialize Pygame
    pygame.init()

    # Set up some constants
    WIDTH, HEIGHT = 800, 600
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    RED = (255, 0, 0)
    ORANGE = (255, 165, 0)
    GREY = (128, 128, 128)
    FONT = pygame.font.Font(None, 36)

    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("JMR's Super Wordle")

    # Choose a random word from the list
    word = random.choice(five_letter_words)

    # Set up the game state
    guesses = [[]]  # Initialize the first guess
    not_in_word = set()
    start_time = time.time()
    max_guesses = 6
    invalid_guesses = []
    guessed_letters = {}

    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    # Get the current guess
                    guess = "".join(guesses[-1])
                    print(guess)
                    # Check if the guess is a valid word
                    if guess.lower() in five_letter_words:
                        # Check if the guess is correct
                        if guess.lower() == word:
                            print("Congratulations, you won!")
                            running = False
                        else:
                            # Update the list of letters not in the word
                            for letter in guess.lower():
                                if letter not in word:
                                    not_in_word.add(letter)
                            # Clear the current guess
                            guesses.append([])
                    else:
                        print("Invalid word. Try again.")
                        invalid_guesses.append(guess)
                        #guesses.append([])
                elif event.key == pygame.K_BACKSPACE:
                    # Remove the last letter from the current guess
                    if guesses[-1]:
                        guesses[-1].pop()
                elif len(guesses[-1]) < 5 and chr(event.key).isalpha():
                    guesses[-1].append(chr(event.key).lower())

        # Draw everything
        screen.fill(WHITE)

        # Draw the keyboard
        keyboard = [
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm']
        ]
        for i, row in enumerate(keyboard):
            for j, letter in enumerate(row):
                color = BLACK
                if letter in guessed_letters:
                    if guessed_letters[letter]:
                        color = GREEN
                    else:
                        color = GREY
                text = FONT.render(letter, True, color)
                screen.blit(text, (50 + j * 36, 300 + i * 36))

        for i, guess in enumerate(guesses):
            for j, letter in enumerate(guess):
                if i == len(guesses) - 1:
                    color = GREY
                else:
                    if letter == word[j]:
                        color = GREEN
                    elif letter in word:
                        color = ORANGE
                    else:
                        color = BLACK
                text = FONT.render(letter, True, color)
                screen.blit(text, (50 + j * 36, 50 + i * 36))
        text = FONT.render("Not in word: " + ", ".join(sorted(not_in_word)), True, BLACK)
        screen.blit(text, (50, HEIGHT - 50))
        text = FONT.render("Time: " + str(int(time.time() - start_time)), True, BLACK)
        screen.blit(text, (50, HEIGHT - 80))
        text = FONT.render("Guesses left: " + str(max_guesses - len(guesses) + 1), True, BLACK)
        screen.blit(text, (50, HEIGHT - 110))
        text = FONT.render("Invalid guesses: " + ", ".join(invalid_guesses), True, RED)
        screen.blit(text, (50, HEIGHT - 140))
        text = FONT.render("5-letter words: " + str(len(five_letter_words)), True, BLACK)
        screen.blit(text, (WIDTH - 300, HEIGHT - 50))
        pygame.display.flip()

        # Check if the player has run out of guesses
        if len(guesses) > max_guesses:
            print("You lost! The word was " + word)
            running = False

        # Update the guessed letters
        for guess in guesses:
            for letter in guess:
                if letter in word:
                    guessed_letters[letter] = True
                else:
                    guessed_letters[letter] = False

    # Game over screen
    game_over = True
    while game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    # Restart the game
                    play_game()
                    game_over = False
                elif event.key == pygame.K_n:
                    game_over = False

        # Draw the game over screen
        screen.fill(WHITE)
        if len(guesses) > max_guesses:
            text = FONT.render("You lost! The word was " + word, True, RED)
        else:
            text = FONT.render("Congratulations, you won!", True, GREEN)
        screen.blit(text, (50, HEIGHT // 2 - 18))
        text = FONT.render("Play again? (y/n)", True, BLACK)
        screen.blit(text, (50, HEIGHT // 2 + 18))
        pygame.display.flip()

    # Quit Pygame
    pygame.quit()

play_game()
