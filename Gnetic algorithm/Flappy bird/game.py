import pygame
import sys
import random
import time


pygame.init()
clock = pygame.time.Clock()
WIDTH, HEIGHT = 1600, 900
PADDING = 50

Font=pygame.font.SysFont('timesnewroman',  50)
WHITE = (255, 255, 255)
BLACK = (25, 25, 25)
GRAY = (100, 100, 100)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy bird")





class Pipe:
    def __init__(self):
        self.width = 100
        self.x = WIDTH + self.width
        self.height = 300
        self.y = random.randint(0, HEIGHT - self.height)
        self.color = (110, 255, 110)


    def calculate_frame(self):
        self.x -= 3


    def draw(self):
        pygame.draw.rect(screen, self.color, pygame.Rect(self.x, 0, self.width, self.y))
        pygame.draw.rect(screen, self.color, pygame.Rect(self.x, self.y + self.height, self.width, HEIGHT))



class Bird:
    def __init__(self):
        self.width = 50
        self.height = 50
        self.x = 300
        self.y = HEIGHT // 2
        self.gravity = 1
        self.speed = 0
        self.color = [random.randint(1, 255) for _ in range(3)]
        self.is_alive = True
        self.score = 0


    def jump(self):
        self.speed = -20


    def claculate_frame(self):
        self.speed += self.gravity
        self.y += self.speed

        if self.y < 0:
            self.y = 0
            self.speed = 0
        if self.y + self.height > HEIGHT:
            self.y = HEIGHT - self.height
            self.speed = 0


    def collision_detection(self, pipe:Pipe):
        if self.y < pipe.y or self.y + self.height > pipe.y + pipe.height:
            self.score = FlappyBird.score
            self.is_alive = False


    def draw(self):
        pygame.draw.rect(screen, self.color, pygame.Rect(self.x, self.y, self.width, self.height))



class FlappyBird:
    score = 0
    def __init__(self, birds:list[Bird]):
        self.birds = birds
        self.pipes = [Pipe()]
        self.last_pipe_generation = time.time()
        self.collosion_pipe = self.pipes[0]
        self.bird_x = 300
        self.bird_width = 50


    def calculate_frame(self):
        if time.time() - self.last_pipe_generation > 2:
            self.pipes.append(Pipe())
            self.last_pipe_generation = time.time()

        if self.pipes and self.pipes[0].x + self.pipes[0].width < 0: self.pipes.pop(0)

        for bird in self.birds:
            if bird.is_alive:
                bird.claculate_frame()
            else:
                bird.x -= 3
        for pipe in self.pipes:
            pipe.calculate_frame()
        
        if self.collosion_pipe.x + self.collosion_pipe.width < self.bird_x:
            FlappyBird.score += 1
            for pipe in self.pipes:
                if pipe == self.collosion_pipe: continue
                self.collosion_pipe = pipe
                break

        if self.bird_x < self.collosion_pipe.x + self.collosion_pipe.width and \
           self.collosion_pipe.x < self.bird_x + self.bird_width:
            for bird in self.birds:
                if bird.is_alive:
                    bird.collision_detection(self.collosion_pipe)


    def draw_game(self):
        for bird in self.birds:
            bird.draw()
        for pipe in self.pipes:
            pipe.draw()

        score = Font.render(str(FlappyBird.score), False, WHITE, None)
        screen.blit(score, score.get_rect(center=(50, 50)))




game = FlappyBird([Bird()])



running = True

while running:
    mousex, mousey = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                game.birds[0].jump()


    screen.fill(BLACK)
    game.draw_game()
    game.calculate_frame()
    pygame.display.flip()

    clock.tick(60)



pygame.quit()
sys.exit()



