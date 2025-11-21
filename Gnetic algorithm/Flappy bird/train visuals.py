import pygame
import sys
import random
import time
from brain import NeuralNetwork


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
    def __init__(self, brain:NeuralNetwork):
        self.width = 50
        self.height = 50
        self.x = 300
        self.y = HEIGHT // 2
        self.gravity = 1
        self.speed = 0
        self.color = [random.randint(1, 255) for _ in range(3)]
        self.is_alive = True
        self.score = 0
        self.brain = brain


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


    def collision_detection(self, pipe:Pipe, score):
        if self.y < pipe.y or self.y + self.height > pipe.y + pipe.height:
            self.score = score
            self.is_alive = False


    def draw(self):
        pygame.draw.rect(screen, self.color, pygame.Rect(self.x, self.y, self.width, self.height))


    def make_descision(self, pipe:Pipe):
        result = self.brain.forward([
            self.y / HEIGHT, 
            pipe.y / HEIGHT,
            ])
        
        if result[0][0] > result[0][1]:
            self.jump()



class FlappyBird:
    generation = 1

    def __init__(self, birds:list[Bird]):
        self.birds = birds
        self.pipes = [Pipe()]
        self.last_pipe_generation = time.time()
        self.collosion_pipe = self.pipes[0]
        self.bird_x = 300
        self.bird_width = 50
        self.score = 0


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
            self.score += 1
            for pipe in self.pipes:
                if pipe == self.collosion_pipe: continue
                self.collosion_pipe = pipe
                break

        if self.bird_x < self.collosion_pipe.x + self.collosion_pipe.width and \
           self.collosion_pipe.x < self.bird_x + self.bird_width:
            for bird in self.birds:
                if bird.is_alive:
                    bird.collision_detection(self.collosion_pipe, self.score)


    def draw_game(self):
        for bird in self.birds:
            bird.draw()
        for pipe in self.pipes:
            pipe.draw()

        score = Font.render(f"Gen: {FlappyBird.generation}\nScore: {self.score}", False, WHITE, None)
        screen.blit(score, score.get_rect(center=(100, 60)))


    def make_descision(self):
        for bird in self.birds:
            if bird.is_alive:
                bird.make_descision(self.collosion_pipe)


    def all_dead(self) -> bool:
        for bird in self.birds:
            if bird.is_alive: return False
        return True
    

    def next_gen(self) -> "FlappyBird":
        self.birds.sort(key=lambda b:-b.score)
        new_gen = []
        new_gen += [Bird(bird.brain) for bird in self.birds[:5]]
        new_gen += [Bird(bird.brain.add_randomness(0.02)) for bird in self.birds[:5]]
        new_gen += [Bird(bird.brain.add_randomness(0.05)) for bird in self.birds[:5]]
        new_gen += [Bird(NeuralNetwork(2, 2, 2)) for _ in range(5)]
        for _ in range(10):
            b1, b2 = random.choices(self.birds[:10], [bird.score + 1 for bird in self.birds[:10]], k=2)
            new_gen.append(Bird(NeuralNetwork.crossover(b1.brain, b2.brain).add_randomness(random.random())))
        for _ in range(10):
            b1, b2 = random.choices(self.birds[:10], [bird.score + 1 for bird in self.birds[:10]], k=2)
            new_gen.append(Bird(NeuralNetwork.crossover(b1.brain, b2.brain).add_randomness(random.random() / 10)))
        for _ in range(10):
            b1, b2 = random.choices(self.birds[:10], [bird.score + 1 for bird in self.birds[:10]], k=2)
            new_gen.append(Bird(NeuralNetwork.crossover(b1.brain, b2.brain).add_randomness(min(1, 1-b1.score+b2.score/10))))

        return FlappyBird(new_gen)



game = FlappyBird([Bird(NeuralNetwork(2, 2, 2)) for _ in range(50)])


running = True

while running:
    mousex, mousey = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)
    game.draw_game()
    game.calculate_frame()
    game.make_descision()
    pygame.display.flip()
    clock.tick(60)

    if game.all_dead():
        FlappyBird.generation += 1
        game = game.next_gen()



pygame.quit()
sys.exit()



