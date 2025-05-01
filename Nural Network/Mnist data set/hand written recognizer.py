import pygame
import sys
import numpy
from NeuralNetwork import NeuralNetwork


nn = NeuralNetwork(28 * 28, 256, 128, 64, 32, 10)
nn.load_network('mnist')



pygame.init()
clock = pygame.time.Clock()
WIDTH, HEIGHT = 700, 700
PADDING = 50

Font=pygame.font.SysFont('timesnewroman',  25)
WHITE = (255, 255, 255)
BLACK = (25, 25, 25)
GRAY = (100, 100, 100)


screen = pygame.display.set_mode((WIDTH * 2, HEIGHT))
pygame.display.set_caption("")
screen.fill(BLACK)






class handler(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.list = numpy.zeros((height, width))

    def draw(self):
        r_w = WIDTH // self.width
        r_h = HEIGHT // self.height

        for y in range(self.height):
            for x in range(self.width):
                l = self.list[y][x] * 255
                color = (l, l, l)
                pygame.draw.rect(screen, color, pygame.Rect(x * r_w, y * r_h, r_w, r_h))
                pygame.draw.rect(screen, GRAY, pygame.Rect(x * r_w, y * r_h, r_w, r_h), 1)

        pygame.display.flip()


    def add_pixel(self, mx, my):
        x, y = (mx * self.width) // WIDTH, (my * self.height) // HEIGHT
        try:
            self.list[y][x] += 1
            for i in range(-1, 2):
                for j in range(-1, 2):
                    self.list[y - i][x - j] += 0.1
            for i in range(-2, 3):
                for j in range(-2, 3):
                    self.list[y - i][x - j] += 0.01
        except:
            pass
        self.list = numpy.clip(self.list, 0, 1)



    def remove_pixel(self, mx, my):
        x, y = (mx * self.width) // WIDTH, (my * self.height) // HEIGHT
        self.list[y][x] = 0


    def predict(self):
        flattened = self.list.flatten()
        probabilities = nn.forward(flattened)[0]
        self.draw_probabilities(probabilities)


    def draw_probabilities(self, probabilities):
        screen.fill(BLACK)
        self.draw()

        x = WIDTH / 11
        for i in range(10):
            percent = probabilities[i]
            pygame.draw.rect(screen, (180, 180, 200), pygame.Rect(WIDTH + i * x + x / 2, HEIGHT * 0.9 - HEIGHT * 0.8 * percent, x * 0.9, HEIGHT * 0.8 * percent))

            pygame.draw.rect(screen, WHITE, pygame.Rect(WIDTH + i * x + x / 2, HEIGHT - HEIGHT * 0.9, x * 0.9, HEIGHT * 0.8), 1)

            letter1 = Font.render(str(i), False, WHITE, None)
            screen.blit(letter1, letter1.get_rect(center=(WIDTH + i * x + x, HEIGHT - HEIGHT * 0.05)))
            
        pygame.display.flip()
    

    def reset(self):
        self.list = numpy.zeros((self.height, self.width))
        self.draw()




handle = handler(28, 28)
handle.draw()









running = True
holding_right = False
holding_left = False

while running:
    mousex, mousey = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                holding_right = True
            if event.button == 3:
                holding_left = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                holding_right = False
            if event.button == 3:
                holding_left = False

        elif holding_right and event.type == pygame.MOUSEMOTION:
            handle.add_pixel(mousex, mousey)
            handle.draw()

        elif holding_left and event.type == pygame.MOUSEMOTION:
            handle.remove_pixel(mousex, mousey)
            handle.draw()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                handle.predict()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                handle.reset()
                



    clock.tick(75)



pygame.quit()
sys.exit()



