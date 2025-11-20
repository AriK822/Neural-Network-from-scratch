import pygame
import sys

import random
import math
from enum import Enum, auto

from brain import NeuralNetwork
from time import time



pygame.init()
clock = pygame.time.Clock()
WIDTH, HEIGHT = 700, 700
PADDING = 50

Font=pygame.font.SysFont('timesnewroman',  50)
WHITE = (255, 255, 255)
BLACK = (25, 25, 25)
GRAY = (100, 100, 100)
POWDER_BLUE = (80, 80, 150)


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("")

screen.fill(BLACK)
pygame.display.flip()





class CarState(Enum):
    DRIVING = auto()
    LOST = auto()
    WON = auto()



class Car: 
    def __init__(self, speed = 3, size = 20, color = None, brain = None):
        self.x = WIDTH / 2
        self.y = 0
        self.speed_x = 0
        self.speed_y = 0
        self.size = size
        self.speed = speed
        self.angle = math.pi / 2
        self.score = 0
        self.progress_percent = 0
        self.state = CarState.DRIVING
        self.ray_trace_values = []
        if not brain:
            self.brain = NeuralNetwork(7, 10, 3)
        else:
            self.brain = brain
        if not color:
            self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            self.color = color


    def calculate_frame(self):
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed


    def __repr__(self) -> str:
        return f"Car at x: {self.x}, y: {self.y}.   State: {
            "Driving" if self.state == CarState.DRIVING else
            "Lost" if self.state == CarState.LOST else "WON"}"




class DrivingGame:
    def __init__(self, race_length = 3, dificaulty = 3, road_width = 200, time_limit = 15):
        self.race_length = race_length
        self.dificaulty = dificaulty
        self.road_width = road_width
        self.camera_y = 0
        self.start_time = time()
        self.time_limit = time_limit
        self.gen_number = 1

    
    def generate_race(self):
        self.race_path = [(WIDTH // 2, 0)]
        for i in range(1, self.race_length * self.dificaulty + 1):
            this_height = i * (HEIGHT // self.dificaulty)
            self.race_path.append(
                (
                    random.randint(self.road_width // 2, WIDTH - self.road_width // 2),
                    this_height
                )
            )
    

    def init_cars(self, cars:list[Car]):
        self.cars = cars


    def draw_race_path(self):
        pygame.draw.lines(screen, WHITE, False, [(x - self.road_width / 2, y - self.camera_y) for (x, y) in self.race_path])
        pygame.draw.lines(screen, WHITE, False, [(x + self.road_width / 2, y - self.camera_y) for (x, y) in self.race_path])

    
    def draw_cars(self):
        for car in self.cars:
            car:Car

            pygame.draw.polygon(screen, car.color, [
                (car.x + math.cos(car.angle - math.pi / 4) * car.size, car.y + math.sin(car.angle - math.pi / 4) * car.size - self.camera_y),
                (car.x + math.cos(car.angle + math.pi / 4) * car.size, car.y + math.sin(car.angle + math.pi / 4) * car.size - self.camera_y),
                (car.x + math.cos(car.angle + math.pi * 3 / 4) * car.size, car.y + math.sin(car.angle + math.pi * 3 / 4) * car.size - self.camera_y),
                (car.x + math.cos(car.angle + math.pi * 5 / 4) * car.size, car.y + math.sin(car.angle + math.pi * 5 / 4) * car.size - self.camera_y),
            ])


    def handle_collisions(self):
        def point_on_segment(px, py, x1, y1, x2, y2, eps=1e-9):
            cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
            if abs(cross) > eps:
                return False

            if min(x1, x2) - eps <= px <= max(x1, x2) + eps and min(y1, y2) - eps <= py <= max(y1, y2) + eps:
                return True
            return False


        def point_in_polygon(point, polygon, eps=1e-9):
            px, py = point
            n = len(polygon)
            if n == 0:
                return False

            inside = False
            for i in range(n):
                x1, y1 = polygon[i]
                x2, y2 = polygon[(i + 1) % n]

                if point_on_segment(px, py, x1, y1, x2, y2, eps):
                    return True

                if ((y1 > py) != (y2 > py)):
                    x_int = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                    if px < x_int - eps:
                        inside = not inside

            return inside
        

        car_size = 20
        right_side_raod = [(x + self.road_width / 2 - car_size / 2, y) for (x, y) in self.race_path]
        right_side_raod.append((WIDTH, HEIGHT * self.race_length))
        right_side_raod.append((WIDTH, 0))
        left_side_raod = [(x - self.road_width / 2 + car_size / 2, y) for (x, y) in self.race_path]
        left_side_raod.append((0, HEIGHT * self.race_length))
        left_side_raod.append((0, 0))

        for car in self.cars:
            if car.state != CarState.DRIVING: continue

            if car.y >= HEIGHT * self.race_length: car.state = CarState.WON

            if car.size != car_size:
                car_size = car.size
                right_side_raod = [(x + self.road_width / 2 - car_size / 2, y) for (x, y) in self.race_path]
                right_side_raod.append((WIDTH, HEIGHT * self.race_length))
                right_side_raod.append((WIDTH, 0))
                left_side_raod = [(x - self.road_width / 2 + car_size / 2, y) for (x, y) in self.race_path]
                left_side_raod.append((0, HEIGHT * self.race_length))
                left_side_raod.append((0, 0))

            if point_in_polygon((car.x, car.y), right_side_raod):
                car.state = CarState.LOST
                continue
            if point_in_polygon((car.x, car.y), left_side_raod):
                car.state = CarState.LOST
                continue


    def ray_trace(self):
        def ray_segment_intersection(px, py, dx, dy, x1, y1, x2, y2, eps=1e-9):
            sx = x2 - x1
            sy = y2 - y1

            denom = dx * sy - dy * sx
            if abs(denom) < eps:
                return None

            t = ((x1 - px) * sy - (y1 - py) * sx) / denom
            u = ((x1 - px) * dy - (y1 - py) * dx) / denom

            if t >= 0 and 0 <= u <= 1:
                return t

            return None


        def ray_polygon_intersection(point, polygon, angle):
            px, py = point

            dx = math.cos(angle)
            dy = math.sin(angle)

            closest_t = None
            closest_point = None

            n = len(polygon)
            for i in range(n):
                x1, y1 = polygon[i]
                x2, y2 = polygon[(i + 1) % n]

                t = ray_segment_intersection(px, py, dx, dy, x1, y1, x2, y2)
                if t is not None:
                    if closest_t is None or t < closest_t:
                        closest_t = t
                        closest_point = (px + t * dx, py + t * dy)

            return closest_point
        

        road = [(x + self.road_width / 2, y) for (x, y) in self.race_path]
        road.append((WIDTH, road[-1][1] + HEIGHT))
        road.append((0, road[-1][1] + HEIGHT))
        road += [(x - self.road_width / 2, y) for (x, y) in self.race_path][::-1]

        for car in self.cars:
            if car.state != CarState.DRIVING: continue
            ray_trace_values = []
            look_angles = [0, math.pi / 6, -math.pi / 6, math.pi / 3, -math.pi / 3, math.pi / 2, -math.pi / 2]

            for angle_shift in look_angles:
                collision = ray_polygon_intersection((car.x, car.y), road, car.angle + angle_shift)
                if collision: 
                    ray_trace_values.append(min(math.sqrt((car.x-collision[0])**2 + (car.y-collision[1])**2) / HEIGHT, 1))
                    pygame.draw.line(screen, POWDER_BLUE, (car.x, car.y - self.camera_y), (collision[0], collision[1] - self.camera_y))
                else: ray_trace_values.append(1)

            car.ray_trace_values = ray_trace_values


    def calculate_frame(self):
        self.handle_collisions()
        self.ray_trace()

        self.camera_y = 0
        for car in self.cars:
            car:Car
            if car.state != CarState.LOST:
                car.calculate_frame()
                car.progress_percent = car.y / (HEIGHT * self.race_length) # type: ignore
            
            self.camera_y = max(self.camera_y, car.y - HEIGHT / 2)

        self.camera_y = min(max(0, self.camera_y), self.race_length * HEIGHT - HEIGHT)


    @property
    def ended(self) -> bool:
        if time() - self.start_time > self.time_limit: return True

        ended = True
        for car in self.cars:
            if car.state == CarState.DRIVING:
                ended = False
                break

        return ended
    

    def draw_generation_number(self):
        letter1 = Font.render(str(self.gen_number), False, WHITE, None)
        screen.blit(letter1, letter1.get_rect(center=(WIDTH - 50,  50)))




def next_generation(cars:list[Car]) -> list[Car]:
    cars.sort(key = lambda car:-car.y)
    elite_cars = cars[:10]

    new_gen = [Car(brain=elite_cars[0].brain), Car(brain=elite_cars[1].brain), Car(), Car()]

    for car in elite_cars:
        new_gen.append(Car(brain=car.brain.add_randomness(max(0.05, car.progress_percent))))

    for _ in range(36):
        sample = random.choices(elite_cars, weights = [car.progress_percent for car in elite_cars], k = 2)
        new_gen.append(Car(
            brain=NeuralNetwork.crossover(sample[0].brain, sample[1].brain).add_randomness(
                max(0.05, (sample[0].progress_percent + sample[1].progress_percent) / 2)
            )
        ))


    return new_gen



game = DrivingGame()
game.generate_race()

cars = [Car() for _ in range(50)]
game.init_cars(cars)


running = True


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    screen.fill(BLACK)
    game.calculate_frame()
    game.draw_race_path()
    game.draw_cars()
    game.draw_generation_number()
    pygame.display.flip()


    for car in cars:
        if car.state != CarState.DRIVING: continue
        ans = car.brain.forward(car.ray_trace_values)[0] # type: ignore
        choice = "l" if ans.max() == ans[0] else "r" if ans.max() == ans[2] else "s"
        if choice == "r": car.angle += math.pi / 60
        if choice == "l": car.angle -= math.pi / 60


    if game.ended:
        game.start_time = time()
        cars = next_generation(cars)
        game.init_cars(cars)
        game.gen_number += 1


    clock.tick(60)



pygame.quit()
sys.exit()



