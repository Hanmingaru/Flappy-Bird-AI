import pygame
from pygame import Rect
import random
import os
import time
import numpy as np 
import nn

pygame.init()

WIDTH, HEIGHT =  700,700
pygame.display.set_caption("Flappy Bird")

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
FPS = 60
FONT = pygame.font.SysFont('arial.ttf', 32)
WHITE = (255,255,255)
BLACK = (0,0,0)
LIGHT_BLUE = (137,209,254)
RED = (255,0,0)
POPULATION_SIZE = 300
NUM_GENERATIONS = 500

PIPE_TOP_IMAGE = pygame.image.load(os.path.join("assets","pipe_top.png"))
PIPE_BOT_IMAGE = pygame.image.load(os.path.join("assets","pipe_bot.png"))
PIPE_WIDTH = PIPE_TOP_IMAGE.get_rect().width
PIPE_HEIGHT = PIPE_TOP_IMAGE.get_rect().height
PIPE_SPACING = 200
PIPE_SPEED = 2
PIPE_RATE = 300

BIRD_IMAGE = pygame.image.load(os.path.join("assets","flappy.png"))
BIRD_RADIUS = 10
BIRD_YVELOCITY = .8
BIRD_JUMP = -12

bird_group = pygame.sprite.Group()
top_pipe_group = pygame.sprite.Group()
bot_pipe_group = pygame.sprite.Group()


#Global variables :(
num_alive = 0
cur_gen = 1
avg_fit = 0.0
dist_travel = 0
#Bird class has x and y positions and yvelocity to control up and down speeds
class Bird(pygame.sprite.Sprite):

    def __init__(self, *args) -> None:
        super().__init__()
        self.x = BIRD_RADIUS * 10
        self.y = HEIGHT//2
        self.y_vel = 0
        self.fitness = 0
        self.norm_fitness = 0
        self.image = BIRD_IMAGE
        self.rect = self.image.get_rect(x = self.x, y = self.y)
        if len(args) == 1:
            self.brain = args[0]
        elif len(args) == 0:
            self.brain = nn.NeuralNetwork(2,2,1)

    def update(self, pipes, birds, dead_birds):
    
        global num_alive
        self.fitness += 1
        self.rect.y += self.y_vel
        self.y += self.y_vel
        self.y_vel += BIRD_YVELOCITY

        x1 = (self.x - pipes[0].x) / WIDTH
        x2 = (self.rect.centery - pipes[0].center)  / HEIGHT
        

        if self.brain.predict(np.array([[x1],[x2]])) > .5:
            self.y_vel = BIRD_JUMP
        if pygame.sprite.spritecollide(self, top_pipe_group, False) or pygame.sprite.spritecollide(self, bot_pipe_group, False) or self.rect.y > HEIGHT or self.rect.y < 0:
            num_alive -= 1
            dead_birds.append(birds.pop(0))
            self.kill()
    def copy(self):
        return Bird(self.brain)
# Create one pipe with the spacing at a random location
# Find the X positions of the top and bottom half
class Pipe(pygame.sprite.Sprite):
    def __init__(self, rand, image, is_top) -> None:
        super().__init__()
        space = abs( (not is_top) * HEIGHT - (-1 * is_top) * (rand - PIPE_SPACING//2))
        cropped= pygame.Surface((PIPE_WIDTH, space))
        cropped.blit(image, (0,0), (0,is_top * (PIPE_HEIGHT - space), PIPE_WIDTH, space))
        self.is_top = is_top
        self.x = WIDTH
        self.center = rand
        self.image = cropped
        self.rect = cropped.get_rect(x = self.x, y = (not is_top)*(rand+PIPE_SPACING//2))
    
    def update(self, pipes):
            self.rect.x -= PIPE_SPEED
            self.x -= PIPE_SPEED
            if(self.rect.x + PIPE_WIDTH <= 0):
                self.kill()
                #print("pipe killed")
                if self.is_top == True: 
                    pipes.pop(0)

def find_total_fitness(birds):
    s = 0.0;
    for bird in birds:
        s+=bird.fitness
    return s

def find_average_fitness(birds):
    return find_total_fitness(birds) / len(birds)

def select_bird(birds):
    rand = random.random();
    index = 0;
    while rand > 0:
        rand -= birds[index].norm_fitness
        index += 1
        #print(f'random is {rand} and index is {index}')
    index -= 1
    return birds[index]

def make_child(p1_brain, p2_brain):
    (w0,b0, w1,b1) = nn.crossover(p1_brain,p2_brain)
    bird = Bird()
    bird.brain = nn.NeuralNetwork(w0, b0, w1, b1)
    return bird

def clear(pipes):
    bird_group.empty()
    top_pipe_group.empty()
    bot_pipe_group.empty()
    pipes.clear()

def draw(center):
    global num_alive, cur_gen, avg_fit
    WIN.fill(LIGHT_BLUE)
    top_pipe_group.draw(WIN)
    bot_pipe_group.draw(WIN)
    bird_group.draw(WIN)

    text1 = FONT.render("Number alive: " + str(num_alive), True, RED)      
    text2 = FONT.render("Current Generations: " + str(cur_gen), True, RED)      
    text3 = FONT.render("Average Fitness: " + str(avg_fit), True, RED)      
    text4 = FONT.render("Distance: " + str(dist_travel), True, RED)      
    WIN.blit(text1, text1.get_rect(x = 0, y = HEIGHT-120))
    WIN.blit(text2, text2.get_rect(x = 0, y = HEIGHT-90))
    WIN.blit(text3, text3.get_rect(x = 0,  y= HEIGHT-60))
    WIN.blit(text4, text4.get_rect(x = 0,  y= HEIGHT-30))

    pygame.draw.circle(WIN, BLACK, center, 10)
    pygame.display.update()

def normalize(birds):
    for bird in birds:
        bird.fitness = bird.fitness ** 2
    s = find_total_fitness(birds)
    for bird in birds:
        bird.norm_fitness = bird.fitness / s

def play():
    global num_alive, cur_gen, avg_fit, dist_travel
    run_game = True
    current_generation = 1
    current_population = []
    pipes = []
    for i in range(POPULATION_SIZE):
            bird = Bird()
            current_population.append(bird)
            bird_group.add(bird)

    while run_game == True and current_generation <= NUM_GENERATIONS:
        clock = pygame.time.Clock()
        pipe_place = PIPE_RATE
        run_round = True;
        dead_birds = []
        num_alive = POPULATION_SIZE
        dist_travel = 0
        #print(current_population[0].brain.weights0)
        while len(current_population) > 0 and run_round == True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run_round = False
                    run_game = False
            if(pipe_place % PIPE_RATE == 0):
                rand = random.randint(HEIGHT//5,HEIGHT-HEIGHT//5)
                pipe = Pipe(rand, PIPE_TOP_IMAGE, 1)
                top_pipe_group.add(pipe)
                bot_pipe_group.add(Pipe(rand, PIPE_BOT_IMAGE, 0))
                pipes.append(pipe)
            current_center = (pipes[0].x + PIPE_WIDTH//2, pipes[0].center) if len(pipes) > 0 else 0

            dist_travel += 1
            pipe_place += 1
            top_pipe_group.update(pipes)
            bot_pipe_group.update(pipes)
            bird_group.update(pipes, current_population, dead_birds)
            draw(current_center)
            clock.tick(FPS)
        
        current_generation += 1
        cur_gen += 1
        clear(pipes)
        current_population = []
        
        #Normalize
        normalize(dead_birds)

        totalf = find_total_fitness(dead_birds)
        avg_fit = totalf / POPULATION_SIZE
        
        #Selection and crossover
        for x in range(POPULATION_SIZE):
            parent1 = select_bird(dead_birds)
            parent2 = select_bird(dead_birds)
            child = make_child(parent1.brain, parent2.brain)
            current_population.append(child)
        
        #Mutation
        for new_bird in current_population:
            new_bird.brain.mutate();
            bird_group.add(new_bird)
        
        
    pygame.quit()

if __name__ == "__main__":
    play()