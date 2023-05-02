import math
import random
import sys
import os
import neat
import networkx as nx
import matplotlib.pyplot as plt
import pygame
import graphviz
import numpy as np

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 60    
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255, 255) 

current_generation = 0 

class Car:

    def __init__(self):
        self.sprite = pygame.image.load('car.png').convert() 
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite 
        self.position = [830, 920] 

        self.angle = 0
        self.speed = 0
        self.speed_set = False 

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2] 
        self.radars = [] 
        self.drawing_radars = [] 
        self.alive = True 
        
        self.distance = 0 
        self.time = 0 

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position) 
        self.draw_radar(screen) 

    def draw_radar(self, screen):
        # Draw sensorss // radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])
    
    def update(self, game_map):

        if not self.speed_set:
            self.speed = 20
            self.speed_set = True
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH) #- 120)
        self.distance += self.speed
        self.time += 1
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH) #- 120)

        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check collisions // clear radrs
        self.check_collision(game_map)
        self.radars.clear()

        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        # Distances to border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image


def run_simulation(genomes, config):
    
    furthest_car_index = 0
    # Empty collectinos
    nets = []
    cars = []

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

    # Neural network strcuture
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    # Clock 
    # Font // loaading map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load('map2.png').convert() # Convert Speeds Up A Lot

    global current_generation
    current_generation += 1

    # Counter (? needs fixing)
    counter = 0

    while True:
        # Exit pygame on quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10 # Left
            elif choice == 1:
                car.angle -= 10 # Right
            elif choice == 2:
                if(car.speed - 2 >= 12):
                    car.speed -= 2 # Slow down
            else:
                car.speed += 2 # Speed up
        
        # Check If Car Is Still Alive
        # Increase fitess if yes, else, reset
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                if car.distance > cars[furthest_car_index].distance:
                    furthest_car_index = i
                genomes[i][1].fitness += car.get_reward()
               

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40: # Stop after ~20 secs (will crash otherwise)
            break

        # Draw map
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)
        
        # Display 
        text = generation_font.render("Generation: " + str(current_generation), True, (0,0,0))
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)
        draw_neural_network(genomes[furthest_car_index][1], config, screen)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60) # 60 FPS

def draw_neural_network(genome, config, surface):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    input_y = 100
    hidden_y = 200
    output_y = 300
    node_radius = 15
    layer_spacing = 100

    for i in range(len(net.input_nodes)):
        x = i * layer_spacing + layer_spacing
        y = input_y
        pygame.draw.circle(surface, (0, 0, 0), (x, y), node_radius)

    for i in range(len(net.node_evals)):
        x = i * layer_spacing + layer_spacing
        y = hidden_y
        pygame.draw.circle(surface, (0, 0, 0), (x, y), node_radius)

    for i in range(len(net.output_nodes)):
        x = i * layer_spacing + layer_spacing
        y = output_y
        pygame.draw.circle(surface, (0, 0, 0), (x, y), node_radius)

    for connection in genome.connections.values():
        if connection.enabled:
            in_node_pos = None
            out_node_pos = None

            if connection.key[0] in net.input_nodes:
                index = net.input_nodes.index(connection.key[0])
                x = index * layer_spacing + layer_spacing
                y = input_y
                in_node_pos = (x, y)

            for i in range(len(net.node_evals)):
                if net.node_evals[i][0] == connection.key[0]:
                    x = i * layer_spacing + layer_spacing
                    y = hidden_y
                    in_node_pos = (x, y)
                    break

            for i in range(len(net.node_evals)):
                if net.node_evals[i][0] == connection.key[1]:
                    x = i * layer_spacing + layer_spacing
                    y = hidden_y
                    out_node_pos = (x, y)
                    break

            if connection.key[1] in net.output_nodes:
                index = net.output_nodes.index(connection.key[1])
                x = index * layer_spacing + layer_spacing
                y = output_y
                out_node_pos = (x, y)

            if in_node_pos is not None and out_node_pos is not None:
                color = (255, 0, 0) if connection.weight < 0 else (0, 255, 0)
                width = int(abs(connection.weight) * 5)
                pygame.draw.line(surface, color, in_node_pos, out_node_pos, width)


if __name__ == "__main__":
    
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    #  maximum genertion
    population.run(run_simulation, 1000)
