# References: 
# **** https://www.youtube.com/watch?v=JtiK0DOeI4A&ab_channel=TechWithTim **** 
# https://www.redblobgames.com/pathfinding/a-star/introduction.html 
# https://www.youtube.com/watch?v=P4d0pUkAsz4&ab_channel=RANJIRAJ
# https://www.youtube.com/watch?v=-L-WgKMFuhE&ab_channel=SebastianLague 
# https://www.simplifiedpython.net/a-star-algorithm-python-tutorial/ 
# https://en.wikipedia.org/wiki/A*_search_algorithm 
# https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2 
# https://www.youtube.com/watch?v=ob4faIum4kQ&ab_channel=TrevorPayne 

import pygame
import math
from queue import PriorityQueue
import random

environment = pygame.display.set_mode((800, 800))

RED = (255, 0, 0) # closed 
GREEN = (0, 255, 0) # open 
BLUE = (0, 0, 255) # goal
ORANGE = (255, 165, 0) # start 
BLACK = (0, 0, 0) # barrier 
WHITE = (255, 255, 255) # entry 
PURPLE = (128, 0, 128) # path 
GREY = (128, 128, 128) # environment 




class Node:
    def __init__(self, row, column, size, numberOfentries):
        self.row = row
        self.column = column
        self.i = row * size 
        self.j = column * size 
        self.color = WHITE
        self.neighbors = []
        self.size = size 
        self.numberOfentries = numberOfentries

    def getPosition(self):
        return self.row, self.column

    def makeBarrier(self):
        self.color = BLACK

    def draw(self, environment):
        pygame.draw.rect(environment, self.color, (self.i, self.j, self.size, self.size))

    def updateNeighbors(self, graph):
        self.neighbors = []
        if self.row < self.numberOfentries - 1 and not graph[self.row+1][self.column].color == BLACK:
            self.neighbors.append(graph[self.row + 1][self.column])
        if self.row > 0 and not graph[self.row - 1][self.column].color == BLACK:
            self.neighbors.append(graph[self.row - 1][self.column])
        if self.column < self.numberOfentries - 1 and not graph[self.row][self.column + 1].color == BLACK:
            self.neighbors.append(graph[self.row][self.column + 1])
        if self.column > 0 and not graph[self.row][self.column - 1].color == BLACK:
            self.neighbors.append(graph[self.row][self.column - 1])

def heuristic(pointA, pointB):
    i1, j1 = pointA
    i2, j2 = pointB
    return abs(i1 - i2) + abs(j1 - j2)

def reconstructPath(origin, current, draw):
    while current in origin:
        current = origin[current]
        current.color = PURPLE
    draw()

def A_Star(draw, graph, start, goal):
    openPQ = PriorityQueue()
    count = 0
    openPQ.put((0, count, start)) 
    origin = {}
    g = {}
    for row in graph:
        for node in row: 
            g[node] = float("inf")
    g[start] = 0
    f = {}
    for row in graph:
        for node in row:
            f[node] = float("inf")
    f[start] = heuristic(start.getPosition(), goal.getPosition())

    ezOpenSet = {start}

    while not openPQ.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = openPQ.get()[2]
        ezOpenSet.remove(current)

        if current == goal:
            reconstructPath(origin, goal, draw)
            goal.color = BLUE
            return True
            
        for neighbor in current.neighbors:
            tentative_gScore = g[current] + 1

            if tentative_gScore < g[neighbor]:
                origin[neighbor] = current
                g[neighbor] = tentative_gScore
                f[neighbor] = tentative_gScore + heuristic(neighbor.getPosition(), goal.getPosition())
                if neighbor not in ezOpenSet:
                    count += 1
                    openPQ.put((f[neighbor], count, neighbor))
                    ezOpenSet.add(neighbor)
                    neighbor.color = GREEN

        # draw()

        if current != start:
            current.color = RED

    return False

def draw(environment, graph, entries, size):
    environment.fill(WHITE)
    
    for row in graph:
        for node in row:
            node.draw(environment)

    space = size // entries
    for i in range(entries):
        pygame.draw.line(environment, GREY, (0, i * space), (size, i * space))
        for j in range(entries):
            pygame.draw.line(environment, GREY, (j * space, 0), (j * space, size))
    pygame.display.update()

def main(environment):
    entries = int(input("enter grid size: "))
    density = int(input("enter density: "))

    if entries == 100:
        size = 800
    elif entries == 200:
        size = 1000
    elif entries == 300:
        size = 1200



    graph = []
    space = size // entries 
    for i in range(entries):
        graph.append([])
        for j in range(entries):
            node = Node(i, j, space, entries)
            graph[i].append(node)









    start = graph[10][90]
    graph[10][90].color = ORANGE

    goal = graph[95][45]
    graph[95][45].color = BLUE

    if entries == 100:
        if density == 10:
            for i in range(1000):
                graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()
        elif density == 20:
            for i in range(2000):
                graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()
        elif density == 30:
            for i in range(3000):
                graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()


    elif entries == 200:
        if density == 10:
            for i in range(4000):
                graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()
        elif density == 20:
            for i in range(8000):
                graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()
        elif density == 30:
            for i in range(12000):
                graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()


    elif entries == 300:
        if density == 10:
            for i in range(9000):
                graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()
        elif density == 20:
            for i in range(18000):
                graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()
        elif density == 30:
            for i in range(27000):
                graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()



    while True:
        draw(environment, graph, entries, size)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            for row in graph:
                for node in row:
                    node.updateNeighbors(graph)
            A_Star(lambda: draw(environment, graph, entries, size), graph, start, goal)
            break


main(environment)
