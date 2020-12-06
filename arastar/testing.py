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
# import time # for testing time taken 

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

def arastar(draw, graph, start, goal):
    G = 1000000
    delta_w = 50
    w = 1000
    g = {}
    for row in graph:
        for node in row:
            g[node] = float("inf")
    g[start] = 0
    origin = {}
    f = {}
    for row in graph:
        for node in row:
            f[node] = float("inf")
    f[start] = w*heuristic(start.getPosition(), goal.getPosition())
    incumbent = None
    ezOpenSet = {start}
    while len(ezOpenSet) != 0 and w:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        new_solution, cost_new_solution = arastar_improve_solution(draw, graph, start, goal, ezOpenSet, w, G, incumbent, g, origin, f)
        if new_solution is not None:
            G = cost_new_solution
            incumbent = new_solution
        w -= delta_w
    reconstructPath(incumbent, goal, draw)

def arastar_improve_solution(draw, graph, start, goal, ezOpenSet, w, G, incumbent, g, origin, f):
    openPQ = PriorityQueue()
    count = 0
    openPQ.put((0, count, start))
    while not openPQ.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        n = openPQ.get()[2]
        if n in ezOpenSet:
            ezOpenSet.remove(n)
        if G <= f[n]:
            return None, None
        for successor_nprime in n.neighbors:
            if successor_nprime not in ezOpenSet and g[n] + 1 < g[successor_nprime]:
                f[successor_nprime] = (g[n]+1)+(w*heuristic(successor_nprime.getPosition(), goal.getPosition()))
                g[successor_nprime] = g[n] + 1
                count = count + 1
                openPQ.put((f[successor_nprime], count, successor_nprime))
                successor_nprime.color = GREEN
                if f[successor_nprime] < G:
                    origin[successor_nprime] = n
                    if successor_nprime == goal:
                        cost_new_solution = g[successor_nprime]
                        return origin,cost_new_solution
                    ezOpenSet.add(successor_nprime)




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

def main(environment, size):
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

    a = random.randrange(1, entries//1.8)
    b = random.randrange(1, entries//1.8)

    c = random.randrange(1, entries//1.8)
    d = random.randrange(1, entries//1.8)

    start = graph[a][b]
    graph[a][b].color = ORANGE

    goal = graph[c][d]
    graph[c][d].color = BLUE

    if entries == 100:
        if density == 10:
            for i in range(1000):
                x = random.randrange(1, entries)
                y = random.randrange(1, entries)
                if x != a and x != c and y != b and y != d:
                    graph[x][y].makeBarrier()
                # graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()
        elif density == 20:
            for i in range(2000):
                x = random.randrange(1, entries)
                y = random.randrange(1, entries)
                if x != a and x != c and y != b and y != d:
                    graph[x][y].makeBarrier()
                # graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()
        elif density == 30:
            for i in range(3000):
                x = random.randrange(1, entries)
                y = random.randrange(1, entries)
                if x != a and x != c and y != b and y != d:
                    graph[x][y].makeBarrier()
                # graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()

    elif entries == 200:
        if density == 10:
            for i in range(4000):
                x = random.randrange(1, entries)
                y = random.randrange(1, entries)
                if x != a and x != c and y != b and y != d:
                    graph[x][y].makeBarrier()
                # graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()
        elif density == 20:
            for i in range(8000):
                x = random.randrange(1, entries)
                y = random.randrange(1, entries)
                if x != a and x != c and y != b and y != d:
                    graph[x][y].makeBarrier()
                # graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()
        elif density == 30:
            for i in range(12000):
                x = random.randrange(1, entries)
                y = random.randrange(1, entries)
                if x != a and x != c and y != b and y != d:
                    graph[x][y].makeBarrier()
                # graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()

    elif entries == 300:
        if density == 10:
            for i in range(9000):
                x = random.randrange(1, entries)
                y = random.randrange(1, entries)
                if x != a and x != c and y != b and y != d:
                    graph[x][y].makeBarrier()
                # graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()
        elif density == 20:
            for i in range(18000):
                x = random.randrange(1, entries)
                y = random.randrange(1, entries)
                if x != a and x != c and y != b and y != d:
                    graph[x][y].makeBarrier()
                # graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()
        elif density == 30:
            for i in range(27000):
                x = random.randrange(1, entries)
                y = random.randrange(1, entries)
                if x != a and x != c and y != b and y != d:
                    graph[x][y].makeBarrier()
                # graph[random.randrange(1, entries)][random.randrange(1, entries)].makeBarrier()






    while True:
        draw(environment, graph, entries, size)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            for row in graph:
                for node in row:
                    node.updateNeighbors(graph)
            # begin = time.time() # for testing time taken 
            arastar(lambda: draw(environment, graph, entries, size), graph, start, goal)
            # end = time.time() # for testing time taken 
            # print(end-begin) # for testing time taken 
            break


main(environment, 800)
