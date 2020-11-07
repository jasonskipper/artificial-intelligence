# first attempt  
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
import numpy as np
from queue import PriorityQueue

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

def cost(c, g, h):
    return (c-g)/h

# def boundedcost(draw, graph, start, goal, c):
#     open = []
#     closed = []
#     g = []
#     open.append(start)
#     n = []

#     while not open.empty(): 
#         for nprime in n:
#             if nprime in open or nprime in closed and g[nprime] <= cost(n, nprime):
#                 continue
#             g[nprime] = g[n] + cost(n, nprime)
#             if g[nprime] + h[nprime] >= c:
#                 continue
#             if nprime == goal:
#                 reconstructPath(origin, goal, draw)
#                 goal.color = BLUE
#                 return True
#             if nprime in open:
#                 print("placeholder  ")
#             else:
#                 open.append(nprime)
       
        

        






def A_Star(draw, graph, start, goal, constraint):
    # constraint = 150
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
                #  and f[neighbor] < constraint    
                # print("g: ", g[neighbor])
                # print("h: ", heuristic(neighbor.getPosition(), goal.getPosition()))
                # print("f: ", f[neighbor])
                # print("cost: ", cost(constraint, g[neighbor], heuristic(neighbor.getPosition(), goal.getPosition())))
                if neighbor not in ezOpenSet and f[neighbor] < constraint:
                    count += 1
                    openPQ.put((f[neighbor], count, neighbor))
                    ezOpenSet.add(neighbor)
                    neighbor.color = GREEN

        draw()

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

def main(environment, size):
    entries = 100
    graph = []
    space = size // entries 
    for i in range(entries):
        graph.append([])
        for j in range(entries):
            node = Node(i, j, space, entries)
            graph[i].append(node)






    # Constructing the environment from Figure 3.31: 
    # --------------------------------------------------------------------------------  
    start = graph[10][90]
    graph[10][90].color = ORANGE

    goal = graph[95][45]
    graph[95][45].color = BLUE
    




    def environmentA():
        # Horizontal Rectangle 
        # ------------------------------------- 
        # Vertices: 
        graph[16][95].makeBarrier() #HR1 
        graph[16][85].makeBarrier() #HR2 
        
        graph[46][85].makeBarrier() #HR3 
        graph[46][95].makeBarrier() #HR4 

        # left side of horizontal rectangle 
        for i in range(85, 95):
            graph[16][i].makeBarrier()

        # top side of horizontal rectangle 
        for i in range(17, 46):
            graph[i][85].makeBarrier() 

        # right side of horizontal rectangle
        for i in reversed(range(86, 95)):
            graph[46][i].makeBarrier()

        # bottom side of horizontal rectangle 
        for i in range(17, 46):
            graph[i][95].makeBarrier()
        # ------------------------------------- 







        # Pentagon 
        # -------------------------------------
        graph[25][75].makeBarrier() # P1 

        graph[14][70].makeBarrier() # P2 
        
        graph[10][56].makeBarrier() # P3 

        graph[20][45].makeBarrier() # P4 

        graph[30][56].makeBarrier() # P5  



        # bottom left side of pentagon  
        graph[24][74].makeBarrier()
        graph[23][74].makeBarrier()
        graph[22][73].makeBarrier()
        graph[21][73].makeBarrier()
        graph[20][72].makeBarrier()
        graph[19][72].makeBarrier()
        graph[18][71].makeBarrier()
        graph[17][71].makeBarrier()
        graph[16][70].makeBarrier()
        graph[15][70].makeBarrier()

        # top left side of pentagon 
        graph[10][55].makeBarrier()
        graph[11][55].makeBarrier()
        graph[11][54].makeBarrier()
        graph[12][54].makeBarrier()
        graph[13][53].makeBarrier()
        graph[13][52].makeBarrier()
        graph[14][51].makeBarrier()
        graph[15][51].makeBarrier()
        graph[16][50].makeBarrier()
        graph[16][49].makeBarrier()
        graph[17][48].makeBarrier()
        graph[17][47].makeBarrier()
        graph[18][47].makeBarrier()
        graph[19][46].makeBarrier()

        # left side of pentagon  
        graph[10][57].makeBarrier()
        graph[10][58].makeBarrier()
        graph[10][59].makeBarrier()
        graph[11][60].makeBarrier()
        graph[12][61].makeBarrier()
        graph[12][62].makeBarrier()
        graph[12][63].makeBarrier()
        graph[13][64].makeBarrier()
        graph[13][65].makeBarrier()
        graph[13][66].makeBarrier()
        graph[14][67].makeBarrier()
        graph[14][68].makeBarrier()
        graph[14][69].makeBarrier()  

        # top side of pentagon  
        graph[21][46].makeBarrier()
        graph[22][46].makeBarrier()
        graph[23][47].makeBarrier()
        graph[24][47].makeBarrier()
        graph[24][48].makeBarrier()
        graph[24][49].makeBarrier()
        graph[25][49].makeBarrier()
        graph[25][50].makeBarrier()
        graph[25][51].makeBarrier()
        graph[26][51].makeBarrier()
        graph[26][52].makeBarrier()
        graph[26][53].makeBarrier()
        graph[27][54].makeBarrier()
        graph[28][54].makeBarrier()
        graph[28][55].makeBarrier()
        graph[29][55].makeBarrier()
        graph[29][56].makeBarrier()

        # bottom right side of pentagon 
        graph[30][57].makeBarrier()
        graph[30][58].makeBarrier()
        graph[30][59].makeBarrier()
        graph[29][59].makeBarrier()
        graph[29][60].makeBarrier()
        graph[29][61].makeBarrier()
        graph[29][62].makeBarrier()
        graph[28][63].makeBarrier()
        graph[28][64].makeBarrier()
        graph[28][65].makeBarrier()
        graph[28][66].makeBarrier()
        graph[27][67].makeBarrier()
        graph[27][68].makeBarrier()
        graph[27][69].makeBarrier()
        graph[27][70].makeBarrier()
        graph[26][71].makeBarrier()
        graph[26][72].makeBarrier()
        graph[26][73].makeBarrier()
        graph[26][74].makeBarrier()




        









        # Isosceles Triangle 
        graph[37][55].makeBarrier() # IT1 
        graph[41][78].makeBarrier() # IT2 
        graph[33][78].makeBarrier() # IT3  

        # graph[36][56].makeBarrier()
        for i in range(56, 62):
            graph[36][i].makeBarrier()
        for i in range(62, 67):
            graph[35][i].makeBarrier()
        for i in range(67, 73):
            graph[34][i].makeBarrier()
        for i in range(73, 78):
            graph[33][i].makeBarrier()

        for i in range(56, 62):
            graph[38][i].makeBarrier()
        for i in range(62, 67):
            graph[39][i].makeBarrier()
        for i in range(67, 73):
            graph[40][i].makeBarrier()
        for i in range(73, 78):
            graph[41][i].makeBarrier()

        for i in range(34, 41):
            graph[i][78].makeBarrier()




        


        # Quadrilateral 
        graph[43][60].makeBarrier() # Q1 
        graph[43][44].makeBarrier() # Q2 
        graph[51][41].makeBarrier() # Q3 
        graph[56][48].makeBarrier() # Q4  

        for i in range(45, 60):
            graph[43][i].makeBarrier()
        graph[44][43].makeBarrier()
        graph[45][43].makeBarrier()
        graph[46][43].makeBarrier()
        graph[47][42].makeBarrier()
        graph[48][42].makeBarrier()
        graph[49][42].makeBarrier()
        graph[50][42].makeBarrier()

        graph[52][42].makeBarrier()
        graph[52][43].makeBarrier()
        graph[53][44].makeBarrier()
        graph[53][45].makeBarrier()
        graph[54][46].makeBarrier()
        graph[55][47].makeBarrier()

        graph[55][49].makeBarrier()
        graph[54][50].makeBarrier()
        graph[53][51].makeBarrier()
        graph[52][52].makeBarrier()
        graph[51][53].makeBarrier()
        graph[50][54].makeBarrier()
        graph[49][55].makeBarrier()
        graph[48][56].makeBarrier()
        graph[47][57].makeBarrier()
        graph[46][58].makeBarrier()
        graph[45][59].makeBarrier()
        graph[44][60].makeBarrier()
        

        # Right Triangle 
        graph[56][90].makeBarrier() #RT3 

        graph[66][83].makeBarrier()# RT2 

        graph[49][70].makeBarrier() #RT1 

        # left side 

        graph[56][91].makeBarrier()
        graph[56][89].makeBarrier()
        graph[55][89].makeBarrier()
        graph[55][88].makeBarrier()
        graph[55][87].makeBarrier()
        graph[54][87].makeBarrier()
        graph[53][87].makeBarrier()
        graph[53][86].makeBarrier()
        graph[53][85].makeBarrier()
        graph[53][84].makeBarrier()
        graph[53][83].makeBarrier()
        graph[52][82].makeBarrier()
        graph[52][81].makeBarrier()
        graph[52][80].makeBarrier()
        graph[52][79].makeBarrier()
        graph[51][78].makeBarrier()
        graph[51][77].makeBarrier()
        graph[51][76].makeBarrier()
        graph[51][75].makeBarrier()
        graph[50][74].makeBarrier()
        graph[50][73].makeBarrier()
        graph[49][72].makeBarrier()
        graph[49][71].makeBarrier()

        # right side   
        graph[50][70].makeBarrier()
        graph[51][71].makeBarrier()
        graph[52][72].makeBarrier()
        graph[53][73].makeBarrier()
        graph[54][73].makeBarrier()
        graph[55][74].makeBarrier()
        graph[56][75].makeBarrier()
        graph[57][76].makeBarrier()
        graph[58][77].makeBarrier()
        graph[59][78].makeBarrier()
        graph[60][79].makeBarrier()
        graph[61][79].makeBarrier()
        graph[62][80].makeBarrier()
        graph[63][81].makeBarrier()
        graph[64][82].makeBarrier()
        graph[65][82].makeBarrier()

        graph[65][84].makeBarrier() 
        graph[64][85].makeBarrier() 
        graph[63][86].makeBarrier() 
        graph[62][87].makeBarrier() 
        graph[61][88].makeBarrier() 
        graph[60][89].makeBarrier() 
        graph[59][90].makeBarrier() 
        graph[58][91].makeBarrier() 
        graph[57][92].makeBarrier() 



        


        # Vertical Rectangle 
        graph[62][46].makeBarrier() # R1 
        graph[62][75].makeBarrier() # R2 
        graph[77][46].makeBarrier() # R3 
        graph[77][75].makeBarrier() # R4

        for i in range(47, 75):
            graph[62][i].makeBarrier()
        for i in range(47, 75):
            graph[77][i].makeBarrier()
        for i in range(63, 77):
            graph[i][46].makeBarrier()
        for i in range(63, 77):
            graph[i][75].makeBarrier()

        

        # Hexagon  
        graph[79][78].makeBarrier() # H1 # highest 
        graph[74][83].makeBarrier() # H2 # top left 
        graph[74][88].makeBarrier() # H3 # bottom left 
        graph[79][92].makeBarrier() # H4 # lowest 
        graph[84][83].makeBarrier() # H5 # top right 
        graph[84][88].makeBarrier() # H6 # bottom right 

        graph[78][79].makeBarrier() 
        graph[77][80].makeBarrier()
        graph[76][81].makeBarrier()
        graph[75][82].makeBarrier() 

        graph[80][79].makeBarrier()
        graph[81][80].makeBarrier()
        graph[82][81].makeBarrier()
        graph[83][82].makeBarrier()

        for i in range(84, 88):
            graph[74][i].makeBarrier()

        for i in range(84, 88):
            graph[84][i].makeBarrier()
        
        graph[78][91].makeBarrier() 
        graph[77][91].makeBarrier()
        graph[76][90].makeBarrier()
        graph[75][89].makeBarrier() 

        graph[80][91].makeBarrier() 
        graph[81][91].makeBarrier()
        graph[82][90].makeBarrier() 
        graph[83][89].makeBarrier()

        









        # Kite 
        graph[80][50].makeBarrier() # K1 
        graph[86][45].makeBarrier() # K2 
        graph[92][50].makeBarrier() # K3 
        graph[86][69].makeBarrier() # K4 

        
        graph[81][51].makeBarrier()
        graph[81][52].makeBarrier()
        graph[81][53].makeBarrier()
        graph[81][54].makeBarrier()
        graph[82][55].makeBarrier()
        graph[82][56].makeBarrier()
        graph[82][57].makeBarrier()
        graph[82][58].makeBarrier()
        graph[83][59].makeBarrier()
        graph[83][60].makeBarrier()
        graph[83][61].makeBarrier()
        graph[83][62].makeBarrier()
        graph[84][63].makeBarrier()
        graph[84][64].makeBarrier()
        graph[84][65].makeBarrier()
        graph[84][66].makeBarrier()
        graph[85][67].makeBarrier()
        graph[85][68].makeBarrier()


        graph[91][51].makeBarrier()
        graph[91][52].makeBarrier()
        graph[91][53].makeBarrier()
        graph[91][54].makeBarrier()
        graph[90][55].makeBarrier()
        graph[90][56].makeBarrier()
        graph[90][57].makeBarrier()
        graph[90][58].makeBarrier()
        graph[89][59].makeBarrier()
        graph[89][60].makeBarrier()
        graph[89][61].makeBarrier()
        graph[89][62].makeBarrier()
        graph[88][63].makeBarrier()
        graph[88][64].makeBarrier()
        graph[88][65].makeBarrier()
        graph[88][66].makeBarrier()
        graph[87][67].makeBarrier()
        graph[87][68].makeBarrier()

        graph[91][49].makeBarrier()
        graph[90][49].makeBarrier()
        graph[89][48].makeBarrier()
        graph[88][47].makeBarrier() 
        graph[87][46].makeBarrier() 

        graph[81][49].makeBarrier()
        graph[82][49].makeBarrier()
        graph[83][48].makeBarrier()
        graph[84][47].makeBarrier() 
        graph[85][46].makeBarrier() 

        # --------------------------------------------------------------------------------       
    def environmentB():
        # rectangle 
        graph[ 20 ][ 64 ].makeBarrier()
        graph[ 20 ][ 64 ].makeBarrier()
        graph[ 20 ][ 65 ].makeBarrier()
        graph[ 20 ][ 65 ].makeBarrier()
        graph[ 20 ][ 65 ].makeBarrier()
        graph[ 20 ][ 66 ].makeBarrier()
        graph[ 20 ][ 66 ].makeBarrier()
        graph[ 20 ][ 66 ].makeBarrier()
        graph[ 20 ][ 66 ].makeBarrier()
        graph[ 20 ][ 67 ].makeBarrier()
        graph[ 20 ][ 67 ].makeBarrier()
        graph[ 20 ][ 67 ].makeBarrier()
        graph[ 20 ][ 67 ].makeBarrier()
        graph[ 20 ][ 68 ].makeBarrier()
        graph[ 20 ][ 68 ].makeBarrier()
        graph[ 20 ][ 68 ].makeBarrier()
        graph[ 20 ][ 69 ].makeBarrier()
        graph[ 20 ][ 69 ].makeBarrier()
        graph[ 20 ][ 70 ].makeBarrier()
        graph[ 20 ][ 70 ].makeBarrier()
        graph[ 20 ][ 71 ].makeBarrier()
        graph[ 20 ][ 71 ].makeBarrier()
        graph[ 20 ][ 71 ].makeBarrier()
        graph[ 20 ][ 72 ].makeBarrier()
        graph[ 20 ][ 72 ].makeBarrier()
        graph[ 20 ][ 72 ].makeBarrier()
        graph[ 20 ][ 72 ].makeBarrier()
        graph[ 20 ][ 73 ].makeBarrier()
        graph[ 20 ][ 73 ].makeBarrier()
        graph[ 20 ][ 73 ].makeBarrier()
        graph[ 20 ][ 74 ].makeBarrier()
        graph[ 20 ][ 74 ].makeBarrier()
        graph[ 20 ][ 74 ].makeBarrier()
        graph[ 21 ][ 74 ].makeBarrier()
        graph[ 21 ][ 74 ].makeBarrier()
        graph[ 21 ][ 74 ].makeBarrier()
        graph[ 21 ][ 74 ].makeBarrier()
        graph[ 21 ][ 74 ].makeBarrier()
        graph[ 22 ][ 74 ].makeBarrier()
        graph[ 22 ][ 74 ].makeBarrier()
        graph[ 22 ][ 74 ].makeBarrier()
        graph[ 22 ][ 74 ].makeBarrier()
        graph[ 23 ][ 74 ].makeBarrier()
        graph[ 23 ][ 74 ].makeBarrier()
        graph[ 23 ][ 74 ].makeBarrier()
        graph[ 23 ][ 74 ].makeBarrier()
        graph[ 24 ][ 74 ].makeBarrier()
        graph[ 24 ][ 74 ].makeBarrier()
        graph[ 24 ][ 74 ].makeBarrier()
        graph[ 25 ][ 74 ].makeBarrier()
        graph[ 25 ][ 74 ].makeBarrier()
        graph[ 26 ][ 74 ].makeBarrier()
        graph[ 26 ][ 74 ].makeBarrier()
        graph[ 26 ][ 74 ].makeBarrier()
        graph[ 27 ][ 74 ].makeBarrier()
        graph[ 27 ][ 74 ].makeBarrier()
        graph[ 27 ][ 74 ].makeBarrier()
        graph[ 28 ][ 74 ].makeBarrier()
        graph[ 28 ][ 74 ].makeBarrier()
        graph[ 28 ][ 74 ].makeBarrier()
        graph[ 29 ][ 74 ].makeBarrier()
        graph[ 29 ][ 74 ].makeBarrier()
        graph[ 29 ][ 74 ].makeBarrier()
        graph[ 30 ][ 74 ].makeBarrier()
        graph[ 30 ][ 74 ].makeBarrier()
        graph[ 30 ][ 74 ].makeBarrier()
        graph[ 30 ][ 74 ].makeBarrier()
        graph[ 31 ][ 74 ].makeBarrier()
        graph[ 32 ][ 74 ].makeBarrier()
        graph[ 32 ][ 74 ].makeBarrier()
        graph[ 32 ][ 74 ].makeBarrier()
        graph[ 33 ][ 74 ].makeBarrier()
        graph[ 33 ][ 74 ].makeBarrier()
        graph[ 33 ][ 74 ].makeBarrier()
        graph[ 33 ][ 74 ].makeBarrier()
        graph[ 34 ][ 74 ].makeBarrier()
        graph[ 34 ][ 74 ].makeBarrier()
        graph[ 34 ][ 74 ].makeBarrier()
        graph[ 35 ][ 74 ].makeBarrier()
        graph[ 35 ][ 74 ].makeBarrier()
        graph[ 35 ][ 74 ].makeBarrier()
        graph[ 35 ][ 74 ].makeBarrier()
        graph[ 35 ][ 74 ].makeBarrier()
        graph[ 36 ][ 74 ].makeBarrier()
        graph[ 36 ][ 74 ].makeBarrier()
        graph[ 36 ][ 74 ].makeBarrier()
        graph[ 36 ][ 74 ].makeBarrier()
        graph[ 36 ][ 74 ].makeBarrier()
        graph[ 36 ][ 74 ].makeBarrier()
        graph[ 37 ][ 74 ].makeBarrier()
        graph[ 37 ][ 74 ].makeBarrier()
        graph[ 38 ][ 74 ].makeBarrier()
        graph[ 38 ][ 74 ].makeBarrier()
        graph[ 38 ][ 74 ].makeBarrier()
        graph[ 38 ][ 74 ].makeBarrier()
        graph[ 39 ][ 74 ].makeBarrier()
        graph[ 39 ][ 74 ].makeBarrier()
        graph[ 39 ][ 74 ].makeBarrier()
        graph[ 39 ][ 74 ].makeBarrier()
        graph[ 39 ][ 74 ].makeBarrier()
        graph[ 40 ][ 74 ].makeBarrier()
        graph[ 40 ][ 74 ].makeBarrier()
        graph[ 40 ][ 74 ].makeBarrier()
        graph[ 40 ][ 74 ].makeBarrier()
        graph[ 40 ][ 74 ].makeBarrier()
        graph[ 40 ][ 74 ].makeBarrier()
        graph[ 40 ][ 74 ].makeBarrier()
        graph[ 40 ][ 74 ].makeBarrier()
        graph[ 40 ][ 73 ].makeBarrier()
        graph[ 40 ][ 73 ].makeBarrier()
        graph[ 40 ][ 73 ].makeBarrier()
        graph[ 40 ][ 72 ].makeBarrier()
        graph[ 40 ][ 72 ].makeBarrier()
        graph[ 40 ][ 71 ].makeBarrier()
        graph[ 40 ][ 71 ].makeBarrier()
        graph[ 40 ][ 71 ].makeBarrier()
        graph[ 40 ][ 70 ].makeBarrier()
        graph[ 40 ][ 70 ].makeBarrier()
        graph[ 40 ][ 70 ].makeBarrier()
        graph[ 40 ][ 69 ].makeBarrier()
        graph[ 40 ][ 69 ].makeBarrier()
        graph[ 40 ][ 68 ].makeBarrier()
        graph[ 40 ][ 68 ].makeBarrier()
        graph[ 40 ][ 68 ].makeBarrier()
        graph[ 40 ][ 67 ].makeBarrier()
        graph[ 40 ][ 67 ].makeBarrier()
        graph[ 40 ][ 67 ].makeBarrier()
        graph[ 40 ][ 66 ].makeBarrier()
        graph[ 40 ][ 66 ].makeBarrier()
        graph[ 40 ][ 66 ].makeBarrier()
        graph[ 40 ][ 65 ].makeBarrier()
        graph[ 40 ][ 65 ].makeBarrier()
        graph[ 40 ][ 65 ].makeBarrier()
        graph[ 40 ][ 64 ].makeBarrier()
        graph[ 40 ][ 64 ].makeBarrier()
        graph[ 40 ][ 64 ].makeBarrier()
        graph[ 40 ][ 64 ].makeBarrier()
        graph[ 40 ][ 64 ].makeBarrier()
        graph[ 40 ][ 64 ].makeBarrier()
        graph[ 40 ][ 64 ].makeBarrier()
        graph[ 40 ][ 64 ].makeBarrier()
        graph[ 40 ][ 64 ].makeBarrier()
        graph[ 39 ][ 64 ].makeBarrier()
        graph[ 39 ][ 64 ].makeBarrier()
        graph[ 39 ][ 64 ].makeBarrier()
        graph[ 39 ][ 64 ].makeBarrier()
        graph[ 39 ][ 64 ].makeBarrier()
        graph[ 38 ][ 64 ].makeBarrier()
        graph[ 38 ][ 64 ].makeBarrier()
        graph[ 38 ][ 64 ].makeBarrier()
        graph[ 37 ][ 64 ].makeBarrier()
        graph[ 37 ][ 64 ].makeBarrier()
        graph[ 37 ][ 64 ].makeBarrier()
        graph[ 36 ][ 64 ].makeBarrier()
        graph[ 36 ][ 64 ].makeBarrier()
        graph[ 36 ][ 64 ].makeBarrier()
        graph[ 35 ][ 64 ].makeBarrier()
        graph[ 35 ][ 64 ].makeBarrier()
        graph[ 34 ][ 64 ].makeBarrier()
        graph[ 34 ][ 64 ].makeBarrier()
        graph[ 33 ][ 64 ].makeBarrier()
        graph[ 33 ][ 64 ].makeBarrier()
        graph[ 33 ][ 64 ].makeBarrier()
        graph[ 33 ][ 64 ].makeBarrier()
        graph[ 32 ][ 64 ].makeBarrier()
        graph[ 32 ][ 64 ].makeBarrier()
        graph[ 31 ][ 64 ].makeBarrier()
        graph[ 31 ][ 64 ].makeBarrier()
        graph[ 30 ][ 64 ].makeBarrier()
        graph[ 30 ][ 64 ].makeBarrier()
        graph[ 29 ][ 64 ].makeBarrier()
        graph[ 29 ][ 64 ].makeBarrier()
        graph[ 28 ][ 64 ].makeBarrier()
        graph[ 28 ][ 64 ].makeBarrier()
        graph[ 28 ][ 64 ].makeBarrier()
        graph[ 27 ][ 64 ].makeBarrier()
        graph[ 27 ][ 64 ].makeBarrier()
        graph[ 27 ][ 64 ].makeBarrier()
        graph[ 26 ][ 64 ].makeBarrier()
        graph[ 26 ][ 64 ].makeBarrier()
        graph[ 25 ][ 64 ].makeBarrier()
        graph[ 25 ][ 64 ].makeBarrier()
        graph[ 24 ][ 64 ].makeBarrier()
        graph[ 24 ][ 64 ].makeBarrier()
        graph[ 24 ][ 64 ].makeBarrier()
        graph[ 23 ][ 64 ].makeBarrier()
        graph[ 23 ][ 64 ].makeBarrier()
        graph[ 22 ][ 64 ].makeBarrier()
        graph[ 22 ][ 64 ].makeBarrier()
        graph[ 22 ][ 64 ].makeBarrier()
        graph[ 22 ][ 63 ].makeBarrier()
        graph[ 21 ][ 63 ].makeBarrier()
        graph[ 21 ][ 63 ].makeBarrier()
        graph[ 21 ][ 63 ].makeBarrier()
        graph[ 21 ][ 64 ].makeBarrier()
        graph[ 21 ][ 64 ].makeBarrier()
        graph[ 21 ][ 64 ].makeBarrier()
        graph[ 21 ][ 64 ].makeBarrier()
        graph[ 20 ][ 64 ].makeBarrier()
        graph[ 20 ][ 64 ].makeBarrier()
        graph[ 20 ][ 64 ].makeBarrier()

        # triangle, hexagon, quadrilateral 
        graph[ 32 ][ 15 ].makeBarrier()
        graph[ 32 ][ 15 ].makeBarrier()
        graph[ 32 ][ 15 ].makeBarrier()
        graph[ 32 ][ 16 ].makeBarrier()
        graph[ 33 ][ 16 ].makeBarrier()
        graph[ 33 ][ 16 ].makeBarrier()
        graph[ 33 ][ 16 ].makeBarrier()
        graph[ 33 ][ 17 ].makeBarrier()
        graph[ 33 ][ 17 ].makeBarrier()
        graph[ 34 ][ 18 ].makeBarrier()
        graph[ 34 ][ 18 ].makeBarrier()
        graph[ 34 ][ 18 ].makeBarrier()
        graph[ 34 ][ 19 ].makeBarrier()
        graph[ 34 ][ 19 ].makeBarrier()
        graph[ 35 ][ 20 ].makeBarrier()
        graph[ 35 ][ 20 ].makeBarrier()
        graph[ 35 ][ 21 ].makeBarrier()
        graph[ 36 ][ 22 ].makeBarrier()
        graph[ 36 ][ 22 ].makeBarrier()
        graph[ 36 ][ 23 ].makeBarrier()
        graph[ 36 ][ 24 ].makeBarrier()
        graph[ 36 ][ 24 ].makeBarrier()
        graph[ 37 ][ 25 ].makeBarrier()
        graph[ 37 ][ 25 ].makeBarrier()
        graph[ 37 ][ 26 ].makeBarrier()
        graph[ 37 ][ 26 ].makeBarrier()
        graph[ 37 ][ 27 ].makeBarrier()
        graph[ 38 ][ 27 ].makeBarrier()
        graph[ 38 ][ 28 ].makeBarrier()
        graph[ 38 ][ 28 ].makeBarrier()
        graph[ 38 ][ 28 ].makeBarrier()
        graph[ 38 ][ 29 ].makeBarrier()
        graph[ 38 ][ 29 ].makeBarrier()
        graph[ 39 ][ 29 ].makeBarrier()
        graph[ 39 ][ 29 ].makeBarrier()
        graph[ 38 ][ 29 ].makeBarrier()
        graph[ 37 ][ 29 ].makeBarrier()
        graph[ 36 ][ 29 ].makeBarrier()
        graph[ 36 ][ 29 ].makeBarrier()
        graph[ 36 ][ 29 ].makeBarrier()
        graph[ 35 ][ 29 ].makeBarrier()
        graph[ 35 ][ 29 ].makeBarrier()
        graph[ 34 ][ 29 ].makeBarrier()
        graph[ 34 ][ 29 ].makeBarrier()
        graph[ 33 ][ 29 ].makeBarrier()
        graph[ 32 ][ 29 ].makeBarrier()
        graph[ 31 ][ 29 ].makeBarrier()
        graph[ 30 ][ 29 ].makeBarrier()
        graph[ 29 ][ 29 ].makeBarrier()
        graph[ 29 ][ 29 ].makeBarrier()
        graph[ 28 ][ 29 ].makeBarrier()
        graph[ 27 ][ 29 ].makeBarrier()
        graph[ 27 ][ 29 ].makeBarrier()
        graph[ 26 ][ 29 ].makeBarrier()
        graph[ 25 ][ 29 ].makeBarrier()
        graph[ 25 ][ 29 ].makeBarrier()
        graph[ 24 ][ 29 ].makeBarrier()
        graph[ 24 ][ 29 ].makeBarrier()
        graph[ 24 ][ 28 ].makeBarrier()
        graph[ 23 ][ 28 ].makeBarrier()
        graph[ 23 ][ 28 ].makeBarrier()
        graph[ 23 ][ 28 ].makeBarrier()
        graph[ 23 ][ 28 ].makeBarrier()
        graph[ 23 ][ 28 ].makeBarrier()
        graph[ 24 ][ 28 ].makeBarrier()
        graph[ 24 ][ 27 ].makeBarrier()
        graph[ 24 ][ 27 ].makeBarrier()
        graph[ 24 ][ 26 ].makeBarrier()
        graph[ 25 ][ 26 ].makeBarrier()
        graph[ 25 ][ 25 ].makeBarrier()
        graph[ 25 ][ 25 ].makeBarrier()
        graph[ 25 ][ 24 ].makeBarrier()
        graph[ 26 ][ 24 ].makeBarrier()
        graph[ 26 ][ 23 ].makeBarrier()
        graph[ 27 ][ 22 ].makeBarrier()
        graph[ 28 ][ 22 ].makeBarrier()
        graph[ 28 ][ 21 ].makeBarrier()
        graph[ 28 ][ 21 ].makeBarrier()
        graph[ 29 ][ 21 ].makeBarrier()
        graph[ 29 ][ 20 ].makeBarrier()
        graph[ 29 ][ 19 ].makeBarrier()
        graph[ 30 ][ 18 ].makeBarrier()
        graph[ 30 ][ 18 ].makeBarrier()
        graph[ 31 ][ 17 ].makeBarrier()
        graph[ 31 ][ 16 ].makeBarrier()
        graph[ 31 ][ 16 ].makeBarrier()
        graph[ 32 ][ 15 ].makeBarrier()
        graph[ 32 ][ 15 ].makeBarrier()
        graph[ 65 ][ 19 ].makeBarrier()
        graph[ 65 ][ 19 ].makeBarrier()
        graph[ 66 ][ 18 ].makeBarrier()
        graph[ 67 ][ 18 ].makeBarrier()
        graph[ 67 ][ 18 ].makeBarrier()
        graph[ 68 ][ 18 ].makeBarrier()
        graph[ 68 ][ 18 ].makeBarrier()
        graph[ 69 ][ 18 ].makeBarrier()
        graph[ 69 ][ 18 ].makeBarrier()
        graph[ 70 ][ 18 ].makeBarrier()
        graph[ 70 ][ 18 ].makeBarrier()
        graph[ 71 ][ 18 ].makeBarrier()
        graph[ 71 ][ 18 ].makeBarrier()
        graph[ 72 ][ 18 ].makeBarrier()
        graph[ 72 ][ 18 ].makeBarrier()
        graph[ 73 ][ 18 ].makeBarrier()
        graph[ 73 ][ 18 ].makeBarrier()
        graph[ 73 ][ 19 ].makeBarrier()
        graph[ 74 ][ 19 ].makeBarrier()
        graph[ 75 ][ 20 ].makeBarrier()
        graph[ 75 ][ 20 ].makeBarrier()
        graph[ 75 ][ 21 ].makeBarrier()
        graph[ 76 ][ 21 ].makeBarrier()
        graph[ 76 ][ 21 ].makeBarrier()
        graph[ 77 ][ 22 ].makeBarrier()
        graph[ 77 ][ 22 ].makeBarrier()
        graph[ 78 ][ 23 ].makeBarrier()
        graph[ 78 ][ 23 ].makeBarrier()
        graph[ 78 ][ 23 ].makeBarrier()
        graph[ 78 ][ 24 ].makeBarrier()
        graph[ 79 ][ 24 ].makeBarrier()
        graph[ 78 ][ 24 ].makeBarrier()
        graph[ 78 ][ 24 ].makeBarrier()
        graph[ 78 ][ 25 ].makeBarrier()
        graph[ 78 ][ 25 ].makeBarrier()
        graph[ 78 ][ 25 ].makeBarrier()
        graph[ 77 ][ 26 ].makeBarrier()
        graph[ 77 ][ 26 ].makeBarrier()
        graph[ 77 ][ 27 ].makeBarrier()
        graph[ 77 ][ 27 ].makeBarrier()
        graph[ 76 ][ 28 ].makeBarrier()
        graph[ 76 ][ 28 ].makeBarrier()
        graph[ 76 ][ 28 ].makeBarrier()
        graph[ 76 ][ 29 ].makeBarrier()
        graph[ 75 ][ 29 ].makeBarrier()
        graph[ 75 ][ 29 ].makeBarrier()
        graph[ 75 ][ 30 ].makeBarrier()
        graph[ 75 ][ 30 ].makeBarrier()
        graph[ 74 ][ 30 ].makeBarrier()
        graph[ 74 ][ 30 ].makeBarrier()
        graph[ 74 ][ 31 ].makeBarrier()
        graph[ 74 ][ 31 ].makeBarrier()
        graph[ 74 ][ 31 ].makeBarrier()
        graph[ 73 ][ 31 ].makeBarrier()
        graph[ 72 ][ 31 ].makeBarrier()
        graph[ 71 ][ 30 ].makeBarrier()
        graph[ 71 ][ 30 ].makeBarrier()
        graph[ 71 ][ 30 ].makeBarrier()
        graph[ 70 ][ 30 ].makeBarrier()
        graph[ 70 ][ 30 ].makeBarrier()
        graph[ 69 ][ 30 ].makeBarrier()
        graph[ 68 ][ 30 ].makeBarrier()
        graph[ 68 ][ 30 ].makeBarrier()
        graph[ 68 ][ 30 ].makeBarrier()
        graph[ 67 ][ 30 ].makeBarrier()
        graph[ 67 ][ 31 ].makeBarrier()
        graph[ 67 ][ 31 ].makeBarrier()
        graph[ 66 ][ 31 ].makeBarrier()
        graph[ 66 ][ 30 ].makeBarrier()
        graph[ 66 ][ 30 ].makeBarrier()
        graph[ 65 ][ 29 ].makeBarrier()
        graph[ 65 ][ 29 ].makeBarrier()
        graph[ 64 ][ 28 ].makeBarrier()
        graph[ 64 ][ 28 ].makeBarrier()
        graph[ 63 ][ 27 ].makeBarrier()
        graph[ 62 ][ 27 ].makeBarrier()
        graph[ 62 ][ 26 ].makeBarrier()
        graph[ 62 ][ 26 ].makeBarrier()
        graph[ 61 ][ 26 ].makeBarrier()
        graph[ 61 ][ 26 ].makeBarrier()
        graph[ 61 ][ 25 ].makeBarrier()
        graph[ 60 ][ 25 ].makeBarrier()
        graph[ 61 ][ 25 ].makeBarrier()
        graph[ 61 ][ 25 ].makeBarrier()
        graph[ 61 ][ 25 ].makeBarrier()
        graph[ 62 ][ 24 ].makeBarrier()
        graph[ 62 ][ 24 ].makeBarrier()
        graph[ 62 ][ 23 ].makeBarrier()
        graph[ 62 ][ 23 ].makeBarrier()
        graph[ 63 ][ 22 ].makeBarrier()
        graph[ 63 ][ 22 ].makeBarrier()
        graph[ 64 ][ 21 ].makeBarrier()
        graph[ 64 ][ 20 ].makeBarrier()
        graph[ 64 ][ 20 ].makeBarrier()
        graph[ 64 ][ 19 ].makeBarrier()
        graph[ 64 ][ 19 ].makeBarrier()
        graph[ 65 ][ 19 ].makeBarrier()
        graph[ 73 ][ 65 ].makeBarrier()
        graph[ 73 ][ 65 ].makeBarrier()
        graph[ 74 ][ 63 ].makeBarrier()
        graph[ 74 ][ 62 ].makeBarrier()
        graph[ 74 ][ 62 ].makeBarrier()
        graph[ 74 ][ 61 ].makeBarrier()
        graph[ 75 ][ 60 ].makeBarrier()
        graph[ 75 ][ 59 ].makeBarrier()
        graph[ 75 ][ 59 ].makeBarrier()
        graph[ 75 ][ 58 ].makeBarrier()
        graph[ 76 ][ 57 ].makeBarrier()
        graph[ 76 ][ 56 ].makeBarrier()
        graph[ 76 ][ 56 ].makeBarrier()
        graph[ 77 ][ 55 ].makeBarrier()
        graph[ 77 ][ 55 ].makeBarrier()
        graph[ 77 ][ 54 ].makeBarrier()
        graph[ 77 ][ 54 ].makeBarrier()
        graph[ 77 ][ 53 ].makeBarrier()
        graph[ 77 ][ 53 ].makeBarrier()
        graph[ 78 ][ 53 ].makeBarrier()
        graph[ 78 ][ 53 ].makeBarrier()
        graph[ 79 ][ 53 ].makeBarrier()
        graph[ 79 ][ 53 ].makeBarrier()
        graph[ 80 ][ 53 ].makeBarrier()
        graph[ 81 ][ 53 ].makeBarrier()
        graph[ 81 ][ 53 ].makeBarrier()
        graph[ 81 ][ 53 ].makeBarrier()
        graph[ 82 ][ 53 ].makeBarrier()
        graph[ 82 ][ 53 ].makeBarrier()
        graph[ 83 ][ 53 ].makeBarrier()
        graph[ 83 ][ 53 ].makeBarrier()
        graph[ 84 ][ 53 ].makeBarrier()
        graph[ 84 ][ 53 ].makeBarrier()
        graph[ 85 ][ 53 ].makeBarrier()
        graph[ 85 ][ 53 ].makeBarrier()
        graph[ 86 ][ 53 ].makeBarrier()
        graph[ 86 ][ 53 ].makeBarrier()
        graph[ 86 ][ 53 ].makeBarrier()
        graph[ 87 ][ 53 ].makeBarrier()
        graph[ 88 ][ 53 ].makeBarrier()
        graph[ 88 ][ 53 ].makeBarrier()
        graph[ 88 ][ 53 ].makeBarrier()
        graph[ 89 ][ 53 ].makeBarrier()
        graph[ 89 ][ 53 ].makeBarrier()
        graph[ 89 ][ 53 ].makeBarrier()
        graph[ 89 ][ 54 ].makeBarrier()
        graph[ 89 ][ 55 ].makeBarrier()
        graph[ 89 ][ 56 ].makeBarrier()
        graph[ 89 ][ 57 ].makeBarrier()
        graph[ 89 ][ 58 ].makeBarrier()
        graph[ 89 ][ 58 ].makeBarrier()
        graph[ 89 ][ 59 ].makeBarrier()
        graph[ 89 ][ 60 ].makeBarrier()
        graph[ 89 ][ 62 ].makeBarrier()
        graph[ 89 ][ 63 ].makeBarrier()
        graph[ 89 ][ 64 ].makeBarrier()
        graph[ 89 ][ 65 ].makeBarrier()
        graph[ 89 ][ 66 ].makeBarrier()
        graph[ 89 ][ 67 ].makeBarrier()
        graph[ 89 ][ 68 ].makeBarrier()
        graph[ 89 ][ 69 ].makeBarrier()
        graph[ 89 ][ 70 ].makeBarrier()
        graph[ 89 ][ 70 ].makeBarrier()
        graph[ 89 ][ 71 ].makeBarrier()
        graph[ 89 ][ 71 ].makeBarrier()
        graph[ 89 ][ 71 ].makeBarrier()
        graph[ 88 ][ 71 ].makeBarrier()
        graph[ 88 ][ 71 ].makeBarrier()
        graph[ 88 ][ 71 ].makeBarrier()
        graph[ 88 ][ 71 ].makeBarrier()
        graph[ 87 ][ 71 ].makeBarrier()
        graph[ 86 ][ 72 ].makeBarrier()
        graph[ 86 ][ 72 ].makeBarrier()
        graph[ 85 ][ 72 ].makeBarrier()
        graph[ 83 ][ 72 ].makeBarrier()
        graph[ 81 ][ 72 ].makeBarrier()
        graph[ 79 ][ 72 ].makeBarrier()
        graph[ 77 ][ 72 ].makeBarrier()
        graph[ 76 ][ 72 ].makeBarrier()
        graph[ 75 ][ 72 ].makeBarrier()
        graph[ 74 ][ 72 ].makeBarrier()
        graph[ 73 ][ 72 ].makeBarrier()
        graph[ 72 ][ 72 ].makeBarrier()
        graph[ 72 ][ 72 ].makeBarrier()
        graph[ 71 ][ 72 ].makeBarrier()
        graph[ 71 ][ 72 ].makeBarrier()
        graph[ 70 ][ 72 ].makeBarrier()
        graph[ 71 ][ 71 ].makeBarrier()
        graph[ 71 ][ 70 ].makeBarrier()
        graph[ 71 ][ 69 ].makeBarrier()
        graph[ 71 ][ 68 ].makeBarrier()
        graph[ 72 ][ 68 ].makeBarrier()
        graph[ 72 ][ 67 ].makeBarrier()
        graph[ 72 ][ 66 ].makeBarrier()
        graph[ 73 ][ 65 ].makeBarrier()
        graph[ 73 ][ 65 ].makeBarrier()
        graph[ 73 ][ 65 ].makeBarrier()
        graph[ 73 ][ 64 ].makeBarrier()
        graph[ 73 ][ 64 ].makeBarrier()
        graph[ 74 ][ 63 ].makeBarrier()
        graph[ 74 ][ 63 ].makeBarrier()
        graph[ 74 ][ 62 ].makeBarrier()
        graph[ 74 ][ 62 ].makeBarrier()
        graph[ 74 ][ 61 ].makeBarrier()
        graph[ 75 ][ 60 ].makeBarrier()
        graph[ 75 ][ 59 ].makeBarrier()
        graph[ 75 ][ 58 ].makeBarrier()
        graph[ 76 ][ 58 ].makeBarrier()
        graph[ 76 ][ 57 ].makeBarrier()
        graph[ 76 ][ 56 ].makeBarrier()
        graph[ 77 ][ 56 ].makeBarrier()
        graph[ 77 ][ 55 ].makeBarrier()
        graph[ 77 ][ 55 ].makeBarrier()
        graph[ 77 ][ 54 ].makeBarrier()
        graph[ 77 ][ 54 ].makeBarrier()
        graph[ 77 ][ 54 ].makeBarrier()
        graph[ 77 ][ 71 ].makeBarrier()
        graph[ 77 ][ 71 ].makeBarrier()
        graph[ 77 ][ 72 ].makeBarrier()
        graph[ 78 ][ 72 ].makeBarrier()
        graph[ 78 ][ 72 ].makeBarrier()
        graph[ 79 ][ 72 ].makeBarrier()
        graph[ 79 ][ 72 ].makeBarrier()
        graph[ 80 ][ 72 ].makeBarrier()
        graph[ 80 ][ 72 ].makeBarrier()
        graph[ 81 ][ 72 ].makeBarrier()
        graph[ 82 ][ 71 ].makeBarrier()
        graph[ 83 ][ 71 ].makeBarrier()
        graph[ 83 ][ 72 ].makeBarrier()
        graph[ 84 ][ 72 ].makeBarrier()
        graph[ 85 ][ 71 ].makeBarrier()
        graph[ 86 ][ 71 ].makeBarrier()
        graph[ 86 ][ 72 ].makeBarrier()
        graph[ 87 ][ 72 ].makeBarrier()
        graph[ 87 ][ 72 ].makeBarrier()
        graph[ 88 ][ 72 ].makeBarrier()
        graph[ 88 ][ 72 ].makeBarrier()
        graph[ 88 ][ 72 ].makeBarrier()
        graph[ 89 ][ 72 ].makeBarrier()
        graph[ 89 ][ 71 ].makeBarrier()
        graph[ 89 ][ 71 ].makeBarrier()
        graph[ 89 ][ 71 ].makeBarrier()
        graph[ 89 ][ 70 ].makeBarrier()
        graph[ 89 ][ 69 ].makeBarrier()
        graph[ 89 ][ 69 ].makeBarrier()
        graph[ 89 ][ 68 ].makeBarrier()
        graph[ 89 ][ 68 ].makeBarrier()
        graph[ 89 ][ 67 ].makeBarrier()
        graph[ 89 ][ 66 ].makeBarrier()
        graph[ 89 ][ 65 ].makeBarrier()
        graph[ 89 ][ 64 ].makeBarrier()
        graph[ 89 ][ 63 ].makeBarrier()
        graph[ 89 ][ 63 ].makeBarrier()
        graph[ 89 ][ 62 ].makeBarrier()
        graph[ 89 ][ 61 ].makeBarrier()
        graph[ 89 ][ 61 ].makeBarrier()
        graph[ 89 ][ 60 ].makeBarrier()
        graph[ 89 ][ 60 ].makeBarrier()
        graph[ 89 ][ 60 ].makeBarrier()

        graph[ 34 ][ 46 ].makeBarrier()
        graph[ 34 ][ 48 ].makeBarrier()
        graph[ 34 ][ 49 ].makeBarrier()
        graph[ 34 ][ 49 ].makeBarrier()
        graph[ 34 ][ 50 ].makeBarrier()
        graph[ 34 ][ 50 ].makeBarrier()
        graph[ 34 ][ 50 ].makeBarrier()
        graph[ 34 ][ 51 ].makeBarrier()
        graph[ 34 ][ 52 ].makeBarrier()
        graph[ 34 ][ 53 ].makeBarrier()
        graph[ 34 ][ 53 ].makeBarrier()
        graph[ 34 ][ 54 ].makeBarrier()
        graph[ 34 ][ 54 ].makeBarrier()
        graph[ 34 ][ 55 ].makeBarrier()
        graph[ 34 ][ 56 ].makeBarrier()
        graph[ 34 ][ 56 ].makeBarrier()
        graph[ 34 ][ 57 ].makeBarrier()
        graph[ 34 ][ 58 ].makeBarrier()
        graph[ 34 ][ 59 ].makeBarrier()
        graph[ 34 ][ 59 ].makeBarrier()
        graph[ 34 ][ 60 ].makeBarrier()
        graph[ 34 ][ 60 ].makeBarrier()
        graph[ 34 ][ 61 ].makeBarrier()
        graph[ 34 ][ 61 ].makeBarrier()
        graph[ 35 ][ 61 ].makeBarrier()
        graph[ 36 ][ 61 ].makeBarrier()
        graph[ 37 ][ 61 ].makeBarrier()
        graph[ 38 ][ 61 ].makeBarrier()
        graph[ 38 ][ 61 ].makeBarrier()
        graph[ 39 ][ 61 ].makeBarrier()
        graph[ 39 ][ 61 ].makeBarrier()
        graph[ 40 ][ 61 ].makeBarrier()
        graph[ 40 ][ 61 ].makeBarrier()
        graph[ 41 ][ 61 ].makeBarrier()
        graph[ 42 ][ 61 ].makeBarrier()
        graph[ 43 ][ 61 ].makeBarrier()
        graph[ 44 ][ 61 ].makeBarrier()
        graph[ 45 ][ 61 ].makeBarrier()
        graph[ 46 ][ 61 ].makeBarrier()
        graph[ 48 ][ 61 ].makeBarrier()
        graph[ 48 ][ 61 ].makeBarrier()
        graph[ 49 ][ 61 ].makeBarrier()
        graph[ 50 ][ 61 ].makeBarrier()
        graph[ 52 ][ 61 ].makeBarrier()
        graph[ 52 ][ 61 ].makeBarrier()
        graph[ 53 ][ 61 ].makeBarrier()
        graph[ 54 ][ 61 ].makeBarrier()
        graph[ 55 ][ 61 ].makeBarrier()
        graph[ 56 ][ 61 ].makeBarrier()
        graph[ 57 ][ 61 ].makeBarrier()
        graph[ 58 ][ 62 ].makeBarrier()
        graph[ 59 ][ 62 ].makeBarrier()
        graph[ 59 ][ 62 ].makeBarrier()
        graph[ 59 ][ 62 ].makeBarrier()
        graph[ 59 ][ 59 ].makeBarrier()
        graph[ 59 ][ 57 ].makeBarrier()
        graph[ 59 ][ 55 ].makeBarrier()
        graph[ 59 ][ 53 ].makeBarrier()
        graph[ 59 ][ 52 ].makeBarrier()
        graph[ 59 ][ 50 ].makeBarrier()
        graph[ 59 ][ 49 ].makeBarrier()
        graph[ 59 ][ 47 ].makeBarrier()
        graph[ 59 ][ 47 ].makeBarrier()
        graph[ 59 ][ 46 ].makeBarrier()
        graph[ 59 ][ 46 ].makeBarrier()
        graph[ 59 ][ 45 ].makeBarrier()
        graph[ 59 ][ 45 ].makeBarrier()
        graph[ 57 ][ 46 ].makeBarrier()
        graph[ 54 ][ 46 ].makeBarrier()
        graph[ 51 ][ 46 ].makeBarrier()
        graph[ 50 ][ 46 ].makeBarrier()
        graph[ 48 ][ 45 ].makeBarrier()
        graph[ 46 ][ 45 ].makeBarrier()
        graph[ 44 ][ 45 ].makeBarrier()
        graph[ 42 ][ 45 ].makeBarrier()
        graph[ 41 ][ 45 ].makeBarrier()
        graph[ 39 ][ 45 ].makeBarrier()
        graph[ 38 ][ 45 ].makeBarrier()
        graph[ 37 ][ 45 ].makeBarrier()
        graph[ 36 ][ 45 ].makeBarrier()
        graph[ 35 ][ 45 ].makeBarrier()
        graph[ 35 ][ 45 ].makeBarrier()
        graph[ 34 ][ 45 ].makeBarrier()
        graph[ 34 ][ 46 ].makeBarrier()
        graph[ 34 ][ 46 ].makeBarrier()
        graph[ 34 ][ 47 ].makeBarrier()
        graph[ 34 ][ 48 ].makeBarrier()
        graph[ 34 ][ 48 ].makeBarrier()
        graph[ 34 ][ 49 ].makeBarrier()
        graph[ 34 ][ 49 ].makeBarrier()
        graph[ 34 ][ 49 ].makeBarrier()
        graph[ 34 ][ 50 ].makeBarrier()
        graph[ 34 ][ 50 ].makeBarrier()
        graph[ 39 ][ 45 ].makeBarrier()
        graph[ 39 ][ 45 ].makeBarrier()
        graph[ 41 ][ 45 ].makeBarrier()
        graph[ 42 ][ 45 ].makeBarrier()
        graph[ 43 ][ 45 ].makeBarrier()
        graph[ 44 ][ 46 ].makeBarrier()
        graph[ 45 ][ 46 ].makeBarrier()
        graph[ 45 ][ 46 ].makeBarrier()
        graph[ 46 ][ 46 ].makeBarrier()
        graph[ 47 ][ 46 ].makeBarrier()
        graph[ 48 ][ 46 ].makeBarrier()
        graph[ 49 ][ 46 ].makeBarrier()
        graph[ 51 ][ 46 ].makeBarrier()
        graph[ 52 ][ 46 ].makeBarrier()
        graph[ 52 ][ 46 ].makeBarrier()
        graph[ 53 ][ 46 ].makeBarrier()
        graph[ 54 ][ 46 ].makeBarrier()
        graph[ 55 ][ 46 ].makeBarrier()
        graph[ 56 ][ 46 ].makeBarrier()
        graph[ 56 ][ 46 ].makeBarrier()
        graph[ 57 ][ 46 ].makeBarrier()
        graph[ 58 ][ 46 ].makeBarrier()
        graph[ 58 ][ 46 ].makeBarrier()
        graph[ 58 ][ 46 ].makeBarrier()
        graph[ 58 ][ 46 ].makeBarrier()
        graph[ 59 ][ 46 ].makeBarrier()
        graph[ 40 ][ 44 ].makeBarrier()
        graph[ 40 ][ 45 ].makeBarrier()
        graph[ 47 ][ 61 ].makeBarrier()
        graph[ 52 ][ 62 ].makeBarrier()
        graph[ 51 ][ 62 ].makeBarrier()
        graph[ 51 ][ 61 ].makeBarrier()
        graph[ 59 ][ 61 ].makeBarrier()
        graph[ 59 ][ 61 ].makeBarrier()
        graph[ 59 ][ 61 ].makeBarrier()
        graph[ 59 ][ 60 ].makeBarrier()
        graph[ 59 ][ 59 ].makeBarrier()
        graph[ 59 ][ 58 ].makeBarrier()
        graph[ 59 ][ 58 ].makeBarrier()
        graph[ 59 ][ 57 ].makeBarrier()
        graph[ 59 ][ 56 ].makeBarrier()
        graph[ 59 ][ 56 ].makeBarrier()
        graph[ 59 ][ 55 ].makeBarrier()
        graph[ 59 ][ 54 ].makeBarrier()
        graph[ 59 ][ 54 ].makeBarrier()
        graph[ 59 ][ 53 ].makeBarrier()
        graph[ 59 ][ 53 ].makeBarrier()
        graph[ 59 ][ 52 ].makeBarrier()
        graph[ 47 ][ 80 ].makeBarrier()
        graph[ 47 ][ 80 ].makeBarrier()
        graph[ 48 ][ 80 ].makeBarrier()
        graph[ 49 ][ 80 ].makeBarrier()
        graph[ 50 ][ 80 ].makeBarrier()
        graph[ 51 ][ 80 ].makeBarrier()
        graph[ 52 ][ 80 ].makeBarrier()
        graph[ 52 ][ 80 ].makeBarrier()
        graph[ 54 ][ 80 ].makeBarrier()
        graph[ 55 ][ 80 ].makeBarrier()
        graph[ 55 ][ 80 ].makeBarrier()
        graph[ 55 ][ 80 ].makeBarrier()
        graph[ 56 ][ 80 ].makeBarrier()
        graph[ 57 ][ 82 ].makeBarrier()
        graph[ 58 ][ 83 ].makeBarrier()
        graph[ 59 ][ 83 ].makeBarrier()
        graph[ 59 ][ 84 ].makeBarrier()
        graph[ 60 ][ 85 ].makeBarrier()
        graph[ 61 ][ 85 ].makeBarrier()
        graph[ 61 ][ 86 ].makeBarrier()
        graph[ 61 ][ 86 ].makeBarrier()
        graph[ 61 ][ 88 ].makeBarrier()
        graph[ 61 ][ 89 ].makeBarrier()
        graph[ 61 ][ 91 ].makeBarrier()
        graph[ 61 ][ 92 ].makeBarrier()
        graph[ 61 ][ 92 ].makeBarrier()
        graph[ 61 ][ 92 ].makeBarrier()
        graph[ 61 ][ 92 ].makeBarrier()
        graph[ 61 ][ 93 ].makeBarrier()
        graph[ 60 ][ 93 ].makeBarrier()
        graph[ 59 ][ 94 ].makeBarrier()
        graph[ 59 ][ 94 ].makeBarrier()
        graph[ 59 ][ 95 ].makeBarrier()
        graph[ 59 ][ 95 ].makeBarrier()
        graph[ 58 ][ 95 ].makeBarrier()
        graph[ 58 ][ 95 ].makeBarrier()
        graph[ 58 ][ 95 ].makeBarrier()
        graph[ 57 ][ 95 ].makeBarrier()
        graph[ 55 ][ 95 ].makeBarrier()
        graph[ 54 ][ 96 ].makeBarrier()
        graph[ 53 ][ 96 ].makeBarrier()
        graph[ 52 ][ 96 ].makeBarrier()
        graph[ 51 ][ 96 ].makeBarrier()
        graph[ 50 ][ 95 ].makeBarrier()
        graph[ 48 ][ 95 ].makeBarrier()
        graph[ 48 ][ 95 ].makeBarrier()
        graph[ 47 ][ 95 ].makeBarrier()
        graph[ 47 ][ 95 ].makeBarrier()
        graph[ 46 ][ 94 ].makeBarrier()
        graph[ 45 ][ 93 ].makeBarrier()
        graph[ 44 ][ 92 ].makeBarrier()
        graph[ 43 ][ 91 ].makeBarrier()
        graph[ 42 ][ 91 ].makeBarrier()
        graph[ 42 ][ 90 ].makeBarrier()
        graph[ 42 ][ 90 ].makeBarrier()
        graph[ 42 ][ 88 ].makeBarrier()
        graph[ 42 ][ 88 ].makeBarrier()
        graph[ 42 ][ 87 ].makeBarrier()
        graph[ 42 ][ 86 ].makeBarrier()
        graph[ 42 ][ 85 ].makeBarrier()
        graph[ 42 ][ 84 ].makeBarrier()
        graph[ 42 ][ 84 ].makeBarrier()
        graph[ 42 ][ 84 ].makeBarrier()
        graph[ 43 ][ 83 ].makeBarrier()
        graph[ 43 ][ 82 ].makeBarrier()
        graph[ 44 ][ 82 ].makeBarrier()
        graph[ 44 ][ 82 ].makeBarrier()
        graph[ 45 ][ 81 ].makeBarrier()
        graph[ 45 ][ 81 ].makeBarrier()
        graph[ 46 ][ 80 ].makeBarrier()
        graph[ 46 ][ 80 ].makeBarrier()
        graph[ 46 ][ 80 ].makeBarrier()
        graph[ 47 ][ 80 ].makeBarrier()
        graph[ 46 ][ 80 ].makeBarrier()
        graph[ 45 ][ 80 ].makeBarrier()
        graph[ 44 ][ 81 ].makeBarrier()
        graph[ 44 ][ 81 ].makeBarrier()
        graph[ 44 ][ 82 ].makeBarrier()
        graph[ 43 ][ 82 ].makeBarrier()
        graph[ 43 ][ 83 ].makeBarrier()
        graph[ 43 ][ 83 ].makeBarrier()
        graph[ 42 ][ 84 ].makeBarrier()
        graph[ 42 ][ 85 ].makeBarrier()
        graph[ 42 ][ 85 ].makeBarrier()
        graph[ 42 ][ 86 ].makeBarrier()
        graph[ 42 ][ 87 ].makeBarrier()
        graph[ 42 ][ 87 ].makeBarrier()
        graph[ 42 ][ 88 ].makeBarrier()
        graph[ 42 ][ 88 ].makeBarrier()
        graph[ 42 ][ 89 ].makeBarrier()
        graph[ 42 ][ 89 ].makeBarrier()
        graph[ 42 ][ 90 ].makeBarrier()
        graph[ 42 ][ 91 ].makeBarrier()
        graph[ 42 ][ 91 ].makeBarrier()
        graph[ 42 ][ 91 ].makeBarrier()
        graph[ 43 ][ 91 ].makeBarrier()
        graph[ 43 ][ 92 ].makeBarrier()
        graph[ 44 ][ 92 ].makeBarrier()
        graph[ 44 ][ 93 ].makeBarrier()
        graph[ 45 ][ 93 ].makeBarrier()
        graph[ 45 ][ 93 ].makeBarrier()
        graph[ 45 ][ 94 ].makeBarrier()
        graph[ 46 ][ 94 ].makeBarrier()
        graph[ 46 ][ 94 ].makeBarrier()
        graph[ 47 ][ 95 ].makeBarrier()
        graph[ 47 ][ 95 ].makeBarrier()
        graph[ 48 ][ 95 ].makeBarrier()
        graph[ 48 ][ 95 ].makeBarrier()
        graph[ 49 ][ 95 ].makeBarrier()
        graph[ 49 ][ 95 ].makeBarrier()
        graph[ 50 ][ 95 ].makeBarrier()
        graph[ 50 ][ 95 ].makeBarrier()
        graph[ 51 ][ 95 ].makeBarrier()
        graph[ 51 ][ 96 ].makeBarrier()
        graph[ 51 ][ 96 ].makeBarrier()
        graph[ 52 ][ 96 ].makeBarrier()
        graph[ 52 ][ 96 ].makeBarrier()
        graph[ 52 ][ 96 ].makeBarrier()
        graph[ 53 ][ 96 ].makeBarrier()
        graph[ 53 ][ 96 ].makeBarrier()
        graph[ 54 ][ 95 ].makeBarrier()
        graph[ 55 ][ 95 ].makeBarrier()
        graph[ 55 ][ 95 ].makeBarrier()
        graph[ 56 ][ 95 ].makeBarrier()
        graph[ 56 ][ 95 ].makeBarrier()
        graph[ 57 ][ 95 ].makeBarrier()
        graph[ 57 ][ 95 ].makeBarrier()
        graph[ 58 ][ 94 ].makeBarrier()
        graph[ 59 ][ 93 ].makeBarrier()
        graph[ 59 ][ 93 ].makeBarrier()
        graph[ 60 ][ 93 ].makeBarrier()
        graph[ 60 ][ 92 ].makeBarrier()
        graph[ 61 ][ 92 ].makeBarrier()
        graph[ 61 ][ 92 ].makeBarrier()
        graph[ 61 ][ 91 ].makeBarrier()
        graph[ 61 ][ 91 ].makeBarrier()
        graph[ 61 ][ 90 ].makeBarrier()
        graph[ 62 ][ 89 ].makeBarrier()
        graph[ 62 ][ 89 ].makeBarrier()
        graph[ 62 ][ 88 ].makeBarrier()
        graph[ 62 ][ 88 ].makeBarrier()
        graph[ 61 ][ 87 ].makeBarrier()
        graph[ 61 ][ 86 ].makeBarrier()
        graph[ 61 ][ 86 ].makeBarrier()
        graph[ 61 ][ 86 ].makeBarrier()
        graph[ 61 ][ 85 ].makeBarrier()
        graph[ 61 ][ 85 ].makeBarrier()
        graph[ 61 ][ 85 ].makeBarrier()
        graph[ 61 ][ 85 ].makeBarrier()
        graph[ 60 ][ 84 ].makeBarrier()
        graph[ 59 ][ 84 ].makeBarrier()
        graph[ 59 ][ 84 ].makeBarrier()
        graph[ 58 ][ 83 ].makeBarrier()
        graph[ 58 ][ 83 ].makeBarrier()
        graph[ 57 ][ 82 ].makeBarrier()
        graph[ 57 ][ 82 ].makeBarrier()
        graph[ 57 ][ 82 ].makeBarrier()
        graph[ 56 ][ 81 ].makeBarrier()
        graph[ 56 ][ 81 ].makeBarrier()
        graph[ 56 ][ 80 ].makeBarrier()
        graph[ 56 ][ 80 ].makeBarrier()
        graph[ 56 ][ 80 ].makeBarrier()
        graph[ 56 ][ 80 ].makeBarrier()
        graph[ 56 ][ 80 ].makeBarrier()
        graph[ 55 ][ 80 ].makeBarrier()
        graph[ 54 ][ 80 ].makeBarrier()
        graph[ 54 ][ 80 ].makeBarrier()
        graph[ 53 ][ 80 ].makeBarrier()
        graph[ 53 ][ 80 ].makeBarrier()
        graph[ 53 ][ 80 ].makeBarrier()
        graph[ 52 ][ 80 ].makeBarrier()

        graph[ 45 ][ 2 ].makeBarrier()
        graph[ 45 ][ 2 ].makeBarrier()
        graph[ 47 ][ 2 ].makeBarrier()
        graph[ 48 ][ 2 ].makeBarrier()
        graph[ 49 ][ 2 ].makeBarrier()
        graph[ 49 ][ 2 ].makeBarrier()
        graph[ 50 ][ 2 ].makeBarrier()
        graph[ 50 ][ 2 ].makeBarrier()
        graph[ 51 ][ 2 ].makeBarrier()
        graph[ 52 ][ 2 ].makeBarrier()
        graph[ 52 ][ 2 ].makeBarrier()
        graph[ 53 ][ 2 ].makeBarrier()
        graph[ 53 ][ 2 ].makeBarrier()
        graph[ 54 ][ 2 ].makeBarrier()
        graph[ 55 ][ 2 ].makeBarrier()
        graph[ 56 ][ 2 ].makeBarrier()
        graph[ 56 ][ 2 ].makeBarrier()
        graph[ 56 ][ 2 ].makeBarrier()
        graph[ 56 ][ 2 ].makeBarrier()
        graph[ 57 ][ 3 ].makeBarrier()
        graph[ 58 ][ 4 ].makeBarrier()
        graph[ 58 ][ 4 ].makeBarrier()
        graph[ 59 ][ 5 ].makeBarrier()
        graph[ 59 ][ 5 ].makeBarrier()
        graph[ 60 ][ 6 ].makeBarrier()
        graph[ 60 ][ 6 ].makeBarrier()
        graph[ 61 ][ 6 ].makeBarrier()
        graph[ 61 ][ 7 ].makeBarrier()
        graph[ 61 ][ 7 ].makeBarrier()
        graph[ 61 ][ 7 ].makeBarrier()
        graph[ 61 ][ 8 ].makeBarrier()
        graph[ 61 ][ 8 ].makeBarrier()
        graph[ 61 ][ 8 ].makeBarrier()
        graph[ 61 ][ 9 ].makeBarrier()
        graph[ 61 ][ 10 ].makeBarrier()
        graph[ 61 ][ 10 ].makeBarrier()
        graph[ 61 ][ 10 ].makeBarrier()
        graph[ 61 ][ 11 ].makeBarrier()
        graph[ 61 ][ 11 ].makeBarrier()
        graph[ 61 ][ 12 ].makeBarrier()
        graph[ 61 ][ 12 ].makeBarrier()
        graph[ 61 ][ 12 ].makeBarrier()
        graph[ 61 ][ 12 ].makeBarrier()
        graph[ 61 ][ 13 ].makeBarrier()
        graph[ 60 ][ 13 ].makeBarrier()
        graph[ 59 ][ 14 ].makeBarrier()
        graph[ 59 ][ 14 ].makeBarrier()
        graph[ 59 ][ 15 ].makeBarrier()
        graph[ 58 ][ 15 ].makeBarrier()
        graph[ 58 ][ 15 ].makeBarrier()
        graph[ 57 ][ 16 ].makeBarrier()
        graph[ 57 ][ 16 ].makeBarrier()
        graph[ 57 ][ 16 ].makeBarrier()
        graph[ 57 ][ 17 ].makeBarrier()
        graph[ 56 ][ 17 ].makeBarrier()
        graph[ 56 ][ 17 ].makeBarrier()
        graph[ 55 ][ 17 ].makeBarrier()
        graph[ 54 ][ 17 ].makeBarrier()
        graph[ 52 ][ 17 ].makeBarrier()
        graph[ 52 ][ 17 ].makeBarrier()
        graph[ 51 ][ 16 ].makeBarrier()
        graph[ 51 ][ 16 ].makeBarrier()
        graph[ 50 ][ 16 ].makeBarrier()
        graph[ 49 ][ 16 ].makeBarrier()
        graph[ 48 ][ 16 ].makeBarrier()
        graph[ 48 ][ 16 ].makeBarrier()
        graph[ 47 ][ 16 ].makeBarrier()
        graph[ 47 ][ 16 ].makeBarrier()
        graph[ 46 ][ 16 ].makeBarrier()
        graph[ 46 ][ 16 ].makeBarrier()
        graph[ 46 ][ 16 ].makeBarrier()
        graph[ 44 ][ 14 ].makeBarrier()
        graph[ 43 ][ 13 ].makeBarrier()
        graph[ 42 ][ 13 ].makeBarrier()
        graph[ 42 ][ 12 ].makeBarrier()
        graph[ 41 ][ 12 ].makeBarrier()
        graph[ 41 ][ 12 ].makeBarrier()
        graph[ 40 ][ 12 ].makeBarrier()
        graph[ 40 ][ 12 ].makeBarrier()
        graph[ 40 ][ 11 ].makeBarrier()
        graph[ 40 ][ 11 ].makeBarrier()
        graph[ 40 ][ 10 ].makeBarrier()
        graph[ 40 ][ 9 ].makeBarrier()
        graph[ 40 ][ 8 ].makeBarrier()
        graph[ 40 ][ 8 ].makeBarrier()
        graph[ 40 ][ 7 ].makeBarrier()
        graph[ 40 ][ 6 ].makeBarrier()
        graph[ 40 ][ 6 ].makeBarrier()
        graph[ 40 ][ 5 ].makeBarrier()
        graph[ 40 ][ 5 ].makeBarrier()
        graph[ 40 ][ 5 ].makeBarrier()
        graph[ 41 ][ 5 ].makeBarrier()
        graph[ 42 ][ 4 ].makeBarrier()
        graph[ 42 ][ 4 ].makeBarrier()
        graph[ 43 ][ 4 ].makeBarrier()
        graph[ 44 ][ 3 ].makeBarrier()
        graph[ 44 ][ 2 ].makeBarrier()
        graph[ 45 ][ 2 ].makeBarrier()
        graph[ 45 ][ 2 ].makeBarrier()
        graph[ 46 ][ 2 ].makeBarrier()
        graph[ 46 ][ 1 ].makeBarrier()
        graph[ 46 ][ 1 ].makeBarrier()
        graph[ 46 ][ 1 ].makeBarrier()
        graph[ 47 ][ 2 ].makeBarrier()
        graph[ 48 ][ 2 ].makeBarrier()
        graph[ 48 ][ 2 ].makeBarrier()
        graph[ 48 ][ 2 ].makeBarrier()
        graph[ 54 ][ 17 ].makeBarrier()
        graph[ 53 ][ 17 ].makeBarrier()
        graph[ 51 ][ 17 ].makeBarrier()
        graph[ 50 ][ 17 ].makeBarrier()
        graph[ 45 ][ 16 ].makeBarrier()
        graph[ 45 ][ 15 ].makeBarrier()
        graph[ 44 ][ 15 ].makeBarrier()
        graph[ 43 ][ 14 ].makeBarrier()
        graph[ 43 ][ 14 ].makeBarrier()
        graph[ 83 ][ 83 ].makeBarrier()
        graph[ 83 ][ 84 ].makeBarrier()
        graph[ 83 ][ 84 ].makeBarrier()
        graph[ 84 ][ 85 ].makeBarrier()
        graph[ 84 ][ 85 ].makeBarrier()
        graph[ 85 ][ 86 ].makeBarrier()
        graph[ 85 ][ 87 ].makeBarrier()
        graph[ 86 ][ 87 ].makeBarrier()
        graph[ 86 ][ 88 ].makeBarrier()
        graph[ 87 ][ 88 ].makeBarrier()
        graph[ 87 ][ 89 ].makeBarrier()
        graph[ 88 ][ 90 ].makeBarrier()
        graph[ 88 ][ 91 ].makeBarrier()
        graph[ 88 ][ 91 ].makeBarrier()
        graph[ 89 ][ 92 ].makeBarrier()
        graph[ 89 ][ 93 ].makeBarrier()
        graph[ 89 ][ 94 ].makeBarrier()
        graph[ 89 ][ 94 ].makeBarrier()
        graph[ 89 ][ 94 ].makeBarrier()
        graph[ 89 ][ 94 ].makeBarrier()
        graph[ 88 ][ 94 ].makeBarrier()
        graph[ 87 ][ 94 ].makeBarrier()
        graph[ 86 ][ 94 ].makeBarrier()
        graph[ 86 ][ 94 ].makeBarrier()
        graph[ 85 ][ 94 ].makeBarrier()
        graph[ 85 ][ 94 ].makeBarrier()
        graph[ 84 ][ 94 ].makeBarrier()
        graph[ 84 ][ 94 ].makeBarrier()
        graph[ 83 ][ 94 ].makeBarrier()
        graph[ 82 ][ 94 ].makeBarrier()
        graph[ 81 ][ 94 ].makeBarrier()
        graph[ 81 ][ 94 ].makeBarrier()
        graph[ 80 ][ 94 ].makeBarrier()
        graph[ 80 ][ 94 ].makeBarrier()
        graph[ 79 ][ 94 ].makeBarrier()
        graph[ 78 ][ 94 ].makeBarrier()
        graph[ 78 ][ 94 ].makeBarrier()
        graph[ 78 ][ 94 ].makeBarrier()
        graph[ 77 ][ 94 ].makeBarrier()
        graph[ 77 ][ 93 ].makeBarrier()
        graph[ 77 ][ 93 ].makeBarrier()
        graph[ 78 ][ 93 ].makeBarrier()
        graph[ 78 ][ 93 ].makeBarrier()
        graph[ 78 ][ 92 ].makeBarrier()
        graph[ 79 ][ 92 ].makeBarrier()
        graph[ 79 ][ 92 ].makeBarrier()
        graph[ 79 ][ 91 ].makeBarrier()
        graph[ 79 ][ 91 ].makeBarrier()
        graph[ 80 ][ 90 ].makeBarrier()
        graph[ 80 ][ 89 ].makeBarrier()
        graph[ 81 ][ 88 ].makeBarrier()
        graph[ 81 ][ 88 ].makeBarrier()
        graph[ 81 ][ 87 ].makeBarrier()
        graph[ 82 ][ 87 ].makeBarrier()
        graph[ 82 ][ 86 ].makeBarrier()
        graph[ 82 ][ 85 ].makeBarrier()
        graph[ 83 ][ 85 ].makeBarrier()
        graph[ 83 ][ 84 ].makeBarrier()
        graph[ 83 ][ 84 ].makeBarrier()
        graph[ 83 ][ 84 ].makeBarrier()

        graph[ 76 ][ 40 ].makeBarrier()
        graph[ 77 ][ 41 ].makeBarrier()
        graph[ 77 ][ 41 ].makeBarrier()
        graph[ 77 ][ 42 ].makeBarrier()
        graph[ 78 ][ 42 ].makeBarrier()
        graph[ 78 ][ 43 ].makeBarrier()
        graph[ 79 ][ 43 ].makeBarrier()
        graph[ 79 ][ 44 ].makeBarrier()
        graph[ 80 ][ 45 ].makeBarrier()
        graph[ 80 ][ 45 ].makeBarrier()
        graph[ 81 ][ 46 ].makeBarrier()
        graph[ 81 ][ 47 ].makeBarrier()
        graph[ 81 ][ 48 ].makeBarrier()
        graph[ 81 ][ 48 ].makeBarrier()
        graph[ 81 ][ 48 ].makeBarrier()
        graph[ 80 ][ 48 ].makeBarrier()
        graph[ 78 ][ 48 ].makeBarrier()
        graph[ 77 ][ 48 ].makeBarrier()
        graph[ 76 ][ 48 ].makeBarrier()
        graph[ 75 ][ 48 ].makeBarrier()
        graph[ 74 ][ 48 ].makeBarrier()
        graph[ 74 ][ 48 ].makeBarrier()
        graph[ 73 ][ 48 ].makeBarrier()
        graph[ 73 ][ 48 ].makeBarrier()
        graph[ 73 ][ 48 ].makeBarrier()
        graph[ 73 ][ 48 ].makeBarrier()
        graph[ 73 ][ 48 ].makeBarrier()
        graph[ 73 ][ 47 ].makeBarrier()
        graph[ 74 ][ 46 ].makeBarrier()
        graph[ 74 ][ 45 ].makeBarrier()
        graph[ 74 ][ 44 ].makeBarrier()
        graph[ 75 ][ 43 ].makeBarrier()
        graph[ 75 ][ 43 ].makeBarrier()
        graph[ 75 ][ 42 ].makeBarrier()
        graph[ 76 ][ 41 ].makeBarrier()
        graph[ 76 ][ 41 ].makeBarrier()
        graph[ 76 ][ 41 ].makeBarrier()
        graph[ 78 ][ 49 ].makeBarrier()
        graph[ 79 ][ 49 ].makeBarrier()
        graph[ 79 ][ 49 ].makeBarrier()
        graph[ 79 ][ 49 ].makeBarrier()
        graph[ 80 ][ 49 ].makeBarrier()
        graph[ 80 ][ 48 ].makeBarrier()
        graph[ 80 ][ 48 ].makeBarrier()
        graph[ 79 ][ 48 ].makeBarrier()
        graph[ 79 ][ 48 ].makeBarrier()
        graph[ 79 ][ 48 ].makeBarrier()








    while True:
        whichenv, whatval = input("type A for environment 1 or B for environment 2, followed by C value \n").split()
        c = int(whatval)
        if c < 133:
            print("ERROR: Please exit and run again with an appropriate constraint value.  ")
            break
        if whichenv == 'A':
            environmentA()
        elif whichenv == 'B':
            environmentB()
        else:
            print("please enter a valid option ")

        draw(environment, graph, entries, size)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            for row in graph:
                for node in row:
                    node.updateNeighbors(graph)
            A_Star(lambda: draw(environment, graph, entries, size), graph, start, goal, c)
            break


main(environment, 800)
# python3 AStar.py     
