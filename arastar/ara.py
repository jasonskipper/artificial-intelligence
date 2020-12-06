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









    while True:
        draw(environment, graph, entries, size)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            for row in graph:
                for node in row:
                    node.updateNeighbors(graph)
            arastar(lambda: draw(environment, graph, entries, size), graph, start, goal)
            break


main(environment, 800)
