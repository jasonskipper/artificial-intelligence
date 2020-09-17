# References: 
# https://www.youtube.com/watch?v=JtiK0DOeI4A&ab_channel=TechWithTim 
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

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255) 
ORANGE = (255, 165, 0) 
BLACK = (0, 0, 0)
WHITE = (255, 255, 255) 
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
GREY = (128, 128, 128) 

class Node:
    def __init__(self, row, column, size, numberOfRows):
        self.row = row
        self.column = column
        self.i = row * size 
        self.j = column * size 
        self.color = WHITE
        self.neighbors = []
        self.size = size 
        self.numberOfRows = numberOfRows

    def getPosition(self):
        return self.row, self.column

    # def isBarrier(self):
    #     return self.color == BLACK
    
    def makeBarrier(self):
        self.color = BLACK
    
    # def reset(self):
    #     self.color = WHITE

    def makeStart(self):
        self.color = ORANGE
    
    # def makeClosed(self):
    #     self.color = RED

    # def makeOpen(self):
    #     self.color = GREEN

    # def makeGoal(self):
    #     self.color = BLUE

    # def makePath(self):
    #     self.color = PURPLE 

    def draw(self, environment):
        pygame.draw.rect(environment, self.color, (self.i, self.j, self.size, self.size))

    def updateNeighbors(self, grid):
        self.neighbors = []
        # if self.row < self.numberOfRows - 1 and not grid[self.row +1][self.column].isBarrier():
        if self.row < self.numberOfRows - 1 and not grid[self.row+1][self.column].color == BLACK:
            self.neighbors.append(grid[self.row + 1][self.column])
        # if self.row > 0 and not grid[self.row - 1][self.column].isBarrier():
        if self.row > 0 and not grid[self.row - 1][self.column].color == BLACK:
            self.neighbors.append(grid[self.row - 1][self.column])
        if self.column < self.numberOfRows - 1 and not grid[self.row][self.column + 1].color == BLACK:
            self.neighbors.append(grid[self.row][self.column + 1])
        if self.column > 0 and not grid[self.row][self.column - 1].color == BLACK:
            self.neighbors.append(grid[self.row][self.column - 1])

def heuristic(pointA, pointB):
    i1, j1 = pointA
    i2, j2 = pointB
    return abs(i1 - i2) + abs(j1 - j2)

def reconstructPath(origin, current, draw):
    while current in origin:
        current = origin[current]
        # current.makePath()
        current.color = PURPLE
        draw()

def A_Star(draw, grid, start, goal):
    openPQ = PriorityQueue()
    count = 0
    openPQ.put((0, count, start)) # add start node with original f score (0) into the open set 
    origin = {}
    g = {}
    for row in grid:
        for node in row: 
            g[node] = float("inf")
    g[start] = 0
    f = {}
    for row in grid:
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
            # goal.makeGoal()
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
                    # neighbor.makeOpen()
                    neighbor.color = GREEN

        draw()

        if current != start:
            # current.makeClosed()
            current.color = RED

    return False

def draw(environment, grid, rows, size):
    environment.fill(WHITE)
    
    for row in grid:
        for node in row:
            node.draw(environment)

    # drawGrid(window, rows, size)
    space = size // rows
    for i in range(rows):
        pygame.draw.line(environment, GREY, (0, i * space), (size, i * space))
        for j in range(rows):
            pygame.draw.line(environment, GREY, (j * space, 0), (j * space, size))
    pygame.display.update()

def main(environment, size):
    ROWS = 100
    # grid = makeGrid(ROWS, size)
    grid = []
    space = size // ROWS 
    for i in range(ROWS):
        grid.append([])
        for j in range(ROWS):
            node = Node(i, j, space, ROWS)
            grid[i].append(node)



    # Start 
    # grid[10][90].makeStart()

    # Goal 
    # grid[92][45].makeGoal()
    # grid[20][65].makeGoal()
    # grid[92][45].color = BLUE

    # goal = grid[92][45]

    start = grid[10][90]
    grid[10][90].color = ORANGE

    # goal = grid[20][65]
    # grid[20][65].color = BLUE

    goal = grid[92][45]
    grid[92][45].color = BLUE
    




    # Horizontal Rectangle 
    # ------------------------------------- 
    grid[16][95].makeBarrier() #HR1 
    grid[16][85].makeBarrier() #HR2 
    
    grid[46][85].makeBarrier() #HR3 
    grid[46][95].makeBarrier() #HR4 


    # left side of horizontal rectangle 
    for i in range(85, 95):
        grid[16][i].makeBarrier()

    # top side of horizontal rectangle 
    for i in range(17, 46):
        grid[i][85].makeBarrier() 

    # right side of horizontal rectangle
    for i in reversed(range(86, 95)):
        grid[46][i].makeBarrier()

    # bottom side of horizontal rectangle 
    for i in range(17, 46):
        grid[i][95].makeBarrier()
    # -------------------------------------







    # Pentagon 
    # -------------------------------------
    grid[25][75].makeBarrier() # P1 

    grid[14][70].makeBarrier() # P2 
    
    grid[10][56].makeBarrier() # P3 

    grid[20][45].makeBarrier() # P4 

    grid[30][56].makeBarrier() # P5  



    # bottom left side of pentagon  
    grid[24][74].makeBarrier()
    grid[23][74].makeBarrier()
    grid[22][73].makeBarrier()
    grid[21][73].makeBarrier()
    grid[20][72].makeBarrier()
    grid[19][72].makeBarrier()
    grid[18][71].makeBarrier()
    grid[17][71].makeBarrier()
    grid[16][70].makeBarrier()
    grid[15][70].makeBarrier()

    # top left side of pentagon 
    grid[10][55].makeBarrier()
    grid[11][55].makeBarrier()
    grid[11][54].makeBarrier()
    grid[12][54].makeBarrier()
    grid[13][53].makeBarrier()
    grid[13][52].makeBarrier()
    grid[14][51].makeBarrier()
    grid[15][51].makeBarrier()
    grid[16][50].makeBarrier()
    grid[16][49].makeBarrier()
    grid[17][48].makeBarrier()
    grid[17][47].makeBarrier()
    grid[18][47].makeBarrier()
    grid[19][46].makeBarrier()

    # left side of pentagon  
    grid[10][57].makeBarrier()
    grid[10][58].makeBarrier()
    grid[10][59].makeBarrier()
    grid[11][60].makeBarrier()
    grid[12][61].makeBarrier()
    grid[12][62].makeBarrier()
    grid[12][63].makeBarrier()
    grid[13][64].makeBarrier()
    grid[13][65].makeBarrier()
    grid[13][66].makeBarrier()
    grid[14][67].makeBarrier()
    grid[14][68].makeBarrier()
    grid[14][69].makeBarrier()  

    # top side of pentagon  
    grid[21][46].makeBarrier()
    grid[22][46].makeBarrier()
    grid[23][47].makeBarrier()
    grid[24][47].makeBarrier()
    grid[24][48].makeBarrier()
    grid[24][49].makeBarrier()
    grid[25][49].makeBarrier()
    grid[25][50].makeBarrier()
    grid[25][51].makeBarrier()
    grid[26][51].makeBarrier()
    grid[26][52].makeBarrier()
    grid[26][53].makeBarrier()
    grid[27][54].makeBarrier()
    grid[28][54].makeBarrier()
    grid[28][55].makeBarrier()
    grid[29][55].makeBarrier()
    grid[29][56].makeBarrier()

    # bottom right side of pentagon 
    grid[30][57].makeBarrier()
    grid[30][58].makeBarrier()
    grid[30][59].makeBarrier()
    grid[29][59].makeBarrier()
    grid[29][60].makeBarrier()
    grid[29][61].makeBarrier()
    grid[29][62].makeBarrier()
    grid[28][63].makeBarrier()
    grid[28][64].makeBarrier()
    grid[28][65].makeBarrier()
    grid[28][66].makeBarrier()
    grid[27][67].makeBarrier()
    grid[27][68].makeBarrier()
    grid[27][69].makeBarrier()
    grid[27][70].makeBarrier()
    grid[26][71].makeBarrier()
    grid[26][72].makeBarrier()
    grid[26][73].makeBarrier()
    grid[26][74].makeBarrier()




    









    # Isosceles Triangle 
    grid[37][55].makeBarrier() # IT1 
    grid[41][78].makeBarrier() # IT2 
    grid[33][78].makeBarrier() # IT3  

    # grid[36][56].makeBarrier()
    for i in range(56, 62):
        grid[36][i].makeBarrier()
    for i in range(62, 67):
        grid[35][i].makeBarrier()
    for i in range(67, 73):
        grid[34][i].makeBarrier()
    for i in range(73, 78):
        grid[33][i].makeBarrier()

    for i in range(56, 62):
        grid[38][i].makeBarrier()
    for i in range(62, 67):
        grid[39][i].makeBarrier()
    for i in range(67, 73):
        grid[40][i].makeBarrier()
    for i in range(73, 78):
        grid[41][i].makeBarrier()

    for i in range(34, 41):
        grid[i][78].makeBarrier()




    


    # Quadrilateral 
    grid[43][60].makeBarrier() # Q1 
    grid[43][44].makeBarrier() # Q2 
    grid[51][41].makeBarrier() # Q3 
    grid[56][48].makeBarrier() # Q4  

    for i in range(45, 60):
        grid[43][i].makeBarrier()
    grid[44][43].makeBarrier()
    grid[45][43].makeBarrier()
    grid[46][43].makeBarrier()
    grid[47][42].makeBarrier()
    grid[48][42].makeBarrier()
    grid[49][42].makeBarrier()
    grid[50][42].makeBarrier()

    grid[52][42].makeBarrier()
    grid[52][43].makeBarrier()
    grid[53][44].makeBarrier()
    grid[53][45].makeBarrier()
    grid[54][46].makeBarrier()
    grid[55][47].makeBarrier()

    grid[55][49].makeBarrier()
    grid[54][50].makeBarrier()
    grid[53][51].makeBarrier()
    grid[52][52].makeBarrier()
    grid[51][53].makeBarrier()
    grid[50][54].makeBarrier()
    grid[49][55].makeBarrier()
    grid[48][56].makeBarrier()
    grid[47][57].makeBarrier()
    grid[46][58].makeBarrier()
    grid[45][59].makeBarrier()
    grid[44][60].makeBarrier()
    # grid[44][44].makeBarrier()
    # grid[45][44].makeBarrier()
    # grid[46][43].makeBarrier()
    # grid[47][43].makeBarrier()
    # grid[48][43].makeBarrier()
    # grid[49][42].makeBarrier()
    # grid[50][42].makeBarrier()
    # grid[51][42].makeBarrier()

    # grid[53][43].makeBarrier()
    # grid[54][43].makeBarrier()
    # grid[55][43].makeBarrier()
    # grid[56][44].makeBarrier()
    # grid[57][44].makeBarrier()
    # grid[58][44].makeBarrier()
    # grid[59][45].makeBarrier()
    # grid[60][45].makeBarrier()
    # grid[61][45].makeBarrier()

    # grid[62][47].makeBarrier()
    # grid[61][48].makeBarrier()
    # grid[61][49].makeBarrier()
    # grid[60][50].makeBarrier()

    # grid[59][52].makeBarrier()
    # grid[58][53].makeBarrier()
    # grid[57][54].makeBarrier()
    # grid[56][54].makeBarrier()
    # grid[55][55].makeBarrier()
    # grid[55][56].makeBarrier()
    # grid[54][57].makeBarrier()
    # grid[53][57].makeBarrier()
    # grid[52][58].makeBarrier()
    # grid[51][58].makeBarrier()
    # grid[50][58].makeBarrier()
    # grid[49][59].makeBarrier()
    # grid[48][59].makeBarrier()
    # grid[47][59].makeBarrier()
    # grid[46][60].makeBarrier()
    # grid[45][60].makeBarrier()
    # grid[44][60].makeBarrier()



    # Right Triangle 
    grid[56][90].makeBarrier() #RT3 

    grid[66][83].makeBarrier()# RT2 

    grid[49][70].makeBarrier() #RT1 

    # left side 

    grid[56][91].makeBarrier()
    grid[56][89].makeBarrier()
    grid[55][89].makeBarrier()
    grid[55][88].makeBarrier()
    grid[55][87].makeBarrier()
    grid[54][87].makeBarrier()
    grid[53][87].makeBarrier()
    grid[53][86].makeBarrier()
    grid[53][85].makeBarrier()
    grid[53][84].makeBarrier()
    grid[53][83].makeBarrier()
    grid[52][82].makeBarrier()
    grid[52][81].makeBarrier()
    grid[52][80].makeBarrier()
    grid[52][79].makeBarrier()
    grid[51][78].makeBarrier()
    grid[51][77].makeBarrier()
    grid[51][76].makeBarrier()
    grid[51][75].makeBarrier()
    grid[50][74].makeBarrier()
    grid[50][73].makeBarrier()
    grid[49][72].makeBarrier()
    grid[49][71].makeBarrier()

    # right side   
    # grid[49][70].makeBarrier()
    grid[50][70].makeBarrier()
    grid[51][71].makeBarrier()
    grid[52][72].makeBarrier()
    grid[53][73].makeBarrier()
    grid[54][73].makeBarrier()
    grid[55][74].makeBarrier()
    grid[56][75].makeBarrier()
    grid[57][76].makeBarrier()
    grid[58][77].makeBarrier()
    grid[59][78].makeBarrier()
    grid[60][79].makeBarrier()
    grid[61][79].makeBarrier()
    grid[62][80].makeBarrier()
    grid[63][81].makeBarrier()
    grid[64][82].makeBarrier()
    grid[65][82].makeBarrier()

    grid[65][84].makeBarrier() 
    grid[64][85].makeBarrier() 
    grid[63][86].makeBarrier() 
    grid[62][87].makeBarrier() 
    grid[61][88].makeBarrier() 
    grid[60][89].makeBarrier() 
    grid[59][90].makeBarrier() 
    grid[58][91].makeBarrier() 
    grid[57][92].makeBarrier() 



     


    # Vertical Rectangle 
    grid[62][46].makeBarrier() # R1 
    grid[62][75].makeBarrier() # R2 
    grid[77][46].makeBarrier() # R3 
    grid[77][75].makeBarrier() # R4

    for i in range(47, 75):
        grid[62][i].makeBarrier()
    for i in range(47, 75):
        grid[77][i].makeBarrier()
    for i in range(63, 77):
        grid[i][46].makeBarrier()
    for i in range(63, 77):
        grid[i][75].makeBarrier()

    

    # Hexagon  
    grid[79][78].makeBarrier() # H1 # highest 
    grid[74][83].makeBarrier() # H2 # top left 
    grid[74][88].makeBarrier() # H3 # bottom left 
    grid[79][92].makeBarrier() # H4 # lowest 
    grid[84][83].makeBarrier() # H5 # top right 
    grid[84][88].makeBarrier() # H6 # bottom right 

    grid[78][79].makeBarrier() 
    grid[77][80].makeBarrier()
    grid[76][81].makeBarrier()
    grid[75][82].makeBarrier() 

    grid[80][79].makeBarrier()
    grid[81][80].makeBarrier()
    grid[82][81].makeBarrier()
    grid[83][82].makeBarrier()

    for i in range(84, 88):
        grid[74][i].makeBarrier()

    for i in range(84, 88):
        grid[84][i].makeBarrier()
    
    grid[78][91].makeBarrier() 
    grid[77][91].makeBarrier()
    grid[76][90].makeBarrier()
    grid[75][89].makeBarrier() 

    grid[80][91].makeBarrier() 
    grid[81][91].makeBarrier()
    grid[82][90].makeBarrier() 
    grid[83][89].makeBarrier()

    









    # Kite 
    grid[80][50].makeBarrier() # K1 
    grid[86][45].makeBarrier() # K2 
    grid[92][50].makeBarrier() # K3 
    grid[86][69].makeBarrier() # K4 

    
    grid[81][51].makeBarrier()
    grid[81][52].makeBarrier()
    grid[81][53].makeBarrier()
    grid[81][54].makeBarrier()
    grid[82][55].makeBarrier()
    grid[82][56].makeBarrier()
    grid[82][57].makeBarrier()
    grid[82][58].makeBarrier()
    grid[83][59].makeBarrier()
    grid[83][60].makeBarrier()
    grid[83][61].makeBarrier()
    grid[83][62].makeBarrier()
    grid[84][63].makeBarrier()
    grid[84][64].makeBarrier()
    grid[84][65].makeBarrier()
    grid[84][66].makeBarrier()
    grid[85][67].makeBarrier()
    grid[85][68].makeBarrier()


    grid[91][51].makeBarrier()
    grid[91][52].makeBarrier()
    grid[91][53].makeBarrier()
    grid[91][54].makeBarrier()
    grid[90][55].makeBarrier()
    grid[90][56].makeBarrier()
    grid[90][57].makeBarrier()
    grid[90][58].makeBarrier()
    grid[89][59].makeBarrier()
    grid[89][60].makeBarrier()
    grid[89][61].makeBarrier()
    grid[89][62].makeBarrier()
    grid[88][63].makeBarrier()
    grid[88][64].makeBarrier()
    grid[88][65].makeBarrier()
    grid[88][66].makeBarrier()
    grid[87][67].makeBarrier()
    grid[87][68].makeBarrier()

    grid[91][49].makeBarrier()
    grid[90][49].makeBarrier()
    grid[89][48].makeBarrier()
    grid[88][47].makeBarrier() 
    grid[87][46].makeBarrier() 

    grid[81][49].makeBarrier()
    grid[82][49].makeBarrier()
    grid[83][48].makeBarrier()
    grid[84][47].makeBarrier() 
    grid[85][46].makeBarrier() 





    while True:
        draw(environment, grid, ROWS, size)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            for row in grid:
                for node in row:
                    node.updateNeighbors(grid)
            A_Star(lambda: draw(environment, grid, ROWS, size), grid, start, goal)
            break


main(environment, 800)
# python3 ez1.py
