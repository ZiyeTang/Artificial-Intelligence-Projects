# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

import queue

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    row = maze.size.y
    col = maze.size.x
    sr, sc = maze.start
    er, ec = maze.waypoints[0]
    if sr == er and sc == ec:
        return [(sr, sc)]

    q = queue.Queue()
    visited = [[0]*col for _ in range(row)]

    visited[sr][sc] = 1
    parent = {}
    q.put((sr,sc))
    while q.qsize() > 0:
        cur = q.get()
        if cur[0] == er and cur[1] == ec:
            path = [cur]
            temp = cur
            while temp in parent:
                path.insert(0, parent[temp])
                temp = parent[temp]
            return path
        
        for next in maze.neighbors(cur[0], cur[1]):
            if visited[next[0]][next[1]] == 0:
                parent[next] = cur
                visited[next[0]][next[1]] = 1
                q.put(next)

    return []

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    row = maze.size.y
    col = maze.size.x
    sr, sc = maze.start
    er, ec = maze.waypoints[0]
    
    q = queue.PriorityQueue()
    q.put((abs(er-sr) + abs(ec-sc), 0, sr,sc))

    visited = [[0]*col for _ in range(row)]
    
    parent = {}
    while q.qsize() > 0:
        cur = q.get()
        if visited[cur[2]][cur[3]] == 1:
            continue
        visited[cur[2]][cur[3]] = 1
        if cur[2] == er and cur[3] == ec:
            temp = (cur[2],cur[3])
            path = [temp]
            while temp in parent:
                path.insert(0, parent[temp])
                temp = parent[temp]
            return path
        
        for next in maze.neighbors(cur[2], cur[3]):
            if visited[next[0]][next[1]] == 0:
                parent[next] = (cur[2], cur[3])
                q.put((cur[1] + 1 + abs(er-next[0]) + abs(ec-next[1]), cur[1] + 1,next[0], next[1]))

    return []

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    return []
