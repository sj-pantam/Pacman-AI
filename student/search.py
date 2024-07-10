"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    visited = []
    stack = Stack()

    stack.push((problem.startingState(), []))
    while stack:
        node, path = stack.pop()
        if problem.isGoal(node):
            return path
        if node not in visited:
            visited.append(node)
            for i in problem.successorStates(node):
                if i[0] not in visited:
                    stack.push((i[0], path + [i[1]]))

    raise NotImplementedError()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    visited = []
    queue = Queue()

    queue.push((problem.startingState(), []))
    while queue:
        node, path = queue.pop()
        if problem.isGoal(node):
            return path
        if node not in visited:
            visited.append(node)
            for i in problem.successorStates(node):
                if i[0] not in visited:
                    queue.push((i[0], path + [i[1]]))

    raise NotImplementedError()

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    visited = []
    pq = PriorityQueue()

    pq.push((problem.startingState(), []), 0)
    while pq:
        node, path = pq.pop()
        if problem.isGoal(node):
            return path
        if node not in visited:
            visited.append(node)
            for i in problem.successorStates(node):
                if i[0] not in visited:
                    pq.push((i[0], path + [i[1]]), problem.actionsCost(path + [i[1]]) + i[2])

    raise NotImplementedError()

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    visited = []
    pq = PriorityQueue()

    pq.push((problem.startingState(), []), 0)
    while pq:
        node, path = pq.pop()
        if problem.isGoal(node):
            return path
        if node not in visited:
            visited.append(node)
            for i in problem.successorStates(node):
                h = heuristic(i[0], problem)
                pq.push((i[0], path + [i[1]]), problem.actionsCost(path + [i[1]]) + h)
    raise NotImplementedError()