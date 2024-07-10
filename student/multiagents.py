import random
from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import directions

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        dot_d = float('inf')
        ghost_d = 1
        for i in successorGameState.getGhostPositions():
            x1, y1 = i
            x2, y2 = newPosition
            dist = abs(x1 - x2) + abs(y1 - y2)
            ghost_d += dist

        scared = sum(1 / i if i != 0 else 0 for i in newScaredTimes)

        for i in successorGameState.getFood().asList():
            x1, y1 = i
            x2, y2 = newPosition
            dist = abs(x1 - x2) + abs(y1 - y2)
            dot_d = min(dot_d, dist)
        
        return successorGameState.getScore() + (1 / dot_d) - (1 / ghost_d) - scared

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        def mini_max(state, depth, agent, max_or_min):
            if state.isOver() or depth >= self.getTreeDepth():
                return self.getEvaluationFunction()(state), directions.Directions.STOP

            if max_or_min:
                m = float("-inf")
            else:
                m = float("inf")

            d = directions.Directions.STOP

            for i in state.getLegalActions(agent):
                if i != directions.Directions.STOP:
                    n = agent + 1
                    v, _ = mini_max(state.generateSuccessor(agent, i), depth + 1, n, not max_or_min)

                    if (max_or_min and v > m):
                        m = v
                        d = i
                    if (not max_or_min and v < m):
                        m = v
                        d = i

            return m, d

        _, action = mini_max(state, 0, 0, True)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        def ab(gameState, depth, agent, alpha, beta, max_or_min):
            if gameState.isOver() or depth >= self.getTreeDepth():
                return self.getEvaluationFunction()(gameState), directions.Directions.STOP

            a_or_b = float("-inf") if max_or_min == 1 else float("inf")
            r = 0
            for i in gameState.getLegalActions(agent):
                if i != directions.Directions.STOP:
                    s = gameState.generateSuccessor(agent, i)
                    nex = agent + 1
                    v, _ = ab(s, depth + 1, nex, alpha, beta, 0)
                    if max_or_min == 1:
                        if v > a_or_b:
                            a_or_b = v
                            r = i
                        alpha = max(alpha, v)
                    else:
                        if v < a_or_b:
                            a_or_b = v
                            r = i
                        beta = min(beta, v)

                    if beta <= alpha:
                        break

            return a_or_b, r

        _, x = ab(gameState, 0, 0, float("-inf"), float("inf"), 1)
        return x


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        def emax(gameState, depth, agent):
            if gameState.isOver() or depth >= self.getTreeDepth():
                return self.getEvaluationFunction()(gameState), directions.Directions.STOP
            
            if agent == 0:
                mv = float("-inf")
                action = directions.Directions.STOP
                for i in gameState.getLegalActions(agent):
                    if i != directions.Directions.STOP:
                        n = (agent + 1) % gameState.getNumAgents()
                        v, _ = emax(gameState.generateSuccessor(agent, i), depth + 1, n)
                        if v > mv:
                            mv = v
                            action = i
                return mv, action
            else:
                maximum = 0
                a_list = []
                for i in gameState.getLegalActions(agent):
                    if i != directions.Directions.STOP:
                        n = (agent + 1) % gameState.getNumAgents()
                        v, _ = emax(gameState.generateSuccessor(agent, i), depth, n)
                        maximum += v
                        a_list.append(i)
                return maximum / len(a_list), random.choice(a_list)

        _, action = emax(gameState, 0, 0)
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """
    newPosition = currentGameState.getPacmanPosition()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.getScaredTimer() for ghostState in GhostStates]

    dot_d = float('inf')
    ghost_d = 1
    
    for i in currentGameState.getFood().asList():
        x1, y1 = i
        x2, y2 = newPosition
        dist = abs(x1 - x2) + abs(y1 - y2)
        dot_d = min(dot_d, dist)

    for i in currentGameState.getGhostPositions():
        x1, y1 = i
        x2, y2 = newPosition
        dist = abs(x1 - x2) + abs(y1 - y2)
        ghost_d += dist
    
    scared = sum(1 / i if i != 0 else 0 for i in ScaredTimes)

    c = len(currentGameState.getCapsules())
    
    return currentGameState.getScore() + (1 / dot_d) - (1 / ghost_d) - scared - c


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
