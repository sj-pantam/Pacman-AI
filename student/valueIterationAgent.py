from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the q-values for each state.
        for i in self.mdp.getStates():
            self.values[i] = 0

        # Compute the values here.
        for i in range(self.iters):
            c = self.values.copy()
            for state in self.mdp.getStates():
                temp = []
                if self.mdp.isTerminal(state):
                    continue
                for action in self.mdp.getPossibleActions(state):
                    temp.append(self.getQValue(state, action))
                c[state] = max(temp)
            self.values = c

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values[state]

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)
    
    def getQValue(self, state, action):
        q = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            val = (self.discountRate * self.getValue(next_state))
            q += prob * (self.mdp.getReward(state, action, next_state) + val)
        return q
    
    def getPolicy(self, state):
        if self.mdp.isTerminal(state):
            return None
        else:
            a = self.mdp.getPossibleActions(state)
            return max(a, key=lambda x: self.getQValue(state, x))