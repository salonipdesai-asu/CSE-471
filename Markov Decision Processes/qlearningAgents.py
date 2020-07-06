# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        #a table of action values, indexed by state and action
        self.values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #default is 0 anyway (dict- Counter)
        saTuple = (state,action)
        return self.values[saTuple]
        #util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        Actions = self.getLegalActions(state)
        qvs=[]
        if not Actions:
            return 0.0
        #print(Actions)
        for action in Actions:
            qv = self.getQValue(state,action)
            qvs.append(qv)
        
        return max(qvs)
            
        
        
        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = None
        if not self.getLegalActions(state):
            return bestAction
        Actions = self.getLegalActions(state)
        aqvs=[]
        for action in Actions:
            qaction = self.getQValue(state,action)
            aqvs.append((action, qaction))
        qvs=[]
        acs =[]
        for aqv in aqvs:
            qvs.append(aqv[1])
            acs.append(aqv[0])
            
        maximum_qv = max(qvs)
        max_qv_id = qvs.index(maximum_qv)
        
        #if two actions ahve the same maximum value
        optimize=[]
        for i in range(len(acs)):
            if qvs[i] == maximum_qv:
                optimize.append(acs[i])
        
        j = random.randint(0,len(optimize)-1)
        
        #print(optimize)
        #print(j)
        
        return optimize[j]
        
            
        #util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        rand_action_prob  = self.epsilon
        #best_action_prob = (1 - self.epsilon)
        
        if not legalActions:
            return action
        #returns true if r< rand_action_prob / returns true with prob p
        boolean = util.flipCoin(rand_action_prob)
        if boolean == True:
            action = random.choice(legalActions)
            return action
        
        action = self.computeActionFromQValues(state)
        #util.raiseNotDefined()

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        
        
        q= self.values[(state,action)]
        c1 = (1 - self.alpha)
        c2 = self.alpha
        qupdated = c1 * q + c2 * sample
        self.values[(state,action)] = qupdated
        
        
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        
        q=0.0
        featureVector = self.featExtractor.getFeatures(state,action)
        
        for feature in featureVector:
            q = q + self.weights[feature] * featureVector[feature]
        #print(q)
        return q
        
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        weight = self.weights[(state,action)]
        difference = (reward + (self.discount * self.computeValueFromQValues(nextState))) - self.getQValue(state,action)
        featureVector = self.featExtractor.getFeatures(state,action)
        additive_component=0
        for feature in featureVector:
            additive_component = self.alpha * difference * featureVector[feature]
            self.weights[feature] = self.weights[feature] + additive_component
            
        
        
            
        
        
        
        
        
        
        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print(self.weights)