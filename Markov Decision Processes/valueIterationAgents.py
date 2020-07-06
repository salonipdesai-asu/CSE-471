# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0 (the values are initialized to zero)
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #print(type(self.iterations))
        States = self.mdp.getStates()
        #print(States) 
        kminusone_Values = self.values
        
        for iterable in range(self.iterations):
            '''when a state's value is updated in iteration k based on the values of its successor states, the successor state values used in the value update computation should be those from iteration k-1 (even if some of the successor states had already been updated in iteration k)'''
            kminusone_Values = util.Counter()
            #for each state, for each action find values, choose the action with maximum value
            for state in States:
                 Actions = self.mdp.getPossibleActions(state)
                 if not Actions:
                     self.values[state] = 0
                 #print(state, Actions) 
                 val=[]
                 for action in Actions:
                     #print(state,action)
                     Tmodel = self.mdp.getTransitionStatesAndProbs(state,action)
                     #print(Tmodel)
                     nextStates =[]
                     
                     nextStateProb =[]
                     for t in Tmodel:
                         nextStates.append(t[0])
                         nextStateProb.append(t[1])
                     #print(nextStates)
                     #print(nextStateProb)
                     v = 0.0
                     for i in range(len(nextStates)):
                         Reward = self.mdp.getReward(state,action, nextStates[i])
                         #print( Reward)
                         discountF = self.discount
                         #print(discountF)
                         vdash =   self.values[nextStates[i]]
                         #print(vdash)
                         v =  v + (nextStateProb[i] * (Reward + (discountF * vdash)))
                         #print("value", v)
                     
                     val.append(v)
                 
                      
                 if val:
                    maximum= max(val)
                    kminusone_Values[state] =  maximum
                  
            for state in States:
                self.values[state] = kminusone_Values[state]       
                
        #print("Iterationx:", self.values)                         
                
        


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Tmodel = self.mdp.getTransitionStatesAndProbs(state,action)
        nextStates =[]
        nextStateProb =[]
        for t in Tmodel:
            nextStates.append(t[0])
            nextStateProb.append(t[1])
        qv = 0.0
        for i in range(len(nextStates)):
            Reward = self.mdp.getReward(state,action, nextStates[i])
            #print("Reward:", Reward)
            discountF = self.discount
            #print(discountF)
            qdash =   self.values[nextStates[i]]
            #print(qdash)
            qv =  qv + (nextStateProb[i] * (Reward + (discountF * qdash)))
            #print(qv)
        return qv
        
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        Actions = self.mdp.getPossibleActions(state)
        actionValue =[]
        best_action = None
        if not Actions:
            return best_action
        
        for action in Actions:
            qvalue = self.computeQValueFromValues(state,action)
            actionValue.append((action, qvalue))
        
        #print(actionValue)
        actions =[]
        avalues =[]    
        if actionValue:
            for action,value in actionValue:
                      actions.append(action)
                      avalues.append(value)          
            max_value_index = avalues.index(max(avalues))
            best_action =  actions[max_value_index]
                    
        #print("Best Action:", best_action)
        return best_action
        
       
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        States = self.mdp.getStates()
        #print("States:", States)
        leng = len(States)
        for iterable in range(self.iterations):
            state = States[(iterable % leng )]
            #print("Iterx:", state)
            Actions = self.mdp.getPossibleActions(state)
            if not Actions:
                self.values[state] = 0
                #print(state, Actions) 
            val=[]
            for action in Actions:
                #print(state,action)
                Tmodel = self.mdp.getTransitionStatesAndProbs(state,action)
                #print(Tmodel)
                nextStates =[] 
                nextStateProb =[]
                for t in Tmodel:
                    nextStates.append(t[0])
                    nextStateProb.append(t[1])
                #print(nextStates)
                #print(nextStateProb)
                v = 0.0
                for i in range(len(nextStates)):
                    Reward = self.mdp.getReward(state,action, nextStates[i])
                    #print( Reward)
                    discountF = self.discount
                    #print(discountF)
                    vdash =   self.values[nextStates[i]]
                    #print(vdash)
                    v =  v + (nextStateProb[i] * (Reward + (discountF * vdash)))
                    #print("value", v)
                val.append(v)
                 
                      
            if val:
                self.values[state] =max(val)
            #print(self.values[state])     
            
                
        
        

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        #self.values= util.Counter()

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        States = self.mdp.getStates()
        predecessors={}
        for state in States:
            predecessors[state]= set([])
        
          
        for state in States:
            
            Actions = self.mdp.getPossibleActions(state)
            for action in Actions:
                Tmodel = self.mdp.getTransitionStatesAndProbs(state,action)
                for t in Tmodel:
                    print(state,action, Tmodel)
                    if t[1] > 0:
                        temp = predecessors[t[0]]
                        temp.add(state)
                        predecessors[t[0]] = temp
        for state in States:
            print(predecessors[state])
            
            
            
        pq = util.PriorityQueue()
        
        for state in States:
            qvalues =[]
            Actions = self.mdp.getPossibleActions(state)
            print("Actions:" , Actions)
            for action in Actions:
                qv = self.computeQValueFromValues(state,action)
                print(qv)
                qvalues.append(qv)
            if Actions:
                max_qv = max(qvalues)
                print(max_qv)
                diff = abs(self.values[state]- max_qv)
                print(state,diff)
                pq.push(state,(-1 * diff))
        #print("I came till here")               
        for iterable in range(0, self.iterations):
            print("Itex")
            if pq.isEmpty() != True:
                popped = pq.pop()
                if self.mdp.isTerminal(popped) != True:
                    Actions = self.mdp.getPossibleActions(popped)
                
                    #print(state, Actions) 
                    val=[]
                    for action in Actions:
                        #print(state,action)
                        Tmodel = self.mdp.getTransitionStatesAndProbs(popped,action)
                        print(state, action,Tmodel)
                        nextStates =[] 
                        nextStateProb =[]
                        for t in Tmodel:
                            nextStates.append(t[0])
                            nextStateProb.append(t[1])
                            print(nextStates)
                            print(nextStateProb)
                        v = 0.0
                        for i in range(len(nextStates)):
                            Reward = self.mdp.getReward(popped,action, nextStates[i])
                            print( Reward)
                            discountF = self.discount
                            print(discountF)
                            vdash =   self.values[nextStates[i]]
                            print(vdash)
                            v =  v + (nextStateProb[i] * (Reward + (discountF * vdash)))
                            print("value", v)
                            val.append(v)
                 
                      
                    
                    self.values[popped] =max(val)
                    print(popped,self.values[popped])
                    
                         
                prepopped= predecessors[popped]  
                for predecessor in prepopped:
                    print("predecessor:", predecessor)
                    qvalues =[]
                    Actions = self.mdp.getPossibleActions(predecessor)
                    for action in Actions:
                        qv = self.computeQValueFromValues(predecessor,action)
                        qvalues.append(qv)
                    max_qv = max(qvalues)
                    diff = abs(self.values[predecessor]-max_qv)
                    if diff > self.theta:
                        pq.update(predecessor, -diff)
                      
        
        
        
        
        
