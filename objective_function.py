"""NACO assignment 22/23.

By Björn Keyser, Jimmy Oei, and Zoë Breed

## Installing requirements
    pip install ioh>=0.3.3
"""

import typing
import shutil
import random
import numpy as np
import ioh
import csv

from implementation import GeneticAlgorithm


class CellularAutomata:
    """Skeleton CA, you should implement this."""
    
    def __init__(self, rule_number: int, upbound: int):
        """Intialize the cellular automaton with a given rule number"""       
        # upbound: 2 = binary and upbound: 3 = ternary
        self.upbound = upbound

        #gives the set of rules
        if self.upbound == 2:
            self.ruleset = [int(x) for x in np.binary_repr(rule_number, width=8)]
        else: 
            self.ruleset = [int(x) for x in np.base_repr(rule_number, base=3)]
            while len(self.ruleset) != 27:
                self.ruleset.append(0)

        self.ruleset.reverse()

    def __call__(self, c0: typing.List[int], t: int) -> typing.List[int]:
        """Evaluate for T timesteps. Return Ct for a given C0."""
        for _ in range(t):
            c0 = self.compute_state(c0)

        # print("t complete: ", t)

        return c0

    # Compute the next state
    def compute_state(self, c0: typing.List[int]):
        cn = list(c0)

        # accounts for borders
        cn[0] = self.get_rule(0, c0[0], c0[1])
        cn[-1] = self.get_rule(c0[-2], c0[-1], 0)
        
        i = 1
        while i <= len(c0[1:-1]):
            left = c0[i-1]
            middle = c0[i]
            right = c0[i+1]
            newstate = self.get_rule(left, middle, right)
            # print(i, middle, newstate, " - ", left, right)
            cn[i] = newstate
            i += 1

        return cn
    
    # Looks up a new state from the rule set
    def get_rule(self, a, b, c):
        # convert neighborhood binary into decimal 
        s = str(a) + str(b) + str(c)
        index = int(s, self.upbound)
        
        # use that value as the index into the ruleset array
        return self.ruleset[index]   
  
def make_objective_function(ct, rule, t, upbound, similarity_method):
    '''Create a CA objective function.'''
    
    if similarity_method == 1:
        def similarity(ct: typing.List[int], ct_prime: typing.List[int]) -> float:
            """Count number of similar items"""

            count = 0
            for i in range(len(ct)):
                if(ct[i] == ct_prime[i]):
                    count += 1
                
            #print("Count:", count)
            return count
    else:
        def similarity(ct: typing.List[int], ct_prime: typing.List[int]) -> float:
            """You should implement this"""

            return random.normalvariate(0, 10)
            
    # individual is GA = c0_prime
    # calculates similarity between c0_prime and c0 of the CA
    def objective_function(c0_prime: typing.List[int]) -> float:
        """Skeleton objective function. 
        
        You should implement a method  which computes a similarity measure 
        between the state you are given with the state your automata reaches after t steps
        for the ct state given in the sup. material.

        Parameters
        ----------
        c0_prime: list[int] | np.ndarray
            A suggested c0 state
        
        Returns
        -------
        float
            The similarity of ct_prime to the true ct state of the CA           
        """

        ca = CellularAutomata(rule, upbound)
        ct_prime = ca(c0_prime, t)

        # print("ct_prime:", ct_prime)
        return similarity(ct, ct_prime)

    return objective_function

def example(nreps=10):
    """An example of wrapping a objective function in ioh and collecting data
    for inputting in the analyzer."""

    # which state you want to use nr between 1 - 10
    nr = 6
    line = 0
    with open ('ca_input.csv', newline = '', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if line == nr:
                upbound = int(row[0])
                rule = int(row[1])
                t = int(row[2])
                ct = eval(row[3])
            line += 1
            
    # Create an objective function
    objective_function = make_objective_function(ct, rule, t, upbound, 1)
    
    # Wrap objective_function as an ioh problem
    problem = ioh.wrap_problem(
        objective_function,
        name="objective_function_ca_1", # Give an informative name 
        dimension=60, # Should be the size of ct (er stond 10 maar moet dus 60 zijn?)
        problem_type="Integer",
        optimization_type=ioh.OptimizationType.MAX,
        lb=0,
        ub=1,         # 1 for 2d, 2 for 3d
    )

    # Attach a logger to the problem
    logger = ioh.logger.Analyzer()
    problem.attach_logger(logger)

    # run your algoritm on the problem
    for _ in range(nreps):
        algorithm = GeneticAlgorithm(100, upbound)
        algorithm(problem)

        ca = CellularAutomata(rule, upbound)
        ct_prime = ca(problem.state.current_best.x, t)
        print("current_best ct_prime: ", ct_prime)
        print("fit: ", objective_function(problem.state.current_best.x))
        problem.reset()

    logger.close()

    shutil.make_archive("ioh_data", "zip", "ioh_data")
    shutil.rmtree("ioh_data")


if __name__ == "__main__":
    example()
