"""NACO assignment 22/23.

By Björn Keyser, Jimmy Oei, and Zoë Breed

## Installing requirements
    pip install ioh>=0.3.3
"""

import typing
import shutil
import random
import numpy as np
# import difflib
import ioh

from implementation import GeneticAlgorithm


class CellularAutomata:
    """Skeleton CA, you should implement this."""
    
    def __init__(self, rule_number: int):
        """Intialize the cellular automaton with a given rule number"""
        # test cell
        self.cells = [1] * 20

        # upbound: 2 = binary and upbound: 3 = ternary
        self.upbound = 2


        # gives the set of rules
        # self.ruleset = [int(x) for x in np.binary_repr(rule_number, width=8)]
        # self.ruleset.reverse()

        # test 
        self.ruleset = [0,1,0,0,1,0,1,0]
        self.ruleset.reverse()
     
        
    def __call__(self, c0: typing.List[int], t: int) -> typing.List[int]:
        """Evaluate for T timesteps. Return Ct for a given C0."""
        for _ in range(t):
            self.compute_state()

        print("t complete: ", t)

        return self.cells

    # Compute the next state
    def compute_state(self):
        newcells = list(self.cells)
        print(newcells)
        
        print("currCells: ", self.cells)

        # accounts for borders
        newcells[0] = self.get_rule(0, self.cells[0], self.cells[1])
        newcells[-1] = self.get_rule(self.cells[-2], self.cells[-1], 0)
        i = 1

        while i <= len(self.cells[1:-1]):
            left = self.cells[i-1]
            middle = self.cells[i]
            right = self.cells[i+1]
            newstate = self.get_rule(left, middle, right)
            # print(i, middle, newstate, " - ", left, right)
            # print(self.cells)
            newcells[i] = newstate
            i += 1
        self.cells = newcells
        print("newCells: ", self.cells)
    
    # Looks up a new state from the rule set
    def get_rule(self, a, b, c):
        # convert neighborhood into decimal 
        if self.upbound == 2:
            s = str(a) + str(b) + str(c)
            index = int(s, self.upbound)
        else:
            index = a + b + c
        
        # use that value as the index into the ruleset array
        return self.ruleset[index]

#     def test(self):
#         self.compute_state()
    
# def test():
#     t = CellularAutomata()
#     t.test()
        
  
def make_objective_function(ct, rule, t, similarity_method):
    '''Create a CA objective function.'''
    
    if similarity_method == 1:
        def similarity(ct: typing.List[int], ct_prime: typing.List[int]) -> float:
            """Count number of similar items"""
            
            count = 0
            for i in range(len(ct)):
                count += (ct[i] == ct_prime[i])

            print("Count:", count)
            len(ct)
            return count/len(ct)
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

        ca = CellularAutomata(rule)
        ct_prime = ca(c0_prime, t)
        return similarity(ct, ct_prime)

    return objective_function

def example(nreps=10):
    """An example of wrapping a objective function in ioh and collecting data
    for inputting in the analyzer."""

    # ct, rule, t = None, None, None  # Given by the sup. material
    # testing 
    
    ct = [1] * 20
    rule = 30
    t = 3
  
    # Create an objective function
    objective_function = make_objective_function(ct, rule, t, 1)
    
    # Wrap objective_function as an ioh problem
    problem = ioh.wrap_problem(
        objective_function,
        name="objective_function_ca_1", # Give an informative name 
        dimension=10, # Should be the size of ct
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
        algorithm = GeneticAlgorithm(10)
        algorithm(problem)
        problem.reset()

    logger.close()

    shutil.make_archive("ioh_data", "zip", "ioh_data")
    shutil.rmtree("ioh_data")


if __name__ == "__main__":

    example()
