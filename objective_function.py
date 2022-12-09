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

from implementation import GeneticAlgorithm


class CellularAutomata:
    """Skeleton CA, you should implement this."""
    

    def __init__(self, rule_number: int):
        """Intialize the cellular automaton with a given rule number"""
        self.cell = []
        self.cells = []
        # gives the set of rules
        self.ruleset = [int(x) for x in np.binary_repr(rule_number, width=8)]

    def __call__(self, c0: typing.List[int], t: int) -> typing.List[int]:
        """Evaluate for T timesteps. Return Ct for a given C0."""

    def setup(cells, cell):
        newcells = [cells.lenght]
        for i in cells:
            left = cell[i-1]
            middle = cell[i]
            right = cell[i+1]
            newstate = rules(left, middle, right, rule)
            newcells[i] = newstate
            return
    
    def rules(a, b, c, ruleset):
        if   a == 1 and b == 1 and c == 1: return ruleset[0]
        elif a == 1 and b == 1 and c == 0: return ruleset[1]
        elif a == 1 and b == 0 and c == 1: return ruleset[2]
        elif a == 1 and b == 0 and c == 0: return ruleset[3]
        elif a == 0 and b == 1 and c == 1: return ruleset[4]
        elif a == 0 and b == 1 and c == 0: return ruleset[5]
        elif a == 0 and b == 0 and c == 1: return ruleset[6]
        elif a == 0 and b == 0 and c == 0: return ruleset[7]
        return 0

  
def make_objective_function(ct, rule, t, similarity_method):
    '''Create a CA objective function.'''
    
    if similarity_method == 1:
        def similarity(ct: typing.List[int], ct_prime: typing.List[int]) -> float:
            """You should implement this"""

            return random.uniform(0, 100)
    else:
        def similarity(ct: typing.List[int], ct_prime: typing.List[int]) -> float:
            """You should implement this"""

            return random.normalvariate(0, 10)

    
    def objective_function(c0_prime: typing.List[int]) -> float:
        """Skeleton objective function. 
        
        You should implement a method  which computes a similarity measure 
        between c0_prime a suggested by your GA, with the true c0 state 
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

    ct, rule, t = None, None, None  # Given by the sup. material

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
