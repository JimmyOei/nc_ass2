"""NACO assignment 22/23.

By Björn Keyser, Jimmy Oei, and Zoë Breed

This file contains the skeleton code required to solve the first part of the 
assignment for the NACO course 2022. 

## Installing requirements
    pip install ioh>=0.3.3
"""

import random
import shutil
import ioh
import numpy as np

class GeneticAlgorithm:
    """An implementation of the Genetic Algorithm."""

    def __init__(self, budget: int) -> None:
        """Construct a new GA object.

        Parameters
        ----------
        budget: int
            The maximum number objective function evaluations
            the GA is allowed to do when solving a problem.

        pop_size: int
            The number of individuals in set/array

        mating_selection: def
            Function containing the modal operator for mating_selection.

        crossover: def
            Function containing the modal operator for crossover.

        mutation: def
            Function containing the modal operator for mutation.

        environmental_selection: def
            Function containing the modal operator for environmental_selection.

        Notes
        -----
        *   You can add more parameters to this constructor, which are specific to
            the GA or one of the (to be implemented) operators, such as a mutation rate.
        """
    
        self.upbound = 2
       
        self.budget = budget
        self.pop_size = 1000

        self.mutation_rate = 0.4
        self.cross_rate = 0.7
        self.tournament_size = 128

       
        # Variables containing the modal operator the GA will use
        self.mating_selection = self.mat_selection_tournament
        self.crossover = self.crossover_uniform
        self.mutation = self.mutation_point
        self.environmental_selection = self.env_selection_best_of_both

    def __call__(self, problem: ioh.problem.Integer) -> ioh.IntegerSolution:
        """Run the GA on a given problem instance.

        Parameters
        ----------
        problem: ioh.problem.Integer
            An integer problem, from the ioh package. This version of the GA
            should only work on binary/discrete search spaces.

        Notes
        -----
        *   This is the main body of the GA.
        """

        print("optimum:",(problem.optimum), problem.optimum.y)

        # initialize
        pop = self.initialize_population(self.upbound, n=problem.meta_data.n_variables)

        # first evaluation
        fit = self.calculate_fitness(pop, problem)
        gen = 1
    
        for e in range(self.budget - self.pop_size):
            if e % self.pop_size == 0:
                # Generates a new population with GA operators
                new_pop = self.generate_population(fit, pop)
                new_fit = self.calculate_fitness(pop, problem)

                # Environmental selection
                pop = self.environmental_selection(pop, new_pop, fit, new_fit)
                fit = self.calculate_fitness(pop, problem)
                gen = gen + 1

                # Termination criterion
                if problem.state.current_best.y == problem.optimum.y:
                    break

        print(f"curr best: {problem.state.current_best}")
        return problem.state.current_best

    def initialize_population(self, upbound, n):
        """Generates a randomly initialized population

        Parameters
        ----------
        n: int
            The dimensionality of the search space

        Notes
        -----
        *   A solution candidate set is created, consisting of M number of solution
            candidates of the form: solution candidate x_i = (x_1, x_2, ..., x_n) with
            i ∈ {1 . . . M}, which are all initialized randomly.
        """

        pop = []
        for _ in range(self.pop_size):
            pop.append(np.random.randint(0, upbound, n))
            
        return pop

 
    def generate_population(self, fit, pop):
        """ generates a new population

        Parameters
        ----------
        fit: list
            List with all of the fitness values of the population
        
        pop: list
            List containing the population

        Notes
        -----
        *   Creates a new population out of the current population pop
            using the modular operators for mating selection, crossover
            and mutation. The function returns this new population.
        """
        new_pop = []
        while len(new_pop) < self.pop_size:
            # select the parents
            p1 = self.mating_selection(fit, pop)
            p2 = self.mating_selection(fit, pop)
     
            # crossover
            if np.random.rand() < self.cross_rate:
                child = self.crossover(p1, p2)
            else:
                child = p1

            # mutation
            if np.random.rand() < self.mutation_rate:
                child = self.mutation(child, self.upbound)
            new_pop.append(child)

        return new_pop

    @staticmethod
    def calculate_fitness(pop, problem):
        """ Calculates the fitness of a population

        Parameters
        ----------
        pop: list
            List containing the population

        problem: ioh.problem.Integer
            An integer problem, from the ioh package.

        Notes
        -----
        *   Calculates the fitness of each individual in the population
            by calling the problem with it and returns a list of all
            the fitnesses.
        """
        fit = []
        for gene in pop:
            fitness = problem(gene)
            fit.append(fitness)

        return fit

    """
    Mating selection modal operators
        Parameters
        ----------
        fit: list
         List with all of the fitnes values of the population

        pop: list
         List containing the population
    
        Returns
        -------
        A gene that is selected out of the population
    """
    @staticmethod
    def mat_selection_roulette_wheel(fit, pop):
        """ Mating selection: roulette wheel

        Notes
        -----
        *   Implements Roulette Wheel selection
        """
        # sort the individual based on fitness
        sorted_fitpop = sorted(zip(fit, pop), reverse=True, key=lambda pair: pair[0])

        sum_fitness = abs(int(sum(fit)))
        f = random.randint(0, sum_fitness) 
        for fitness, gene in sorted_fitpop:
            if f < abs(fitness):
                return gene
            f -= abs(fitness)
        return sorted_fitpop[-1][1]

    def mat_selection_tournament(self, fit, pop):
        """ Mating selection: tournament selection

        Notes
        -----
        *   Implements tournament selection, where tournament_size is how
            many genes are randomly selected to include in the tournament.
        """
        # randomly select tournament_size many genes from population
        genes = random.sample(list(zip(fit, pop)), k=self.tournament_size)
        # sort it to select the gene with the highest fitness
        sorted_genes = sorted(genes, reverse=True, key=lambda pair: pair[0])
        return sorted_genes[0][1]

    """
    Crossover modal operators
        Parameters
        ----------
        p1: list
          genotype 1
          
        p2: list
          genotype 2
          
        Returns
        -------
        A child genotype created from the parents p1 and p2
    """
    @staticmethod
    def crossover_single_point(p1, p2):
        """ Crossover: single point

        Notes
        -----
        *   Takes two parents and combines them by choosing a point
            on each genotype (bitstring) to split each list in two two,
            and joining the first sublist from one genotype with the second
            sublist of the second genotype.
        """

        # create random splicing point
        split = random.randint(1, len(p1 - 1))
        p1a = p1[0:split]
        p2b = p2[split:]
        child = np.append(p1a, p2b)
        return child

    """
    Mutation modal operators
        Parameters
        ---------- 
        gene: list
            One bit string
            
        Returns
        -------
        The gene after applying mutation
    """
    @staticmethod
    def crossover_uniform(p1, p2):
        """ Crossover: uniform

        Notes
        -----
        *   Takes two parents and combines them by a 50 procent probility of
            selecting a gene from the a parent.
        """
      
        child = []
        for i in range(len(p1)):
            # probability that decides from which parent the gene will be chosen
            prob = random.randint(0, 1)
            if prob == 0:
                child.append(p1[i])
            else: 
                child.append(p2[i])
        return child

    """
    Mutation modal operators
        Parameters
        ---------- 
        gene: list
            One bit string
            
        Returns
        -------
        The gene after applying mutation
    """

    @staticmethod
    def mutation_point(gene, upbound):
        """ Mutation: point mutation

        Notes
        -----
        *   Flips one bit in the gene.
        """
        n = len(gene)
        j = random.randint(0, n - 1)
        # kan dus ook zelfde blijven...
        gene[j] = np.random.randint(0, upbound)

        
        return gene

    @staticmethod
    def mutation_swap(gene, upbound):
        """ Mutation: swap mutation

        Notes
        -----
        *   Swaps two bits in the gene with each other.
        """
        n = len(gene)
        j = random.randint(0, n - 1)
        k = random.randint(0, n - 1)
        tmp = gene[j]
        gene[j] = gene[k]
        gene[k] = tmp
        return gene

    ##insert mutation (pick two bits at random, move the second to follow the first
    # shifting the rest)
    # elif m == 2:
    #     j = random.randint(0, n - 2)
    #     k = random.randint(j, n - 1)
    #     individu[k] = individu[j+1]
    #     return

    ## inversion mutation (pick to bits at random and invert substring between them)
    # else:
    #     j = random.randint(0, n - 2)
    #     k = random.randint(j, n - 1)
    #     for i in range (k-j):
    #       individu[]

    """
    Environmental selection modal operators
        Parameters
        ----------
        new_pop: list
            List containing the new population

        pop: list
            List containing the population

        fit: list
            List with all of the fitness values of the population

        new_fit: list
            List with all of the fitness values of the new population
            
        Returns
        -------
        List containing the new population after selecting from pop and new_pop
    """
    @staticmethod
    def env_selection_best_half(pop, new_pop, fit, new_fit):
        """ Environmental selection: best halves of both

        Notes
        -----
        *   Sorts the two populations based on their fitness and selects
            the top best halves of both to create the new population
            which is then returned.
        """
        # sort the populations based on fitness
        pop_old = [x for _,x in sorted(zip(fit, pop), reverse=True, key=lambda pair: pair[0])]
        pop_new = [x for _,x in sorted(zip(new_fit, new_pop), reverse=True, key=lambda pair: pair[0])]
        # return new population with best fitnesses of both
        return pop_old[0:len(pop_old)//2] + pop_new[0:len(pop_new)//2]

    @staticmethod
    def env_selection_best_of_both(pop, new_pop, fit, new_fit):
        """ Environmental selection: best genes of both
        Notes
        -----
        *   Returns a population of the best genes of the genes from both pop and new_pop
        """
        # sort the populations based on fitness
        fit.extend(new_fit)
        pop.extend(new_pop)
        sorted_pops = [x for _,x in sorted(zip(fit, pop), reverse=True, key=lambda pair: pair[0])]
        # return new population with best fitnesses of both
        return sorted_pops[0:len(sorted_pops)//2]
    
    
def test_algorithm(dimension, instance=1):
    """A function to test if your implementation solves a OneMax problem.

    Parameters
    ----------
    dimension: int
        The dimension of the problem, i.e. the number of search space variables.

    instance: int
        The instance of the problem. Trying different instances of the problem,
        can be interesting if you want to check the robustness, of your GA.
    """

    budget = int(dimension * 5e3)
    problem = ioh.get_problem("OneMax", instance, dimension, "Integer")
    ga = GeneticAlgorithm(budget)
    solution = ga(problem)

    print("GA found solution:\n", solution)

    assert problem.state.optimum_found, "The optimum has not been reached."
    assert problem.state.evaluations <= budget, (
        "The GA has spent more than the allowed number of evaluations to "
        "reach the optimum."
    )

    print(f"OneMax was successfully solved in {dimension}D.\n")


def collect_data(dimension=100, nreps=5):
    """OneMax + LeadingOnes functions 10 instances.

    This function should be used to generate data, for A1.

    Parameters
    ----------
    dimension: int
        The dimension of the problem, i.e. the number of search space variables.

    nreps: int
        The number of repetitions for each problem instance.
    """

    budget = int(dimension * 5e2)
    suite = ioh.suite.PBO([1, 2], list(range(1, 11)), [dimension])
    logger = ioh.logger.Analyzer(algorithm_name="GeneticAlgorithm3")
    suite.attach_logger(logger)

    for problem in suite:
        print("Solving: ", problem)

        for _ in range(nreps):
            ga = GeneticAlgorithm(budget)
            ga(problem)
            problem.reset()
    logger.close()

    shutil.make_archive("ioh_data", "zip", "ioh_data")
    shutil.rmtree("ioh_data")


if __name__ == "__main__":
    # Simple test for development purpose
    test_algorithm(10)

    # Test required for A1, your GA should be able to pass this!
    # test_algorithm(100)

    # If your implementation passes test_algorithm(100)
    # collect_data(100)
