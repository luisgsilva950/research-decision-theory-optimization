import cProfile
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from problem_definition import ProblemDefinition


class Rvns:
    def __init__(self, problem: ProblemDefinition, kmax: int = 3, max_solutions_evaluations: int = 1000, n=5,
                 n_clusters: int = 100):
        self.problem = problem
        self.kmax = kmax
        self.max_solutions_evaluations = max_solutions_evaluations
        self.penal_fitness_historic = []
        self.penal_historic = []
        self.best_solution: Optional[ProblemDefinition] = None
        self.n = n
        self.n_clusters = n_clusters

    def run(self):
        for _ in range(self.n):
            self.penal_fitness_historic.append([])
            self.penal_historic.append([])
            self.problem = self.problem.get_initial_solution(n_clusters=self.n_clusters)
            num_evaluated_solutions = 0
            print(f'Initial Fitness: {self.problem.penal_fitness}\n')
            while num_evaluated_solutions <= self.max_solutions_evaluations:
                # with cProfile.Profile() as pr:
                self.problem.k = 1
                while self.problem.k <= self.kmax and num_evaluated_solutions <= self.max_solutions_evaluations:
                    print(f"Will shake: {self.problem.k}")
                    new_solution = self.problem.shake()
                    new_solution = new_solution.objective_function()
                    num_evaluated_solutions += 1
                    will_change = new_solution.penal_fitness < self.problem.penal_fitness
                    self.problem = self.problem.neighborhood_change(y=new_solution)
                    self.penal_fitness_historic[_].append(self.problem.penal_fitness)
                    self.penal_historic[_].append(self.problem.penal)
                    print(f'\033[3;{"92" if will_change else "91"}m'
                          f'Penal fitness {num_evaluated_solutions}, k: {self.problem.k - 1}: '
                          f'{self.problem.penal_fitness}, '
                          f'penal: {self.problem.penal}\n')
                # pr.print_stats(sort='cumtime')
            if not self.best_solution or self.problem.penal_fitness < self.best_solution.penal_fitness:
                self.best_solution = self.problem
            s = len(self.penal_fitness_historic[_])
            plt.plot(np.linspace(0, s - 1, s), self.penal_fitness_historic[_], '-', label=f'Execution {_}')
        plt.legend()
        plt.savefig(fname='f2_results/f2_solution_kmeans.png')
        plt.close()
        s = len(self.penal_fitness_historic[0])
        plt.plot(np.linspace(0, s - 1, s), self.penal_historic[0], '-', label=f'Penal')
        plt.savefig(fname='f2_results/f2_penal_historic_kmeans.png')
        plt.close()
        best_fitness_per_iteration = [min(values) for values in self.penal_fitness_historic]
        min_fitness = min(best_fitness_per_iteration)
        max_fitness = max(best_fitness_per_iteration)
        fitness_avg = sum(best_fitness_per_iteration) / len(best_fitness_per_iteration)
        dp_sum = sum([(fitness - fitness_avg) ** 2 for fitness in best_fitness_per_iteration])
        dp = (dp_sum / len(best_fitness_per_iteration)) ** (1 / 2)
        print(f"Best fitness per iteration: {best_fitness_per_iteration}")
        print(f"Best fitness: {min_fitness}")
        print(f"Worst fitness: {max_fitness}")
        print(f"Fitness std: {dp}")
        self.best_solution.plot_solution()
