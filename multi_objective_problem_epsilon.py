import copy
from typing import List

import numpy

from models import Customer, AccessPoint, Coordinate
from problem_definition import ProblemDefinition
from utils import get_points_distances_from_file, get_arg_min


class EpsilonRestrictProblem(ProblemDefinition):
    min_customers_attended = 570
    max_distance = 85
    max_consumed_capacity = 150

    def __init__(self, customers: List[Customer], points: List[AccessPoint], customer_point_distances=None,
                 solution=None, active_points=None, penal: float = 0.0, penal_fitness: float = 0.0,
                 fitness: float = 0.0, k: int = 1, max_active_points: int = 100, total_distance: float = 0):
        self.customers = customers or []
        self.points = points or []
        self.k = k
        self.fitness = fitness
        self.penal_fitness = penal_fitness
        self.penal = penal
        self.customer_to_point_distances = customer_point_distances or []
        self.solution = solution or []
        self.active_points = active_points or []
        self.max_active_points = max_active_points
        self.total_distance = total_distance

    @staticmethod
    def from_csv(max_active_points: int) -> 'EpsilonRestrictProblem':
        customers = []
        with open('clientes.csv') as file:
            content = file.readlines()
        for index, row in enumerate(content):
            row = row.split(",")
            customers.append(
                Customer(coordinates=Coordinate(x=float(row[0]), y=float(row[1])), consume=float(row[2]), index=index))
        points = []
        for x in range(0, 1010, 10):
            for y in range(0, 1010, 10):
                points.append(AccessPoint(x=x, y=y, index=len(points)))
        return EpsilonRestrictProblem(customers=customers, points=points,
                                      customer_point_distances=get_points_distances_from_file(),
                                      max_active_points=max_active_points)

    def objective_function(self) -> 'EpsilonRestrictProblem':
        total_distance = 0
        total_active_points = len(self.active_points)
        customers_attended_count = self.get_customers_attended_count()
        consumed_capacity_per_point = self.get_consumed_capacity()
        self.penal = 0.0
        penal_distance_count = 0
        penal_consumed_capacity_count = 0
        for active_point in self.active_points:
            for customer in self.customers:
                point_index = active_point.index
                customer_index = customer.index
                active = self.solution[customer.index][active_point.index]
                if active:
                    consumed_capacity = consumed_capacity_per_point[point_index]
                    distance = self.customer_to_point_distances[customer_index][point_index]
                    total_distance = total_distance + distance
                    if distance > self.max_distance:
                        self.penal = self.penal + (distance - self.max_distance)
                        penal_distance_count += 1
                    if consumed_capacity > self.max_consumed_capacity:
                        self.penal = self.penal + 2 * (consumed_capacity - self.max_consumed_capacity)
                        print(f"The consumed capacity restriction was outdated by customer: {customer_index}. "
                              f"Consumed capacity: {consumed_capacity}")
                        penal_consumed_capacity_count += 1
        if total_active_points > self.max_active_points:
            self.penal = self.penal + 400 * (total_active_points - self.max_active_points)
        if customers_attended_count < self.min_customers_attended:
            self.penal = self.penal + 600 * (self.min_customers_attended - customers_attended_count)
        self.fitness = total_distance
        self.penal_fitness = self.fitness + self.penal
        self.total_distance = total_distance
        print(f"\033[3;94mThe distance restriction was counted as: {penal_distance_count}")
        print(f"\033[3;94mThe consumed capacity restriction was counted as: {penal_consumed_capacity_count}")
        print(f'\033[3;{"93m" if self.penal else "32m"}Solution with penal fitness: {self.penal_fitness}, '
              f'penal: {self.penal} total customers attended: {customers_attended_count} '
              f'total active points: {total_active_points} and max active points: {self.max_active_points}')
        return self

    def neighborhood_change(self, y: 'EpsilonRestrictProblem'):
        if y.penal_fitness < self.penal_fitness:
            y.k = 1
            y = EpsilonRestrictProblem(customers=y.customers, points=y.points,
                                       customer_point_distances=y.customer_to_point_distances,
                                       solution=[list(p) for p in y.solution],
                                       active_points=list(y.active_points), fitness=y.fitness, penal=y.penal,
                                       penal_fitness=y.penal_fitness,
                                       k=y.k, total_distance=y.total_distance, max_active_points=y.max_active_points)
            print(f"\033[3;94mCustomers attended: {y.get_customers_attended_count()} - "
                  f"Total active points: {len(y.active_points)}")
            return y

        else:
            self.k = self.k + 1
            print(f"\033[3;94mCustomers attended: {self.get_customers_attended_count()} - "
                  f"Total active points: {len(self.active_points)}")
            return self

    def shake_k1(self):
        self.connect_random_customers_to_closer_active_access_point()

    def shake_k2(self):
        self.enable_random_customers(size=numpy.random.randint(1, 5))

    def shake_k3(self):
        self.deactivate_random_demand_point_and_connect_closer_point()

    def shake_k4(self):
        self.deactivate_less_demanded_access_point()

    def shake_k5(self):
        self.enable_random_customers(size=numpy.random.randint(1, 20))
        self.deactivate_less_demanded_point_and_enable_highest_access_closer_point()

    def shake(self):
        y = EpsilonRestrictProblem(customers=self.customers, points=self.points,
                                   customer_point_distances=self.customer_to_point_distances,
                                   solution=[list(p) for p in self.solution],
                                   active_points=list(self.active_points), fitness=self.fitness,
                                   penal=self.penal,
                                   penal_fitness=self.penal_fitness,
                                   k=self.k, total_distance=self.total_distance,
                                   max_active_points=self.max_active_points)
        if self.k == 1:
            y.shake_k1()
        elif self.k == 2:
            y.shake_k2()
        elif self.k == 3:
            y.shake_k3()
        elif self.k == 4:
            y.shake_k4()
        elif self.k == 5:
            y.shake_k5()
        y.update_active_points()
        return y

    def get_initial_solution(self) -> 'EpsilonRestrictProblem':
        all_points = self.get_points_with_space_100()
        self.active_points = set()
        self.solution = []
        for customer in self.customers:
            customer_bool_solutions = []
            distances = [self.customer_to_point_distances[customer.index][p.index] for p in all_points]
            index = get_arg_min(distances)
            closer_point = all_points[index]
            if distances[index] > self.max_distance:
                closer_point = customer.get_closer_point(points=self.points,
                                                         distances=self.customer_to_point_distances[customer.index])
            if distances[index] < self.max_distance and len(self.active_points) < self.max_active_points:
                self.active_points.add(closer_point)
            for point_index, point in enumerate(self.points):
                customer_bool_solutions.append(point_index == closer_point.index
                                               and distances[index] < self.max_distance
                                               and len(self.active_points) < self.max_active_points)
            self.solution.append(customer_bool_solutions)
        self.update_active_points()
        self.objective_function()
        print(f"\033[3;92mTotal active points on initial solution: {len(self.active_points)}, "
              f"initial penal: {self.penal}")
        return self
