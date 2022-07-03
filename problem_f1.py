import collections
import random
from typing import List, Dict, Optional

import numpy

from graphic_plotter import GraphicPlotter
from models import Customer, AccessPoint, Coordinate
from problem_definition import ProblemDefinition
from utils import column, get_points_distances_from_file, get_arg_max


class ProblemDefinitionF1(ProblemDefinition):
    min_customers_attended = 570
    max_distance = 85
    max_active_points = 100
    max_consumed_capacity = 150

    def __init__(self, customers: List[Customer], points: List[AccessPoint], customer_point_distances=None,
                 solution=None, active_points=None, penal: float = 0.0, penal_fitness: float = 0.0,
                 fitness: float = 0.0, k: int = 1, total_distance: float = 0):
        self.customers = customers or []
        self.points = points or []
        self.k = k
        self.fitness = fitness
        self.penal_fitness = penal_fitness
        self.penal = penal
        self.customer_to_point_distances = customer_point_distances or []
        self.solution = solution or []
        self.active_points = active_points or set()
        self.total_distance = total_distance

    @staticmethod
    def from_csv() -> 'ProblemDefinitionF1':
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
        return ProblemDefinitionF1(customers=customers, points=points,
                                   customer_point_distances=get_points_distances_from_file())

    def objective_function(self) -> 'ProblemDefinitionF1':
        # with cProfile.Profile() as pr:
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
                        self.penal = self.penal + 100 * (distance - self.max_distance)
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
        self.total_distance = total_distance
        self.fitness = len(self.active_points)
        self.penal_fitness = self.fitness + self.penal
        print(f"\033[3;94mThe distance restriction was counted as: {penal_distance_count}")
        print(f"\033[3;94mThe consumed capacity restriction was counted as: {penal_consumed_capacity_count}")
        print(f'\033[3;{"93m" if self.penal else "32m"}Solution with penal fitness: {self.penal_fitness}, '
              f'penal: {self.penal} total customers attended: {customers_attended_count} '
              f'and total active points: {total_active_points}. K: {self.k}')
        # pr.print_stats(sort='cumtime')
        return self

    def neighborhood_change(self, y: 'ProblemDefinitionF1'):
        if y.penal_fitness < self.penal_fitness:
            y.k = 1
            y = ProblemDefinitionF1(customers=y.customers.copy(), points=y.points.copy(),
                                    customer_point_distances=y.customer_to_point_distances,
                                    solution=[list(p) for p in y.solution],
                                    active_points=y.active_points.copy(), fitness=y.fitness, penal=y.penal,
                                    penal_fitness=y.penal_fitness,
                                    k=y.k, total_distance=y.total_distance)
            print(f"\033[3;94mCustomers attended: {y.get_customers_attended_count()} - "
                  f"Total active points: {len(y.active_points)}")
            return y

        else:
            self.k = self.k + 1
            print(f"\033[3;94mCustomers attended: {self.get_customers_attended_count()} - "
                  f"Total active points: {len(self.active_points)}")
            return self

    def deactivate_random_access_point_and_enable_highest_access_closer_point(self):
        random_point: AccessPoint = numpy.random.choice(list(self.active_points))
        possible_indexes: List[int] = random_point.get_neighbor_indexes()
        consumed_capacity_per_point: Dict[int, float] = self.get_consumed_capacity()
        for customer in self.customers:
            if self.solution[customer.index][random_point.index]:
                each_possible_index_consumed_capacity = [consumed_capacity_per_point[p] for p in possible_indexes]
                point_index = possible_indexes[get_arg_max(each_possible_index_consumed_capacity)]
                self.enable_customer_point(customer=customer, point=self.points[point_index])
        self.deactivate_point(index=random_point.index)

    def deactivate_less_demanded_access_point_and_enable_closer_point(self):
        random_point: AccessPoint = numpy.random.choice(list(self.active_points))
        possible_indexes: List[int] = random_point.get_neighbor_indexes()
        index = random.choice(possible_indexes)
        for customer in self.customers:
            if self.solution[customer.index][random_point.index]:
                self.enable_customer_point(customer=customer, point=self.points[index])
        self.deactivate_point(index=random_point.index)

    def deactivate_random_access_points_and_enable_another_access_demand_point(self, size: int = 2):
        random_points: List[AccessPoint] = list(numpy.random.choice(list(self.active_points), size=size))
        for point in random_points:
            for customer in self.customers:
                if self.solution[customer.index][point.index]:
                    self.enable_customer_point(customer=customer, point=numpy.random.choice(list(self.active_points)))
            self.deactivate_point(index=point.index)

    def shake_k1(self):
        self.connect_random_customers_to_closer_active_access_point()

    def shake_k2(self):
        self.deactivate_less_demanded_access_point()

    def shake_k3(self):
        self.enable_random_customers(size=20)
        self.deactivate_less_demanded_point_and_enable_highest_access_closer_point()

    def shake(self):
        # with cProfile.Profile() as pr:
        y = ProblemDefinitionF1(customers=self.customers, points=self.points,
                                customer_point_distances=self.customer_to_point_distances,
                                solution=[list(p) for p in self.solution],
                                active_points=self.active_points.copy(), fitness=self.fitness,
                                penal=self.penal,
                                penal_fitness=self.penal_fitness,
                                k=self.k, total_distance=self.total_distance)
        if self.k == 1:
            y.shake_k1()
        elif self.k == 2:
            y.shake_k2()
        elif self.k == 3:
            y.shake_k3()
        y.update_active_points()
        # pr.print_stats(sort='cumtime')
        return y

    def get_less_demanded_customer_point(self, customer: Customer) -> AccessPoint:
        consumed_capacity_per_point = self.get_consumed_capacity()
        eligible_points = [p for p in self.active_points if
                           self.customer_to_point_distances[customer.index][p.index] < self.max_distance]
        if not self.active_points or not eligible_points:
            return customer.get_closer_point(points=self.points,
                                             distances=self.customer_to_point_distances[customer.index])
        point = eligible_points[0]
        for p in eligible_points:
            if consumed_capacity_per_point[p.index] < consumed_capacity_per_point[point.index]:
                point = p
        return point

    def get_highest_demanded_point(self, customer: Customer) -> AccessPoint:
        consumed_capacity_per_point = self.get_consumed_capacity()
        eligible_points = [p for p in self.active_points if
                           self.customer_to_point_distances[customer.index][p.index] < self.max_distance]
        if not self.active_points or not eligible_points:
            return customer.get_closer_point(points=self.points,
                                             distances=self.customer_to_point_distances[customer.index])
        point = eligible_points[0]
        for p in eligible_points:
            if consumed_capacity_per_point[p.index] >= consumed_capacity_per_point[point.index]:
                point = p
        return point

    def get_initial_solution(self) -> 'ProblemDefinitionF1':
        self.active_points = set()
        self.solution = []
        for customer in self.customers:
            customer_bool_solutions = []
            highest_demanded_point = self.get_highest_demanded_point(customer=customer)
            if highest_demanded_point.index not in [p.index for p in self.active_points]:
                self.active_points.add(highest_demanded_point)
            for point_index, point in enumerate(self.points):
                customer_bool_solutions.append(point_index == highest_demanded_point.index)
            self.solution.append(customer_bool_solutions)
        self.update_active_points()
        self.objective_function()
        print(f"\033[3;92mTotal active points on initial solution: {len(self.active_points)}, "
              f"initial penal: {self.penal}")
        return self
