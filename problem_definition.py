import collections
from typing import Set, List, Optional

import numpy
from numpy import random

from graphic_plotter import GraphicPlotter
from models import AccessPoint, Customer
from utils import get_arg_min, get_arg_max


class ProblemDefinition:
    k: int
    total_distance: float
    max_distance: float
    max_consumed_capacity: float
    max_active_points: int
    min_customers_attended: int
    active_points: Set[AccessPoint]
    solution: List[List[bool]]
    customers: List[Customer]
    points: List[AccessPoint]
    customer_to_point_distances: List[List[float]]
    penal_fitness: float
    fitness: float
    penal: float

    def objective_function(self) -> 'ProblemDefinition':
        ...

    def neighborhood_change(self, y: 'ProblemDefinition') -> 'ProblemDefinition':
        ...

    def shake(self) -> 'ProblemDefinition':
        ...

    def get_initial_solution(self) -> 'ProblemDefinition':
        ...

    def update_active_points(self):
        self.active_points = set()
        for customer in self.customers:
            if any(self.solution[customer.index]):
                index = get_arg_max(self.solution[customer.index])
                self.active_points.add(self.points[index])

    # FO methods
    def penalize_distance(self, distance: float):
        if distance > self.max_distance:
            self.penal = self.penal + (distance - self.max_distance) * 2

    def penalize_consumed_capacity(self, consumed_capacity: float):
        if consumed_capacity > self.max_consumed_capacity:
            self.penal = self.penal + 2 * (consumed_capacity - self.max_consumed_capacity)
            print(f"The consumed capacity restriction was affected. Consumed capacity: {consumed_capacity}")

    def penalize_total_active_points(self):
        total_active_points = len(self.active_points)
        if total_active_points > self.max_active_points:
            self.penal = self.penal + 600 * (total_active_points - self.max_active_points)
            print(f"The total active points restriction was affected. Total active points: {total_active_points}")

    def penalize_total_customers(self, customers_attended_count: int):
        if customers_attended_count < self.min_customers_attended:
            self.penal = self.penal + 600 * (self.min_customers_attended - customers_attended_count)
            print(f"The total customers restriction was affected. Total customers: {customers_attended_count}")

    # Shake methods
    def deactivate_point(self, index: int):
        for customer in self.customers:
            self.solution[customer.index][index] = False

    def enable_customer_point(self, customer: Customer, point: AccessPoint):
        self.solution[customer.index][point.index] = True

    def disable_customer_point(self, customer: Customer, point: AccessPoint):
        self.solution[customer.index][point.index] = False

    def enable_random_customers(self, size: int = 10, points: List[AccessPoint] = None):
        if not points:
            points = self.active_points
        random_customers: List[Customer] = list(
            numpy.random.choice([c for c in self.customers if max(self.solution[c.index]) == 0], size=size))
        for customer in random_customers:
            closer_point = customer.get_closer_point(points=points)
            if self.customer_to_point_distances[customer.index][closer_point.index] < self.max_distance:
                self.enable_customer_point(customer=customer, point=closer_point)

    def deactivate_random_customers(self, size: int = 10):
        random_customers: List[Customer] = list(
            numpy.random.choice([c for c in self.customers if max(self.solution[c.index]) > 0], size=size))
        for customer in random_customers:
            index_max = get_arg_max(self.solution[customer.index])
            self.disable_customer_point(customer=customer, point=self.points[index_max])

    def deactivate_random_access_points(self, size: int = 2):
        random_points = list(numpy.random.choice(list(self.active_points), size=size))
        for point in random_points:
            self.deactivate_point(index=point.index)

    def deactivate_random_demand_point_and_connect_closer_point(self):
        random_point: AccessPoint = numpy.random.choice(list(self.active_points))
        active_indexes: List[int] = [p.index for p in self.active_points]
        possible_indexes: List[int] = random_point.get_neighbor_indexes()
        possible_indexes: List[int] = [i for i in possible_indexes if i not in active_indexes]
        for customer in self.customers:
            if self.solution[customer.index][random_point.index]:
                possible_distances = [self.customer_to_point_distances[customer.index][i] for i in possible_indexes]
                if possible_distances:
                    closer_index = possible_indexes[get_arg_min(possible_distances)]
                    self.enable_customer_point(customer=customer, point=self.points[closer_index])
        self.deactivate_point(index=random_point.index)

    def connect_random_customers_to_closer_active_access_point(self, size: int = 5):
        random_customers: List[Customer] = list(numpy.random.choice(self.customers, size=size))
        for customer in random_customers:
            index_max = get_arg_max(self.solution[customer.index])
            closer_point = customer.get_closer_point(points=self.active_points,
                                                     distances=self.customer_to_point_distances[customer.index])
            if self.solution[customer.index][index_max] and closer_point.index != index_max:
                self.enable_customer_point(customer=customer, point=closer_point)
                self.disable_customer_point(customer=customer, point=self.points[index_max])

    def deactivate_less_demanded_point_and_enable_highest_access_closer_point(self):
        less_demanded_point: AccessPoint = self.get_less_demanded_point()
        if less_demanded_point:
            for customer in self.customers:
                if self.solution[customer.index][less_demanded_point.index]:
                    candidates = [p for p in self.active_points if
                                  p.index != less_demanded_point.index and
                                  self.customer_to_point_distances[customer.index][p.index] < self.max_distance]
                    if candidates:
                        closer_point = random.choice(candidates)
                        self.enable_customer_point(customer=customer, point=self.points[closer_point.index])
            self.deactivate_point(index=less_demanded_point.index)

    def deactivate_less_demanded_access_point(self):
        less_demanded_point: AccessPoint = self.get_less_demanded_point()
        if less_demanded_point:
            self.deactivate_point(index=less_demanded_point.index)

    # Utils
    def get_customers_attended_count(self) -> int:
        customers_attended_count = 0
        for customer_points in self.solution:
            customers_attended_count = customers_attended_count + max(customer_points)
        return customers_attended_count

    def get_consumed_capacity(self) -> dict:
        consumed_capacity_per_point = collections.defaultdict(float)
        for customer in self.customers:
            for active_point in self.active_points:
                if len(self.solution) > customer.index and self.solution[customer.index][active_point.index]:
                    consumed_capacity_per_point[active_point.index] += self.customers[customer.index].consume
        return consumed_capacity_per_point

    def get_less_demanded_point(self) -> Optional[AccessPoint]:
        consumed_capacity_per_point = self.get_consumed_capacity()
        if not self.active_points:
            return None
        point = next(iter(self.active_points))
        for p in self.active_points:
            if consumed_capacity_per_point[p.index] < consumed_capacity_per_point[point.index]:
                point = p
        return point

    def get_points_with_space_100(self) -> List[AccessPoint]:
        points = []
        for p in self.points:
            if p.x % 100 == 0 and p.y % 100 == 0 and p.x >= 100 and p.y >= 100:
                points.append(p)
        return points

    # Plot methods
    def plot_solution(self):
        plotter = GraphicPlotter(title='Connexions', connexions=self.get_connexions())
        plotter.plot()

    def get_connexions(self):
        result = list()
        for point in self.active_points:
            point_customers = []
            for customer in self.customers:
                if self.solution[customer.index][point.index]:
                    point_customers.append(customer.coordinates)
            result.append((point, point_customers))
        return result
