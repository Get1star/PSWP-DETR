
import warnings
from collections import defaultdict

import paddle
from paddle import _C_ops

from ..fluid import core, framework
from ..fluid.dygraph import base as imperative_base
from ..fluid.framework import Variable, in_dygraph_mode
from .optimizer import Optimizer
import numpy as np


class WPO(Optimizer):
    def __init__(self, dim, upper_bound, lower_bound, objective_function, noP=10, maxIter=15, w=0.5, c1=1.5, c2=1.5):
        self.dim = dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.noP = noP
        self.fobj = objective_function
        self.maxIter = maxIter
        self.w = w  # inertial weighting
        self.c1 = c1  # Individual cognitive factors
        self.c2 = c2  # Social Cognitive Factor
        self.positions = np.array(
            [np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=self.dim) for _ in range(self.noP)])
        self.velocities = np.random.uniform(low=-0.1, high=0.1, size=(self.noP, self.dim))
        self.best_positions = np.copy(self.positions)
        self.best_fitness = [float('inf')] * self.noP
        self.global_best_position = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=self.dim)
        self.global_best_fitness = float('inf')
        self.a = 2  # convergence factor

    def evaluate(self):
        for i in range(self.noP):
            fitness = self.fobj(self.positions[i])
            if fitness < self.best_fitness[i]:
                self.best_fitness[i] = fitness
                self.best_positions[i] = self.positions[i]
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.positions[i]

    def optimize(self):
        for _ in range(self.maxIter):
            self.evaluate()
            a = 2 - 2 * (_ / self.maxIter)  # Linear decrease a
            for i in range(self.noP):
                r = np.random.rand()  # random num
                A = 2 * a * r - a
                C = 2 * r
                b = 1
                l = (np.random.rand() - 0.5) * 2

                p = np.random.rand()

                if p < 0.5:
                    if abs(A) > 1:
                        # Randomly select a search agent
                        rand_index = np.random.randint(0, self.noP)
                        D = abs(C * self.best_positions[rand_index] - self.positions[i])
                        self.positions[i] = self.best_positions[rand_index] - A * D
                    else:
                        D = abs(C * self.global_best_position - self.positions[i])
                        self.positions[i] = self.global_best_position - A * D
                else:
                    D = abs(self.global_best_position - self.positions[i])
                    self.positions[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + self.global_best_position

                # Update velocities and positions
                self.velocities[i] = self.w * self.velocities[i] + self.c1 * np.random.rand() * (self.best_positions[i] - self.positions[i]) + self.c2 * np.random.rand() * (self.global_best_position - self.positions[i])
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_fitness
