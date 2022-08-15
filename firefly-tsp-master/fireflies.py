import random
import math
import itertools
import operator

import traceback

from heuristics import NearestInsertion, NearestNeighbour
import collections
import copy
import math
import matplotlib.pyplot as plt
import numpy as np


import NearestInsertionHeuristic


def cartesian_distance(a, b):
    "a and b should be tuples, computes distance between two cities"
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def compute_distances_matrix(locations):
    "returns array with distances for given points"
    return [[cartesian_distance(a, b) for a in locations] for b in locations]


def random_permutation(iterable, r=None):
    "returns new tuple with random permutation of iterable"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return list(random.sample(pool, r))


def single_path_cost(path, distances):
    "returns total distance of path"
    path = list(path)
    path = path + path[:1]
    return sum(distances[path[i]][path[i + 1]] for i in range(len(path) - 1))


def hamming_distance_with_info(a, b):
    "return number of places and places where two sequences differ"
    assert len(a) == len(b)
    ne = operator.ne
    differ = list(map(ne, a, b))
    return sum(differ), differ


def hamming_distance(a, b):
    dist, info = hamming_distance_with_info(a, b)
    return dist


class TSPSolver():
    def __init__(self, points):
        "points is list of objects of type City"
        self.weights = compute_distances_matrix(points)
        self.indexes = range(len(points))
        self.population = []
        self.light_intensities = []
        self.best_solution = None
        self.best_solution_cost = None
        self.n = None
        self.x = None
        self.g = 0
        self.k = 0
        self.choose = 1
        self.first_heuristic = NearestNeighbour(points)
        self.second_heuristic = NearestInsertion(points)
        self.city_condition = []
        self.city_name = []

    def f(self, individual):  # our objective function? lightness?
        "objective function - describes lightness of firefly"
        return single_path_cost(individual, self.weights)

    def determine_initial_light_intensities(self):
        "initializes light intensities"
        self.light_intensities = [self.f(x) for x in self.population]

    def generate_initial_population(self, number_of_individuals, heuristics_percents, ):
        "generates population of permutation of individuals"
        first_heuristic_part_limit = int(heuristics_percents[0] * number_of_individuals)
        second_heuristic_part_limit = int(heuristics_percents[1] * number_of_individuals)
        random_part_limit = number_of_individuals - first_heuristic_part_limit - second_heuristic_part_limit

        first_heuristic_part = self.first_heuristic.generate_population(first_heuristic_part_limit)
        second_heuristic_part = self.second_heuristic.generate_population(second_heuristic_part_limit)
        random_part = [random_permutation(self.indexes) for i in range(random_part_limit)]

        self.population = random_part + first_heuristic_part + second_heuristic_part
        self.absorptions = []
        for i in range(len(self.population)):
            self.absorptions.append(random.random() * 0.9 + 0.1)

    def check_if_best_solution(self, index):
        new_cost = self.light_intensities[index]
        if new_cost < self.best_solution_cost:
            self.best_solution = copy.deepcopy(self.population[index])
            self.best_solution_cost = new_cost

    def find_global_optimum(self):
        "finds the brightest firefly"
        index = self.light_intensities.index(min(self.light_intensities))
        self.check_if_best_solution(index)

    def move_firefly(self, a, b, r):
        "moving firefly a to b in less than r swaps"
        number_of_swaps = random.randint(0, r-2)
        distance, diff_info = hamming_distance_with_info(self.population[a], self.population[b])
        try:
            while number_of_swaps > 0:
                distance, diff_info = hamming_distance_with_info(self.population[a], self.population[b])
                ch01 = [i for i in range(len(diff_info)) if diff_info[i]]
                if ch01:
                    random_index = random.choice(ch01)
                    value_to_copy = self.population[b][random_index]
                    index_to_move = self.population[a].index(value_to_copy)

                    if number_of_swaps == 1 and self.population[a][index_to_move] == self.population[b][random_index] \
                            and self.population[a][random_index] == self.population[b][index_to_move]:
                        break

                    self.population[a][random_index], self.population[a][index_to_move] = self.population[a][
                                                                                              index_to_move], \
                                                                                          self.population[a][
                                                                                              random_index]
                    if self.population[a][index_to_move] == self.population[b][index_to_move]:
                        number_of_swaps -= 1
                number_of_swaps -= 1

            self.light_intensities[a] = self.f(self.population[a])



        except:
            traceback.print_exc()
            pass

        # number_of_swaps = random.randint(0, r - 2)
        # distance, diff_info = hamming_distance_with_info(self.population[a], self.population[b])
        #
        # while number_of_swaps > 0:
        #     distance, diff_info = hamming_distance_with_info(self.population[a], self.population[b])
        #     random_index = random.choice([i for i in range(len(diff_info)) if diff_info[i]])
        #     value_to_copy = self.population[b][random_index]
        #     index_to_move = self.population[a].index(value_to_copy)
        #
        #     if number_of_swaps == 1 and self.population[a][index_to_move] == self.population[b][random_index] \
        #             and self.population[a][random_index] == self.population[b][index_to_move]:
        #         break
        #
        #     self.population[a][random_index], self.population[a][index_to_move] = self.population[a][index_to_move], \
        #                                                                           self.population[a][random_index]
        #     if self.population[a][index_to_move] == self.population[b][index_to_move]:
        #         number_of_swaps -= 1
        #     number_of_swaps -= 1
        #
        # self.light_intensities[a] = self.f(self.population[a])

    def rotate_single_solution(self, i, value_of_reference):
        point_of_reference = self.population[i].index(value_of_reference)
        self.population[i] = collections.deque(self.population[i])
        l = len(self.population[i])
        number_of_rotations = (l - point_of_reference) % l
        self.population[i].rotate(number_of_rotations)
        self.population[i] = list(self.population[i])

    def rotate_solutions(self, value_of_reference):
        for i in range(1, len(self.population)):
            self.rotate_single_solution(i, value_of_reference)

    def I(self, index, r):
        return self.light_intensities[index] * math.exp(-1.0 * self.absorptions[index] * r ** 2)

    # def ACOpathchoose(self):
    #     AntCount, Distance, MAX_iter, Q, alpha, candidate, city_count, distance_best, etable, i, iter, j, path_best, pheromonetable, rho = self.init_aco()
    #
    #     while iter < MAX_iter:
    #         length = self.start_ant(AntCount, candidate, city_count)
    #
    #         for i in range(AntCount):
    #             # 移除已经访问的第一个元素
    #             unvisit = list(range(city_count))  # 列表形式存储没有访问的城市编号
    #             visit = candidate[i, 0]  # 当前所在点,第i个蚂蚁在第一个城市
    #             unvisit.remove(visit)  # 在未访问的城市中移除当前开始的点
    #             for j in range(1, city_count):  # 访问剩下的city_count个城市，city_count次访问
    #                 k = self.next_ant(alpha, etable, pheromonetable, unvisit, visit)
    #         # print(self.k)
    #         # self.choosepath(self.k, self.g)
    #         # a = self.choosepath(self.k, self.g)
    #         # print("返回值:",a)
    #
    #         # 下一个访问城市的索引值
    #         candidate[i, j] = self.choose
    #         print(self.choose)
    #         # unvisit.remove(self.choose)
    #
    #         length[i] += Distance[visit][self.choose]
    #         visit = self.g  # 更改出发点，继续选择下一个到达点
    #     length[i] += Distance[visit][candidate[i, 0]]  # 最后一个城市和第一个城市的距离值也要加进去
    #
    #     """
    #         更新路径等参数
    #     """
    #     # 如果迭代次数为一次，那么无条件让初始值代替path_best,distance_best.
    #     if iter == 0:
    #         distance_best[iter] = length.min()
    #         path_best[iter] = candidate[length.argmin()].copy()
    #     else:
    #         # 如果当前的解没有之前的解好，那么当前最优还是为之前的那个值；并且用前一个路径替换为当前的最优路径
    #         if length.min() > distance_best[iter - 1]:
    #             distance_best[iter] = distance_best[iter - 1]
    #             path_best[iter] = path_best[iter - 1].copy()
    #         else:  # 当前解比之前的要好，替换当前解和路径
    #             distance_best[iter] = length.min()
    #             path_best[iter] = candidate[length.argmin()].copy()
    #
    #     """
    #         信息素的更新
    #     """
    #     # 信息素的增加量矩阵
    #     changepheromonetable = np.zeros((city_count, city_count))
    #     for i in range(AntCount):
    #         for j in range(city_count - 1):
    #             # 当前路径比如城市23之间的信息素的增量：1/当前蚂蚁行走的总距离的信息素
    #             changepheromonetable[candidate[i, j]][candidate[i][j + 1]] += Q / length[i]
    #         # Distance[candidate[i, j]][candidate[i, j + 1]]
    #         # 最后一个城市和第一个城市的信息素增加量
    #         changepheromonetable[candidate[i, j + 1]][candidate[i, 0]] += Q / length[i]
    #     # 信息素更新的公式：
    #     pheromonetable = (1 - rho) * pheromonetable + changepheromonetable
    #     iter += 1
    #     print("蚁群算法的最优路径", path_best[-1] + 1)
    #     print("迭代", MAX_iter, "次后", "蚁群算法求得最优解", distance_best[-1])
    #
    #     # 路线图绘制
    #     fig = plt.figure()
    #     plt.title("Best roadmap")
    #     x = []
    #     y = []
    #     path = []
    #     for i in range(len(path_best[-1])):
    #         x.append(self.city_condition[int(path_best[-1][i])][0])
    #         y.append(self.city_condition[int(path_best[-1][i])][1])
    #         path.append(int(path_best[-1][i]) + 1)
    #     x.append(x[0])
    #     y.append(y[0])
    #     path.append(path[0])
    #     for i in range(len(x)):
    #         plt.annotate(path[i], xy=(x[i], y[i]), xytext=(x[i] + 0.3, y[i] + 0.3))
    #     plt.plot(x, y, '-o')
    #
    #     # 距离迭代图
    #     fig = plt.figure()
    #     # plt.figure语()---在plt中绘制一张图片
    #     plt.title("Distance iteration graph")  # 距离迭代图
    #     plt.plot(range(1, len(distance_best) + 1), distance_best)
    #     plt.xlabel("Number of iterations")  # 迭代次数
    #     plt.ylabel("Distance value")  # 距离值
    #     plt.show()

    def next_ant(self, alpha, etable, pheromonetable, unvisit, visit):
        protrans = np.zeros(len(unvisit))  # 每次循环都更改当前没有访问的城市的转移概率矩阵1*30,1*29,1*28...
        # 下一城市的概率函数
        for k0 in range(len(unvisit)):
            # 计算当前城市到剩余城市的（信息素浓度^alpha）*（城市适应度的倒数）^beta
            # etable[visit][unvisit[k]],(alpha+1)是倒数分之一，pheromonetable[visit][unvisit[k]]是从本城市到k城市的信息素
            protrans[k0] = np.power(pheromonetable[visit][unvisit[k0]], alpha) * np.power(
                etable[visit][unvisit[k0]], (alpha + 1))
        # 累计概率，轮盘赌选择
        cumsumprobtrans = (protrans / sum(protrans)).cumsum()
        cumsumprobtrans -= np.random.rand()
        # 求出离随机数产生最近的索引值
        # k = unvisit[list(cumsumprobtrans > 0).index(True)]
        if unvisit[list(cumsumprobtrans > 0).index(True)] is None:
            k = 0
        else:
            k = unvisit[list(cumsumprobtrans > 0).index(True)]



        return k


    def start_ant(self, AntCount, candidate, city_count):
        # first：蚂蚁初始点选择
        if AntCount <= city_count:
            # np.random.permutation随机排列一个数组的
            # candidate[:, 0] = np.random.permutation(range(city_count))[:AntCount]
            for i in range(city_count):
                candidate[i][0] = i
        else:
            m = AntCount - city_count
            n = 2
            candidate[:city_count, 0] = np.random.permutation(range(city_count))[:]
            while m > city_count:
                candidate[city_count * (n - 1):city_count * n, 0] = np.random.permutation(range(city_count))[:]
                m = m - city_count
                n = n + 1
            candidate[city_count * (n - 1):AntCount, 0] = np.random.permutation(range(city_count))[:m]
        length = np.zeros(AntCount)  # 每次迭代的N个蚂蚁的距离值
        return length

    def init_aco(self):
        # 蚂蚁数量
        AntCount = 60
        # 信息素
        alpha = 1  # 信息素重要程度因子
        beta = 2  # 启发函数重要程度因子
        rho = 0.1  # 挥发速度
        iter = 0  # 迭代初始值
        MAX_iter = 1000  # 最大迭代值
        # with open('../2城市.txt', 'r', encoding='UTF-8') as f:
        #     lines = f.readlines()
        # for line in lines:
        #     line = line.split('\n')[0]
        #     line = line.split(',')
        #     self.city_name.append(line[0])
        #     self.city_condition.append([float(line[1]), float(line[2])])
        self.city_condition = self.population
        # Distance距离矩阵
        city_count = 60
        Distance = np.zeros((city_count, city_count))

        for i in range(city_count):
            for j in range(city_count):
                if i != j:
                    # Distance[i][j] = math.sqrt((self.city_condition[i][0] - self.city_condition[j][0]) ** 2 + (
                    #         self.city_condition[i][1] - self.city_condition[j][1]) ** 2)
                    a = map(lambda x: (x[0] - x[1]) ** 2, zip(self.population[i], self.population[j]))
                    Distance[i][j] = round(math.sqrt(sum(a)),2)
                else:
                    Distance[i][j] = 100000
        Q = 20
        # 初始信息素矩阵，全是为1组成的矩阵
        pheromonetable = np.ones((city_count, city_count))
        # 候选集列表,存放100只蚂蚁的路径(一只蚂蚁一个路径),一共就Antcount个路径，一共是蚂蚁数量*31个城市数量
        candidate = np.zeros((AntCount, city_count)).astype(int)
        # path_best存放的是相应的，每次迭代后的最优路径，每次迭代只有一个值
        path_best = np.zeros((MAX_iter, city_count))
        # 存放每次迭代的最优距离
        distance_best = np.zeros(MAX_iter)
        # 倒数矩阵
        etable = 1.0 / Distance
        return AntCount, Distance, MAX_iter, Q, alpha, candidate, city_count, distance_best, etable,  iter,  path_best, pheromonetable, rho

    # k为萤火虫路径备选，g为蚁群路径备选，s为备选项即所有点，进行加权选择
    def choosepath(self, k, g):
        x = (0.3 * k) + (0.7 * g)
        choose = 0
        min = math.pow(x, 2)
        s = np.zeros(60)
        for j in range(60):
            s[j] = math.pow(j - x, 2)
            if x != j:
                if s[j] < min:
                    min = s[j]
                    choose = j

        return choose

    def run(self, number_of_individuals=60, iterations=1000, heuristics_percents=(0.0, 0.0, 1.0), beta=0.5):
        "gamma is parameter for light intensities, beta is size of neighbourhood according to hamming distance"
        # hotfix, will rewrite later

        self.bestarray = np.zeros(iterations)
        self.best_solution = random_permutation(self.indexes)
        self.best_solution_cost = single_path_cost(self.best_solution, self.weights)

        self.generate_initial_population(number_of_individuals, heuristics_percents)
        value_of_reference = self.population[0][0]
        self.rotate_solutions(value_of_reference)
        self.determine_initial_light_intensities()
        self.find_global_optimum()
        print(self.best_solution_cost)
        AntCount, Distance, MAX_iter, Q, alpha, candidate, city_count, distance_best, etable,  iter,  path_best, pheromonetable, rho = self.init_aco()

        individuals_indexes = range(number_of_individuals)
        self.n = 0
        neighbourhood = beta * len(individuals_indexes)
        while self.n < iterations:

            length = self.start_ant(AntCount, candidate, city_count)
            for i in individuals_indexes:
                unvisit = list(range(city_count))  # 列表形式存储没有访问的城市编号
                visit = candidate[i, 0]  # 当前所在点,第i个蚂蚁在第一个城市
                unvisit.remove(visit)  # 在未访问的城市中移除当前开始的点
                for j in range(1,number_of_individuals):
                    # unvisit = list(range(city_count))  # 列表形式存储没有访问的城市编号
                    # visit = candidate[i, 0]  # 当前所在点,第i个蚂蚁在第一个城市
                    # unvisit.remove(visit)  # 在未访问的城市中移除当前开始的点


                    r = hamming_distance(self.population[i], self.population[j])
                    # for q in range(1, city_count):
                    k = self.next_ant(alpha, etable, pheromonetable, unvisit, visit)

                    u = self.choosepath(k, i)

                    candidate[i, j] = k
                    unvisit.remove(k)
                    length[i] += Distance[visit][k]
                    visit = k  # 更改出发点，继续选择下一个到达点

                    #     unvisit.remove(k)
                    #     length[i] += Distance[visit][k]
                    #     visit = k  # 更改出发点，继续选择下一个到达点
                    # length[i] += Distance[visit][candidate[i, 0]]  # 最后一个城市和第一个城市的距离值也要加进去


                    if self.I(i, r) > self.I(j, r) and r < neighbourhood :
                        # g = j
                        # u = self.choosepath(k, g)

                        self.move_firefly(u, i, r)
                        self.check_if_best_solution(j)
                        # print(self.n, self.best_solution_cost)


                length[i] += Distance[visit][candidate[i, 0]]

            self.bestarray[self.n - 1] = self.best_solution_cost
            """
                        更新路径等参数
                    """
            # 如果迭代次数为一次，那么无条件让初始值代替path_best,distance_best.
            if self.n == 0:
                distance_best[self.n] = length.min()
                path_best[self.n] = candidate[length.argmin()].copy()
            else:
                # 如果当前的解没有之前的解好，那么当前最优还是为之前的那个值；并且用前一个路径替换为当前的最优路径
                if length.min() > distance_best[self.n - 1]:
                    distance_best[self.n] = distance_best[self.n - 1]
                    path_best[self.n] = path_best[self.n - 1].copy()
                else:  # 当前解比之前的要好，替换当前解和路径
                    distance_best[self.n] = length.min()
                    path_best[self.n] = candidate[length.argmin()].copy()

            """
                信息素的更新
            """
            # 信息素的增加量矩阵
            changepheromonetable = np.zeros((city_count, city_count))
            for i in range(AntCount):
                for j in range(city_count - 1):
                    # 当前路径比如城市23之间的信息素的增量：1/当前蚂蚁行走的总距离的信息素
                    changepheromonetable[candidate[i, j]][candidate[i][j + 1]] += Q / length[i]
                # Distance[candidate[i, j]][candidate[i, j + 1]]
                # 最后一个城市和第一个城市的信息素增加量
                changepheromonetable[candidate[i, j + 1]][candidate[i, 0]] += Q / length[i]
            # 信息素更新的公式：
            pheromonetable = (1 - rho) * pheromonetable + changepheromonetable
            self.n += 1
            if self.n % 100 == 0:
                print(self.n)
                print(self.best_solution_cost)


        # print("蚁群算法的最优路径", path_best[-1] + 1)
        # print("迭代", MAX_iter, "次后", "蚁群算法求得最优解", distance_best[-1])

        # # 路线图绘制
        # fig = plt.figure()
        # plt.title("Best roadmap")
        # x = []
        # y = []
        # path = []
        # for i in range(len(path_best[-1])):
        #     x.append(self.city_condition[int(path_best[-1][i])][0])
        #     y.append(self.city_condition[int(path_best[-1][i])][1])
        #     path.append(int(path_best[-1][i]) + 1)
        # x.append(x[0])
        # y.append(y[0])
        # path.append(path[0])
        # for i in range(len(x)):
        #     plt.annotate(path[i], xy=(x[i], y[i]), xytext=(x[i] + 0.3, y[i] + 0.3))
        # plt.plot(x, y, '-o')
        #
        # # 距离迭代图
        # fig = plt.figure()
        # # plt.figure语()---在plt中绘制一张图片
        # plt.title("Distance iteration graph")  # 距离迭代图
        # plt.plot(range(1, len(distance_best) + 1), distance_best)
        # plt.xlabel("Number of iterations")  # 迭代次数
        # plt.ylabel("Distance value")  # 距离值
        # plt.show()
        self.bestarray[self.n-1] = self.best_solution_cost

        print(self.best_solution_cost)
        return self.best_solution
