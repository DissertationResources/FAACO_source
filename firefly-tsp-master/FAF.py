def run(self, number_of_individuals=10, iterations=10, heuristics_percents=(0.0, 0.0, 1.0), beta=0.5,
        alpha=1,beta2=2,rho=0.1,number_of_points=10):
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

    Distance = np.zeros((number_of_points, number_of_points))
    self.best_solution2 = random_permutation(self.indexes)
    self.best_solution_cost2 = single_path_cost(self.best_solution2, self.weights)

    # 初始信息素矩阵，全是为1组成的矩阵
    pheromonetable = np.ones((number_of_points, number_of_points))
    # 候选集列表,存放100只蚂蚁的路径(一只蚂蚁一个路径),一共就Antcount个路径，一共是蚂蚁数量*31个城市数量
    candidate = np.zeros((number_of_individuals, number_of_points)).astype(int)

    # path_best存放的是相应的，每次迭代后的最优路径，每次迭代只有一个值
    self.path_best = np.zeros((iterations, number_of_points))

    # 存放每次迭代的最优距离
    self.distance_best = np.zeros(iterations)
    # 倒数矩阵
    etable = 1.0 / self.best_solution_cost2

    individuals_indexes = range(number_of_individuals)
    self.n = 0
    neighbourhood = beta * len(individuals_indexes)
    while self.n < iterations:
        # first：蚂蚁初始点选择
        if number_of_individuals <= number_of_points:
            # np.random.permutation随机排列一个数组的
            candidate[:, 0] = np.random.permutation(range(number_of_points))[:number_of_individuals]
        else:
            m = number_of_individuals - number_of_points
            n = 2
            candidate[:number_of_points, 0] = np.random.permutation(range(number_of_points))[:]
            while m > number_of_points:
                candidate[number_of_points * (n - 1):number_of_points * n, 0] = np.random.permutation(range(number_of_points))[:]
                m = m - number_of_points
                n = n + 1
            candidate[number_of_points * (n - 1):number_of_individuals, 0] = np.random.permutation(range(number_of_points))[:m]
        length = np.zeros(number_of_individuals)  # 每次迭代的N个蚂蚁的距离值
        for j in individuals_indexes:
            for i in individuals_indexes:
                r = hamming_distance(self.population[i], self.population[j])
                if self.I(i, r) > self.I(j, r) and r < neighbourhood:
                    # self.g=j
                    # m = self.choosepath(self.k, self.g)
                    self.move_firefly(i, j, r)
                    self.check_if_best_solution(i)
        self.bestarray[self.n - 1] = self.best_solution_cost

        self.n += 1
        if self.n % 100 == 0:
            print(self.n)
            print(self.best_solution_cost)

    print(self.best_solution_cost)
    return self.best_solution