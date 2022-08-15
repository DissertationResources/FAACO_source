#!/usr/bin/python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from City import *
from fireflies import *
matplotlib.rcParams['font.family'] = 'STSong'

def draw(points):
	points = list(points)
	points = points + points[:1]
	cities = map(lambda i: (i.x, i.y), points)
	(x, y) = zip(*cities)
	plt.scatter(x, y)
	plt.plot(x, y)
	plt.show()




if __name__ == '__main__':
	# number_of_points = int(input('Number of points: '))
	# next_random = lambda: random.random() * 100
	# locations = [ City(next_random(), next_random()) for i in range(number_of_points) ]
	# draw(locations)
	city_name = []
	city_con = []
	with open('E:\萤火虫蚁群\旅行商问题(TSP)数据集\旅行商问题(TSP)数据集\eil76.txt', 'r', encoding='UTF-8') as f:
		lines = f.readlines()
		# 调用readlines()一次读取所有内容并按行返回list给lines
		# for循环每次读取一行
		for line in lines:
			line = line.split('\n')[0]
			line = line.split(',')
			city_name.append(line[0])
			city_con.append([float(line[1]), float(line[2])])
		city_con = np.array(city_con)  # 获取30城市坐标


	number_of_points=int(len(lines))
	#
	locations=[ City(city_con[i,0],city_con[i,1]) for i in range(number_of_points)]
	draw(locations)
	solver = TSPSolver(locations)

	new_order = solver.run()
	# new = solver.ACOpathchoose()
	new_locations = [locations[i] for i in new_order]
	draw(new_locations)

	# solver.run(number_of_individuals=30)

	# 距离迭代图
	figFA = plt.figure()
	# plt.figure语()---在plt中绘制一张图片
	plt.title("Distance iteration graph")  # 距离迭代图
	plt.plot(range(0, len(solver.bestarray)), solver.bestarray)
	plt.xlabel("Number of iterations")  # 迭代次数
	plt.ylabel("Distance value")  # 距离
	plt.show()
