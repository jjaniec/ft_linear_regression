#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
import sys
import pandas as pd
import matplotlib.pyplot as plt

import typing
from typing import Tuple

import IPython

DATAFRAME_TARGET="price"
DATAFRAME_FEATURE_1="km"

DEFAULT_LEARNING_RATE=0.1
DEFAULT_MAX_ITER=100

# https://www.miximum.fr/blog/premiers-tests-avec-le-machine-learning/

class linear_regression_dataframe:
	df = None
	m = 0 # Number of rows
	n = 0 # Number of features
	y = 0 # Index of DATAFRAME_TARGET
	x1 = 0 # Index of DATAFRAME_FEATURE_1

	# Cost function
	last_cost_theta = [0, 0] # Last theta params used with the cost function
	last_cost_result = 0 # Last result of cost function

	# Gradient descent
	alpha = 0 # Learning rate
	max_iter = 0 # Max iteration of gradient descent

	def __init__(self: object, dataframe_csv_file: str):
		self.df = pd.read_csv(dataframe_csv_file)
		self.m, self.n = self.__get_dataframe_props()
		self.y = self.df.columns.get_loc(DATAFRAME_TARGET)
		self.x1 = self.df.columns.get_loc(DATAFRAME_FEATURE_1)

	def __get_dataframe_props(self: object) -> Tuple[str, str]:
		"""
		get m & n from dataset
		"""
		return [
			int(self.df.index.stop - self.df.index.start / self.df.index.step),
			len(self.df.columns) - 1
		]

	def __model(self: object, a: float, x: float, b: float) -> float:
		"""
		Model: f(x) = ax + b
		"""
		return ((a * x) + b)
	f = __model

	def	__cost_function(self: object, theta0: float, theta1: float) -> float:
		"""
		Cost function
		j(a, b) = (1 / 2m) * ( (f(x1) - y1)^2 + (f(x2) - y2)^2 + ... + (f(xm) - ym)^2) )
		j(a, b) = (1 / 2m) * (sum of: for i in dataset: (f(xi) - yi) )^2
		j(theta0, theta1) = (1 / 2m) * (sum of: for i in dataset: (htheta(xi) - yi) )^2
		"""
		last_cost_theta = [theta0, theta1]
		cost = 0
		for i in self.df.iterrows():
			cost += (self.f(theta1, i[1][DATAFRAME_FEATURE_1], theta0) - i[1][DATAFRAME_TARGET]) ** 2
		last_cost_result = (1 / (2 * self.m)) * cost
		return last_cost_result
	j = __cost_function

def main(dataframe_file: str) -> None:
	linreg_df = linear_regression_dataframe(dataframe_file)
	df = linreg_df.df
	print(f"Imported dataframe: m = {linreg_df.m}, n = {linreg_df.n}, target = \"{DATAFRAME_TARGET}\"")
	cost = linreg_df.j(0, 0)
	print(f"cost(0, 0) = {linreg_df.j(0, 0)}")
	print(f"cost(1, 0) = {linreg_df.j(1, 0)}")

	df.plot.scatter(x=linreg_df.y, y=linreg_df.x1)
	plt.show()

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Usage: ./train.py data.csv')
		exit(1)
	main(sys.argv[1])
