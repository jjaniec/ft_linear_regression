#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

import typing
from typing import Tuple, Union

import IPython

from ft_linear_regression.config import *
from ft_linear_regression.classes.linearRegressionDataframe import linearRegressionDataframe

# https://www.miximum.fr/blog/premiers-tests-avec-le-machine-learning/
# https://mrmint.fr/gradient-descent-algorithm

def main(dataframe_file: str) -> Union[None, Tuple[float, float]]:
	linreg_df = linearRegressionDataframe(dataframe_file)
	linreg_df.normalize(linreg_df.df.price.max(), linreg_df.df.km.max())
	df = linreg_df.df
	print(f"Imported dataframe: m = {linreg_df.m}, n = {linreg_df.n}, target = \"{DATAFRAME_TARGET}\"")
	theta = [0, 0]
	for i in range(0, DEFAULT_MAX_ITER):
		derivative_theta = linreg_df.derivate_theta(theta[0], theta[1])
		theta = linreg_df.gradient_descent(theta, derivative_theta)
		linreg_df.cost_history.append(linreg_df.j(theta[0], theta[1]))
		print(f"j({theta[0]}, {theta[1]}) = {linreg_df.j(theta[0], theta[1])}")
	print(len(linreg_df.cost_history), len(range(DEFAULT_MAX_ITER)))
	print(linreg_df.cost_history)
	# df.plot.scatter(x=linreg_df.y, y=linreg_df.x1)
	plt.plot(range(DEFAULT_MAX_ITER), linreg_df.cost_history)
	plt.title("Cost function")
	plt.show()
	return (theta)

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Usage: ./train.py data.csv')
		exit(1)
	ret = main(sys.argv[1])
	if ret is None:
		exit(1)
	# if os.path.exists(THETA_SAVE_FILENAME):
	# 	os.remove(THETA_SAVE_FILENAME)
	with open(THETA_SAVE_FILENAME, "w") as f:
		f.write(f"theta0,theta1\n{ret[0]},{ret[1]}")
		f.close()
