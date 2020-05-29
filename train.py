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

def key_press_handler(event):
	if event.key == 'escape':
		plt.close()
		sys.exit(0)

def draw_predict_line(linreg_df: object) -> None:
	x = [
		max(linreg_df.df['km']),
		min(linreg_df.df['km'])
	]
	y = [
		linreg_df.predict(max(linreg_df.df['km'])),
		linreg_df.predict(min(linreg_df.df['km']))
	]
	print(x, y)
	plt.plot(x, y, 'r-', lw=2)

def show_cost_history_graph(linreg_df: object) -> None:
	figure = plt.gcf()
	figure.canvas.set_window_title(CANVAS_WINDOW_TITLES)
	figure.canvas.mpl_connect('key_press_event', key_press_handler)
	plt.plot(range(linreg_df.max_iter), linreg_df.cost_history)
	plt.title("Cost history")
	plt.show()

def	show_linreg_graph(linreg_df: object) -> None:
	figure = plt.gcf()
	figure.canvas.set_window_title(CANVAS_WINDOW_TITLES)
	figure.canvas.mpl_connect('key_press_event', key_press_handler)
	plt.scatter(linreg_df.df.km, linreg_df.df.price)
	draw_predict_line(linreg_df)
	plt.title("Linear regression")
	plt.xlabel('km')
	plt.ylabel('price')
	plt.show()

def main(dataframe_file: str) -> Union[None, Tuple[float, float]]:
	df = pd.read_csv(dataframe_file)
	linreg_df = linearRegressionDataframe(df)
	linreg_df.max_iter = DEFAULT_MAX_ITER
	linreg_df.normalize(linreg_df.df.price.max(), linreg_df.df.km.max())
	print(f"Imported dataframe: m = {linreg_df.m}, n = {linreg_df.n}, target = \"{DATAFRAME_TARGET}\"")
	theta = [0, 0]
	for i in range(0, DEFAULT_MAX_ITER):
		derivative_theta = linreg_df.derivate_theta(theta)
		theta = linreg_df.gradient_descent(theta, derivative_theta)
		linreg_df.cost_history.append(linreg_df.j(theta))
		print(f"j({theta[0]}, {theta[1]}) = {linreg_df.j(theta)}")
	linreg_df.theta = theta
	linreg_df.restore_normalize_df()
	linreg_df.restore_normalize_theta()
	show_cost_history_graph(linreg_df)
	show_linreg_graph(linreg_df)
	return (linreg_df.theta)

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
