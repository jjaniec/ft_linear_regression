#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

from optparse import OptionParser

import typing
from typing import Tuple, Union

import IPython

from ft_linear_regression.config import *
from ft_linear_regression.classes.linearRegressionDataframe import linearRegressionDataframe
from ft_linear_regression.classes.InvalidDataset import InvalidDataset

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

def main(dataframe_file: str, opts: object) -> Union[None, Tuple[float, float]]:
	df = pd.read_csv(dataframe_file)
	try:
		linreg_df = linearRegressionDataframe(df)
	except Exception as err:
		print(err)
		exit(1)
	linreg_df.max_iter = opts.get('max_iter', DEFAULT_MAX_ITER)
	linreg_df.alpha = opts.get('alpha', DEFAULT_LEARNING_RATE)
	linreg_df.normalize(linreg_df.df.price.max(), linreg_df.df.km.max())
	if not opts['quiet']:
		print(f"Imported dataframe: m={linreg_df.m}, n={linreg_df.n}, target=\"{DATAFRAME_TARGET}\"")
	linreg_df.train(opts, verbose=not opts['quiet'])
	linreg_df.restore_normalize_df()
	linreg_df.restore_normalize_theta()
	if opts['visualize']:
		if not opts['least_square']:
			show_cost_history_graph(linreg_df)
		show_linreg_graph(linreg_df)
	return (linreg_df.theta)

if __name__ == '__main__':
	parser = OptionParser(usage="usage: %prog [options] dataset_file")
	parser.add_option("-v", "--visualize",
		action="store_true", dest="visualize", default=False,
		help="show cost history & regression graph")
	parser.add_option("-q", "--quiet",
		action="store_true", dest="quiet", default=False,
		help="hides every stdout output")
	parser.add_option("-a", "--alpha", "--learning-rate",
		action="store", type="float", default=DEFAULT_LEARNING_RATE,
		help="train using specified learning rate")
	parser.add_option("-i", "--max-iter",
		action="store", type="int", default=DEFAULT_MAX_ITER,
		help="train using specified max_iter")
	parser.add_option("-l", "--least-square",
		action="store_true", dest="least_square", default=False,
		help="train using the least square algorithm (more precise but slow with big datasets)")
	(opts, args) = parser.parse_args()
	if len(args) == 0:
		parser.print_help()
		exit(0)
	ret = main(args[0], vars(opts))
	if ret is None:
		exit(1)
	try:
		with open(THETA_SAVE_FILENAME, "w") as f:
			f.write(f"theta0,theta1\n{ret[0]},{ret[1]}")
			f.close()
			if vars(opts)['quiet'] == False:
				print(f"Results saved to {THETA_SAVE_FILENAME}")
	except Exception as err:
		print("An error occurred while saving results to {THETA_SAVE_FILENAME}:")
		print(err)
		exit(1)
