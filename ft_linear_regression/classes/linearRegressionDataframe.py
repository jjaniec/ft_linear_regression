#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

import typing
from typing import Tuple, Union

from ft_linear_regression.config import *

class linearRegressionDataframe:
	df = None
	m = 0 # Number of rows
	n = 0 # Number of features
	y = 0 # Index of DATAFRAME_TARGET
	x1 = 0 # Index of DATAFRAME_FEATURE_1

	# Cost function
	last_cost_theta = [0, 0] # Last theta params used with the cost function
	last_cost_result = 0 # Last result of cost function
	cost_history = []

	# Gradient descent
	alpha = 0 # Learning rate
	max_iter = 0 # Max iteration of gradient descent

	# Predict
	theta = [0, 0]

	def __init__(self: object, dataframe: object):
		self.df = dataframe
		self.m, self.n = self.__get_dataframe_props()
		self.y = self.df.columns.get_loc(DATAFRAME_TARGET)
		self.x1 = self.df.columns.get_loc(DATAFRAME_FEATURE_1)
		self.alpha = DEFAULT_LEARNING_RATE

	def __get_dataframe_props(self: object) -> Tuple[str, str]:
		"""
		get m & n from dataset
		"""
		return [
			int(self.df.index.stop - self.df.index.start / self.df.index.step),
			len(self.df.columns) - 1
		]

	def __model(self: object, theta: Tuple[float, float], x: float) -> float:
		"""
		Model:
		f(x) = ax + b
		h(x) = theta1x + theta0
		"""
		return (theta[0] + (theta[1] * x))
	h = __model

	def	__cost_function(self: object, theta: Tuple[float, float]) -> float:
		"""
		Cost function
		j(a, b) = (1 / 2m) * ( (f(x1) - y1)^2 + (f(x2) - y2)^2 + ... + (f(xm) - ym)^2) )
		j(a, b) = (1 / 2m) * (sum of: for i in dataset: (f(xi) - yi) )^2
		j(theta0, theta1) = (1 / 2m) * (sum of: for i in dataset: (htheta(xi) - yi) )^2
		"""
		self.last_cost_theta = theta
		cost = 0
		for i in self.df.iterrows():
			cost += (self.h(theta, i[1][self.x1]) - i[1][self.y]) ** 2
		last_cost_result = (1 / (2 * self.m)) * cost
		return last_cost_result
	j = __cost_function

	def	derivate_theta(self: object, theta: Tuple[float, float]) -> Tuple[float, float]:
		"""
		Calculate partial derivatives
		(∂ / (∂ * theta0)) * j(theta0, theta1) = (1 / m) * (sum of: for i in dataset: (h(xi) - yi))
		(∂ / (∂ * theta1)) * j(theta0, theta1) = (1 / m) * ((sum of: for i in dataset: (h(xi) - yi)) * xi)
		"""
		derivative_theta = [0, 0]
		for i in self.df.iterrows():
			derivative_theta[0] += float(
					self.h(theta, i[1][self.x1]) - i[1][self.y]
				)
			derivative_theta[1] += float(
					(self.h(theta, i[1][self.x1]) - i[1][self.y]) * i[1][self.x1]
				)
		return list(map(
			lambda theta_: theta_ * (1 / self.m),
			derivative_theta
		))

	def	gradient_descent(self: object, theta: Tuple[float, float], derivative_theta: Tuple[float, float]) -> Tuple[float, float]:
		"""
		∂j := ∂j - (alpha * ( (∂ / (∂ * thetaj)) * j(theta0, theta1) ))
		"""
		return [
			float(theta[0] - (self.alpha * derivative_theta[0])),
			float(theta[1] - (self.alpha * derivative_theta[1]))
		]

	def normalize(self: object, reference_y: float, reference_x1: float):
		self.normalize_references = {
			"y": reference_y,
			"x1": reference_x1
		}
		self.df = pd.DataFrame({
			"km": self.df.km / self.df.km.max(),
			"price": self.df.price / self.df.price.max()
		})

	def	restore_normalize_df(self: object) -> object:
		self.df = pd.DataFrame({
			"km": self.df.km * self.normalize_references['x1'],
			"price": self.df.price * self.normalize_references['y']
		})
		return (self.df)

	def restore_normalize_theta(self: object) -> Tuple[float, float]:
		tmp = [
			self.theta[0] * self.normalize_references['y'],
			(self.theta[1] * self.normalize_references['y']) / self.normalize_references['x1']
		]
		self.theta = tmp
		return (self.theta)

	def predict(self: object, x1: float) -> float:
		return (self.h(self.theta, x1))
