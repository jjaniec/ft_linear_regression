#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

from optparse import OptionParser

import typing
from typing import Tuple, Union, List

import IPython

from ft_linear_regression.config import *

def main(theta_save_file: str, args: List[str]) -> Union[None, Tuple[float, float]]:
	try:
		theta_df = pd.read_csv(theta_save_file)
		theta = [theta_df.theta0[0], theta_df.theta1[0]]
	except Exception as err:
		print(f"An error occurred while reading {theta_save_file}:")
		print(err)
		print("Using theta = [0, 0] instead")
		theta = [0, 0]
	# IPython.embed()
	if len(args) >= 1 and args[0].isnumeric():
		km = float(args[0])
	else:
		km = float(input("Enter a km value to guess the price: "))
	print(f"{theta[0] + (theta[1] * km)}")

if __name__ == '__main__':
	parser = OptionParser(usage="usage: %prog [options] km_value")
	parser.add_option("-f", "--theta-file", dest="theta_filename",
				help="file containing theta values", metavar="FILE",
				default=THETA_SAVE_FILENAME)
	(opts, args) = parser.parse_args()
	ret = main(vars(opts)['theta_filename'], args)
	exit(ret)
