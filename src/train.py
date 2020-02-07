#!/usr/bin/env python3
import typing
import sys
import pandas
import matplotlib.pyplot as plt

def read_data_file(filename: str) -> object:
	df = pandas.read_csv(filename)
	# print(df)
	return (df)

def show_dataframe(df: object) -> None:
	scatterplot = df.plot.scatter(
		x='km',
		y='price'
	)
	plt.show()

def calculate_error_function(df: object, theta: int):
	# h(x) = θ1x + θ0
	print(f"df.size: {df.size}")
	dfsum = 0
	test_arr = [[1, 1], [2, 2], [3, 3]]
	for e in test_arr:
		dfsum += (theta * e[0] - e[1]) ** 2
		print(f"dfsum: {dfsum}")

	return (
		(1 / (2 * len(test_arr))) * dfsum
	)
	# return (
	# 	(1 / (2 * df.size)) * dfsum
	# )

def main():
	if (len(sys.argv) < 2):
		print(f"Usage: ./{sys.argv[0]}")
		exit(0)
	df = read_data_file(sys.argv[1])
	# show_dataframe(df)
	print(calculate_error_function(df, 0.5))

if __name__ == "__main__":
	main()
