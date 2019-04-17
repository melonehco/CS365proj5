"""
loadNetwork.py
loads a network from a file to do digit recognition using the MNIST data base

Zena Abulhab and Melody Mao
CS365 Spring 2019
Project 5
"""

import numpy
import keras
from buildNetwork import loadData

def main():
	
	# input image dimensions
	img_rows, img_cols = 28, 28
	
	x_train, y_train, x_test, y_test, input_shape = loadData()
	x_test = x_test[0:10:1]
	y_test = y_test[0:10:1]
	
	# load model from file
	model = keras.models.load_model("./mnistModel.h5")
	
	predictions = model.predict( x_test ) #default args: batch_size=None, verbose=0, steps=None, callbacks=None )
	
	# print out all outputs
	for i in range(len(predictions)):
		print("example", i, ":")
		line = "  "
		for output in predictions[i]:
			line += "{:04.2f} ".format(output)
		print(line)
		
		maxIdx = numpy.where(predictions[i] == numpy.amax(predictions[i]))[0][0] # indexing into tuple then array
		print("  max index:", maxIdx)
		
		print("  true label:", y_test[i])

if __name__ == "__main__":
	main()
