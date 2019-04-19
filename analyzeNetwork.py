"""
analyzeNetwork.py
loads a network from a file to do digit recognition using the MNIST data base

Zena Abulhab and Melody Mao
CS365 Spring 2019
Project 5
"""

import sys
import keras
import matplotlib
matplotlib.use('Agg') # for saving plot to file
from matplotlib import pyplot
import numpy
import cv2
from buildNetwork import loadData

def main( argv ):
	# load model from file
	model = keras.models.load_model( "./mnistModel.h5" )
	weights = model.layers[0].get_weights()[0] # index 0 for weights, 1 for bias
	
	print("Loaded in model...")
	print("weights dims:", weights.shape)
	
	# print out weights
	for i in range(weights.shape[3]):
		print("filter", i)
		for x in range(weights.shape[0]):
			line = ""
			for y in range(weights.shape[1]):
				line += " " + str(weights[x][y][0][i])
			print(line)
		print()
	
	# display weights as color-coded grids
	pyplot.figure()
	
	#----- adapted from: https://stackoverflow.com/questions/49775515/visualise-filters-in-keras-cnn
	# normalize these filters first, otherwise they won't be in a suitable range for plotting:
	maxVal = weights.max( )
	minVal = weights.min( )
	absMax = max( abs( minVal ), abs( maxVal ) )
	
	normWeights = (weights / absMax) * 255
	
	for filterNum in range( 32 ):
		filter = normWeights[ :, :, 0, filterNum ]
		
		# a trick to see negatives as blue and positives as red
		imageRed = numpy.array( filter )
		imageBlue = numpy.array( filter )
		imageRed[ imageRed < 0 ] = 0
		imageBlue[ imageBlue > 0 ] = 0
		
		# print( imageRed )
		# print( imageBlue )
		
		redBlueGrid = numpy.zeros( (filter.shape[ 0 ], filter.shape[ 1 ], 3) ) # 3x3, 3 color channels
		redBlueGrid[ :, :, 0 ] = imageRed
		redBlueGrid[ :, :, 2 ] = -imageBlue
		
		# plot image here
		pyplot.subplot(8, 4, filterNum+1)
		pyplot.imshow( redBlueGrid )
	#-----------------------------------------------------------------------------------------------
	
	pyplot.savefig( "filters.pdf" )
	print("Drew filters into PDF...")
	
	# load MNIST example data
	x_train, y_train, x_test, y_test, input_shape = loadData( )
	src = x_train[0]
	
	# apply filters to first training example
	print("Displaying filtered images...")
	filteredImgs = []
	for filterNum in range( 32 ):
		filter = normWeights[ :, :, 0, filterNum ]
		dst = cv2.filter2D( src, -1, filter) # -1 to use src ddepth
		filteredImgs.append(dst)
		
		cv2.imshow("filtered img {:d}".format(filterNum), dst)
	
	print("Press any key in image windows to close all")
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
if __name__ == "__main__":
	main( sys.argv )
