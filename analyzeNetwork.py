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
from keras.models import Model

'''
visualizes the given filters along a blue-red color spectrum and saves the result to a PDF
'''
def visualizeFilters(weights):
	# display weights as color-coded grids
	pyplot.figure( )
	
	# ----- adapted from: https://stackoverflow.com/questions/49775515/visualise-filters-in-keras-cnn
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
		
		redBlueGrid = numpy.zeros( (filter.shape[ 0 ], filter.shape[ 1 ], 3) )  # 3x3, 3 color channels
		redBlueGrid[ :, :, 0 ] = imageRed
		redBlueGrid[ :, :, 2 ] = -imageBlue
		
		# plot image here
		pyplot.subplot( 8, 4, filterNum + 1 )
		pyplot.imshow( redBlueGrid )
	# -----------------------------------------------------------------------------------------------
	
	pyplot.savefig( "pdfs/filters.pdf" )
	print( "Drew filters into PDF..." )

'''
applies the given filters to the given image and saves the result to a PDF
'''
def applyFiltersToImage(filters, image):
	# apply filters to first training example
	filteredImgs = [ ]
	for filterNum in range( 32 ):
		filter = filters[ :, :, 0, filterNum ]
		dst = cv2.filter2D( image, -1, filter )  # -1 to use src depth
		
		# if min val is negative, shift values to be in non-negative range
		minVal = dst.min( )
		if minVal < 0:
			dst = dst + -minVal
			maxVal = dst.max( )
			dst = dst * 255.0 / maxVal
		
		filteredWColor = numpy.zeros( (dst.shape[ 0 ], dst.shape[ 1 ], 3) )  # img rows x cols, 3 color channels
		filteredWColor[ :, :, 0 ] = dst  # red
		filteredWColor[ :, :, 1 ] = dst  # green
		filteredWColor[ :, :, 2 ] = 255.0 - dst  # blue
		
		filteredWColor = filteredWColor.astype( "uint8" )
		
		filteredImgs.append( filteredWColor )
	
	pyplot.figure( )
	for filterNum in range( 32 ):
		pyplot.subplot( 8, 4, filterNum + 1 )
		pyplot.imshow( filteredImgs[ filterNum ] )
	
	pyplot.savefig( "pdfs/filterResults.pdf" )
	print( "Drew filtered images into PDF..." )

# Takes in one or more images to run through layers of a given network and saves to a PDF.
# The default number of layers is 1.
def runPartialNetwork(model, imageInput, filename, layersUpTo = 1, figsize = None):
	partialModel = Model(inputs=model.input, outputs=model.get_layer(index=layersUpTo).output)
	predictions = partialModel.predict(imageInput)

	pyplot.figure(figsize=figsize)
	numFilters = predictions.shape[3]
	plotCols = 4
	plotRows = numFilters / plotCols
	for filterNum in range(numFilters):
		pyplot.subplot(plotRows, plotCols, filterNum + 1)
		pyplot.imshow(predictions[0, :, :, filterNum])

	pyplot.savefig(filename)
	print("Drew partial network output into a PDF...")

def main( argv ):
	# load model from file
	model = keras.models.load_model( "./mnistModel.h5" )
	weights = model.layers[0].get_weights()[0] # index 0 for weights, 1 for bias
	
	print("Loaded in model...")
	print("first layer weights dims:", weights.shape)
	
	# print out weights
	for i in range(weights.shape[3]):
		print("filter", i)
		for x in range(weights.shape[0]):
			line = ""
			for y in range(weights.shape[1]):
				line += " " + str(weights[x][y][0][i])
			print(line)
		print()
	
	visualizeFilters(weights)
	
	# load MNIST example data
	x_train, y_train, x_test, y_test, input_shape = loadData( )
	firstImage = x_train[0]
	
	applyFiltersToImage(weights, firstImage)

	# run partial versions of network on different input images
	srcArray = numpy.array([firstImage])
	runPartialNetwork(model, srcArray, "pdfs/oneLayerResults.pdf", 1)
	runPartialNetwork(model, srcArray, "pdfs/twoLayerResults.pdf", 2)
	runPartialNetwork(model, srcArray, "pdfs/threeLayerResults.pdf", 3)

	altSrcArray1 = numpy.array([x_train[1]])
	runPartialNetwork(model, altSrcArray1, "pdfs/poolingAlt1.pdf", 3)
	altSrcArray2 = numpy.array([x_train[2]])
	runPartialNetwork(model, altSrcArray2, "pdfs/poolingAlt2.pdf", 3)
	altSrcArray3 = numpy.array([x_train[3]])
	runPartialNetwork(model, altSrcArray3, "pdfs/poolingAlt3.pdf", 3)


if __name__ == "__main__":
	main( sys.argv )
