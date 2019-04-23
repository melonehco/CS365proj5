"""
networkAsEmbeddingSpace.py
loads a network from a file to use output from a hidden layer
as an embedding space for Greek letters

Zena Abulhab and Melody Mao
CS365 Spring 2019
Project 5
"""

import sys
import numpy
import keras
from keras import Model
from buildNetwork import loadData
from loadNetwork import readInImagesFromDir

'''
reads in all images from the given directory
writes them into a CSV file where each row is one image (flattened to 1-D)
determines their letter categories from the first word in their filenames
and writes the category data (as number categories) into another CSV file
'''
def writeOutLetterData(dirName):
	letterImages, filenames = readInImagesFromDir(dirName, 28, 28)
		
	# determine correct categories from filenames
	labels = []
	for name in filenames:
		labels.append( name.split('_')[0] )
	
	# turn category data into numbers for each category
	unique, mapping = numpy.unique( numpy.array( labels ), return_inverse=True )
	
	# combine image data into 1 np array (# images x 784)
	data = numpy.zeros((len(filenames), 784))
	# use flatten method to flatten each (image) np array to 1d
	for imgNum in range(letterImages.shape[0]):
		data[imgNum] = letterImages[imgNum].flatten()
	# write data out to file
	numpy.savetxt("letterData.csv", data, delimiter=",", header="this is a header?")
	
	# write category data out to file
	numpy.savetxt( "letterCats.csv", mapping, delimiter=",", header="true category" )

'''
reads in a letter image data CSV file and a category data CSV file
from the given filenames and returns their contents as numpy arrays
image data is returned as 28x28 (instead of a single row of 784 values)
'''
def readInLetterData(dataFile, catFile):
	raw_data = numpy.genfromtxt( dataFile, delimiter=',' )
	numImages = raw_data.shape[0]
	
	# restructure letter image data to be 28x28 again
	my_data = numpy.zeros( (numImages, 28, 28) )
	for imgNum in range(numImages):
		for i in range(28):
			startIdx = i * 28
			endIdx = startIdx + 28
			my_data[imgNum][i] = raw_data[imgNum][startIdx:endIdx]
	my_data = numpy.expand_dims(my_data, axis=3)
	
	my_cats = numpy.genfromtxt( catFile, delimiter=',' )
	
	return my_data, my_cats
	
def main(argv):
	if len(argv) < 2:
		print("Please give an image directory name.")
		exit()
	
	# writeOutLetterData(argv[1])
	# print("Wrote out letter data CSV files...")
	
	# load model from file
	model = keras.models.load_model( "./mnistModel.h5" )
	truncatedModel = Model( inputs=model.input, outputs=model.get_layer( index=6 ).output )
	
	print( "Loaded in model..." )
	
	# load MNIST example data
	x_train, y_train, x_test, y_test, input_shape = loadData( )
	firstImage = x_train[ 0 ]
	img1array = numpy.array( [ firstImage ] )
	
	# run truncated model with just the first training image
	predictions = truncatedModel.predict( img1array )
	
	print("Dense layer output shape: ", predictions.shape)
	
	# apply the truncated network to the greek symbols (read from the CSV file)
	letterData, letterCats = readInLetterData("letterData.csv", "letterCats.csv")
	numExamples = letterData.shape[0]
	letterVectors = truncatedModel.predict( letterData ) # a array w/ a 128-elem vector for each letter example
			
	# calculate SSD between first alpha example vector and each other example vector
	alphaSSDs = "alphas: "
	betaSSDs = "alphas: "
	gammaSSDs = "alphas: "
	for i in range(numExamples):
		alphaSSDs += " " + str(numpy.sum((letterVectors[0,:]-letterVectors[i,:]))**2) # 0 for first alpha
		betaSSDs += " " + str(numpy.sum((letterVectors[9,:]-letterVectors[i,:]))**2) # 9 for first beta
		gammaSSDs += " " + str(numpy.sum((letterVectors[18,:]-letterVectors[i,:]))**2) # 18 for first gamma
		
		if i == 8:
			alphaSSDs += "\nbetas:"
			betaSSDs += "\nbetas:"
			gammaSSDs += "\nbetas:"
		elif i == 17:
			alphaSSDs += "\ngammas:"
			betaSSDs += "\ngammas:"
			gammaSSDs += "\ngammas:"
	
	print("SSDs of first alpha")
	print(alphaSSDs, "\n")
	print("SSDs of first beta")
	print(betaSSDs, "\n")
	print("SSDs of first gamma")
	print(gammaSSDs)
		

if __name__ == "__main__":
	main( sys.argv )