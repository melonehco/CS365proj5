"""
loadNetwork.py
loads a network from a file to do digit recognition using the MNIST data base

Zena Abulhab and Melody Mao
CS365 Spring 2019
Project 5
"""

import sys
import numpy
import keras
from buildNetwork import loadData
from pathlib import Path
import cv2

'''
reads in image (JPG) files from the given directory,
converts them to grayscale, and resizes them to the given dimensions
returns a numpy array of the images and a list of the filenames, sorted by filename
'''
def readInImagesFromDir(dirName, img_rows, img_cols, displayWindows=False):
	path = Path( dirName )
	
	if not path.is_dir( ):
		print( "Error:", dirName, "is not a directory" )
		exit( )
	
	imgPaths = list( path.glob( '**/*.jpg' ) )
	imgPaths.extend( path.glob( '**/*.png' ) )
	imgPathStrs = [ ]
	imgNames = []
	for p in imgPaths:
		pathStr = str(p)
		imgPathStrs.append( pathStr )
		
		imgNames.append( pathStr.split('/')[-1] )
	imgPathStrs.sort()
	imgNames.sort()
	
	images = [ ]
	i = 0
	for n in imgPathStrs:
		img = cv2.imread( n, cv2.IMREAD_COLOR )
		img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
		
		if displayWindows:
			cv2.imshow( "image {:d}".format( i ), img )
		
		img = cv2.resize( img, (img_rows, img_cols) )
		img = img[ :, :, numpy.newaxis ]
		images.append( img )
		i += 1
	
	if displayWindows:
		cv2.waitKey( 0 )
		cv2.destroyAllWindows( )
	
	return numpy.asarray( images ), imgNames

def main(argv):
	
	# input image dimensions
	img_rows, img_cols = 28, 28
	
	#x_test = None
	#y_test = None
	if len(argv) > 1:  # if a data source folder was provided
		x_test, filenames = readInImagesFromDir(argv[1], img_rows, img_cols, True)
		y_test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	else:
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
		print("  predicted index:", maxIdx)
		
		print("  true label:", y_test[i])

if __name__ == "__main__":
	main(sys.argv)
