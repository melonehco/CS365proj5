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

def main(argv):
	
	# input image dimensions
	img_rows, img_cols = 28, 28
	
	#x_test = None
	#y_test = None
	if len(argv) > 1:  # if a data source folder was provided
		path = Path( argv[1] )
		
		if not path.is_dir():
			print("Error:", argv[1], "is not a directory")
			exit()
		
		imgPaths = list( path.glob( '**/*.jpg' ) )
		imgNames = []
		for p in imgPaths:
			imgNames.append( str(p) )
		imgNames.sort()
		print(imgNames)
		
		images = []
		i = 0
		for n in imgNames:
			img = cv2.imread( n, cv2.IMREAD_COLOR )
			img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
			cv2.imshow("image {:d}".format(i), img)
			img = cv2.resize( img, (img_rows, img_cols) )
			img = img[:, :, numpy.newaxis]
			images.append( img )
			i += 1
		
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
		x_test = numpy.asarray( images )
		print("xtest shape", x_test.shape)
		y_test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		#y_test = [2, 6, 8, 0, 1, 4, 7, 3, 5, 9]
		input()
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
		print("  max index:", maxIdx)
		
		print("  true label:", y_test[i])
	
	predClasses = model.predict_classes( x_test )
	
	for i in range( len( predClasses ) ):
		print( "predicted:", predClasses[i], "true:", y_test[i] )

if __name__ == "__main__":
	main(sys.argv)
