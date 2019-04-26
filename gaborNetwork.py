"""
gaborNetwork.py
creates a network with Gabor filters for its first layer, held constant

Zena Abulhab and Melody Mao
CS365 Spring 2019
Project 5
"""

import cv2
import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from buildNetwork import loadData

'''
returns a set of Gabor filters to use in a convolutional layer
'''
def getGaborFilters(shape):
	kernels = []
	kernel_size = (shape[0], shape[1])
	
	sigmaVals = [4.0, 6.0] # standard deviation of Gaussian
	thetaVals = [0.0, numpy.pi * 0.25, numpy.pi * 0.5, numpy.pi * 0.75] # orientation
	lambdaVals = [2.0, 3.0, 4.0, 5.0] # wavelength: ≥ 2 (always), ≤ 5 (based on our input size)
	gammaVal = 0.5 # aspect ratio
	
	for sigVal in sigmaVals:
		for thVal in thetaVals:
			for lamVal in lambdaVals:
				# cv2.getGaborKernel( ksize, sigma, theta, lambd, gamma, psi, data type )
				kernels.append( cv2.getGaborKernel( kernel_size, sigVal, thVal, lamVal, gammaVal, 0, ktype=cv2.CV_32F ) )
	
	kernel_array = numpy.array(kernels) # shape (32, 7, 7)
	
	# restructure array to match expected shape (7, 7, 1, 32) for model layer
	kernel_array = numpy.moveaxis(kernel_array, 0, -1) # move 32-dim axis to end
	kernel_array = numpy.expand_dims( kernel_array, axis=2 ) # add axis to have the 1
	
	return kernel_array
	

def main():
	batch_size = 128
	num_classes = 10  # 10 digits
	epochs = 12
	
	x_train, y_train, x_test, y_test, input_shape = loadData( )
	
	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical( y_train, num_classes )
	y_test = keras.utils.to_categorical( y_test, num_classes )
	
	# init model and build up stack of layers
	model = Sequential( )
	model.add( Conv2D( 32, kernel_size=(7, 7),
					   activation='relu',
					   input_shape=input_shape,
					   kernel_initializer=getGaborFilters, trainable=False) ) # use Gabor filters and don't train this layer
	# remaining layers are set up identically to buildNetwork.py
	model.add( Conv2D( 32, kernel_size=(3, 3),
					   activation='relu' ) )
	model.add( MaxPooling2D( pool_size=(2, 2) ) )  # 2x2 window
	model.add( Dropout( 0.25 ) )
	model.add( Flatten( ) )
	model.add( Dense( 128, activation='relu' ) )  # 128 nodes
	model.add( Dropout( 0.5 ) )
	model.add( Dense( num_classes, activation='softmax' ) )  # 10 output nodes
	
	# set up model for training
	model.compile( loss=keras.losses.categorical_crossentropy,
				   optimizer=keras.optimizers.Adam( ),
				   metrics=[ 'accuracy' ] )
	
	# train model for given # epochs
	model.fit( x_train, y_train,
			   batch_size=batch_size,
			   epochs=epochs,
			   verbose=1,
			   validation_data=(x_test, y_test) )
	score = model.evaluate( x_test, y_test, verbose=0 )
	print( 'Test loss:', score[ 0 ] )
	print( 'Test accuracy:', score[ 1 ] )

if __name__ == "__main__":
	main()