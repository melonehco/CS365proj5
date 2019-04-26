"""
optimizeNetwork.py
creates many variations on our network from buildNetwork.py
to try to optimize the network performance and/or training time by adjusting
the convolution filter size, the dropout rate(s), and the number of epochs

Zena Abulhab and Melody Mao
CS365 Spring 2019
Project 5
"""

import sys
import numpy
import keras
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from buildNetwork import loadData
import KerasTimer

# global timer callback variable
timerCallback = KerasTimer.KerasTimer()

# Global training time list.
# appended to after each training session, and saved to csv when all are complete
times = []

'''
tests a network variation with the given convolution filter size,
dropout rates, and number of epochs
'''
def tryNetworkVariation(data, convFilterDim, dropRate1, dropRate2, numEpochs):
	batch_size = 128
	num_classes = 10  # 10 digits
	
	x_train, y_train, x_test, y_test, input_shape = data
	
	# init model and build up stack of layers
	model = Sequential( )
	model.add( Conv2D( 32, kernel_size=(convFilterDim, convFilterDim),
					   activation='relu',
					   input_shape=input_shape ) )  # 32 3x3 filters
	model.add( Conv2D( 32, kernel_size=(convFilterDim, convFilterDim),
					   activation='relu' ) )
	model.add( MaxPooling2D( pool_size=(2, 2) ) )  # 2x2 window
	model.add( Dropout( dropRate1 ) )
	model.add( Flatten( ) )
	model.add( Dense( 128, activation='relu' ) )  # 128 nodes
	model.add( Dropout( dropRate2 ) )
	model.add( Dense( num_classes, activation='softmax' ) )  # 10 output nodes
	
	# set up model for training
	model.compile( loss=keras.losses.categorical_crossentropy,
				   optimizer=keras.optimizers.Adam( ),
				   metrics=[ 'accuracy' ] )
	
	# train model for given # epochs
	model.fit( x_train, y_train,
			   batch_size=batch_size,
			   epochs=numEpochs,
			   callbacks=[timerCallback],
			   verbose=1,
			   validation_data=(x_test, y_test) )
	score = model.evaluate( x_test, y_test, verbose=0 )

	times.append(timerCallback.times)

	return score

def writeTimesToCSV():
	numpy.savetxt("TrainingTimerResults.csv", times, delimiter=",", header="training epoch times")

def main():
	num_classes = 10  # 10 digits
	
	x_train, y_train, x_test, y_test, input_shape = loadData( )
	
	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical( y_train, num_classes )
	y_test = keras.utils.to_categorical( y_test, num_classes )

	data = (x_train, y_train, x_test, y_test, input_shape)
	
	convFilterOptions = [1, 2, 3, 4]
	dropRateOptions = [(0.15, 0.4), (0.25, 0.5), (0.35, 0.6), (0.5, 0.25)]
	numEpochsOptions = [8, 10, 12, 14]
	convFilterIDX = 0 # index of current optimal value
	dropRateIDX = 0 # index of current optimal value
	numEpochsIDX = 0 # index of current optimal value

	for i in range(5): # optimize all three parameters 5 times
		print("-------------------- Optimization run", i, "--------------------")

		# optimize convolution filter size
		maxAccuracy = 0.0 # best accuracy value seen so far
		optimalIDX = convFilterIDX # index of best parameter value used so far
		for paramIDX in range(4):
			print("-----> Optimizing convolution filter size, iteration", paramIDX)
			score = tryNetworkVariation(data, convFilterOptions[paramIDX], dropRateOptions[dropRateIDX][0],
										dropRateOptions[dropRateIDX][1], numEpochsOptions[numEpochsIDX])
			if score[1] > maxAccuracy:
				maxAccuracy = score[1]
				optimalIDX = paramIDX
		convFilterIDX = optimalIDX
		
		# optimize dropout rates
		maxAccuracy = 0.0 # best accuracy value seen so far
		optimalIDX = dropRateIDX # index of best parameter value used so far
		for paramIDX in range(4):
			print( "-----> Optimizing dropout rates, iteration", paramIDX )
			score = tryNetworkVariation(data, convFilterOptions[convFilterIDX], dropRateOptions[paramIDX][0],
										dropRateOptions[paramIDX][1], numEpochsOptions[numEpochsIDX])
			if score[1] > maxAccuracy:
				maxAccuracy = score[1]
				optimalIDX = paramIDX
		dropRateIDX = optimalIDX
		
		# optimize number of epochs
		maxAccuracy = 0.0 # best accuracy value seen so far
		optimalIDX = numEpochsIDX # index of best parameter value used so far
		for paramIDX in range(4):
			print("-----> Optimizing number of epochs, iteration", paramIDX)
			score = tryNetworkVariation(data, convFilterOptions[convFilterIDX], dropRateOptions[dropRateIDX][0],
										dropRateOptions[dropRateIDX][1], numEpochsOptions[paramIDX])
			if score[1] > maxAccuracy:
				maxAccuracy = score[1]
				optimalIDX = paramIDX
		numEpochsIDX = optimalIDX

		print("Now using parameter values:")
		print("    conv filter size =", convFilterOptions[ convFilterIDX ])
		print("    dropout rates =", dropRateOptions[ dropRateIDX ])
		print("    number of epochs =", numEpochsOptions[ numEpochsIDX ])

	print("Optimal parameter values:")
	print("    conv filter size =", convFilterOptions[ convFilterIDX ])
	print("    dropout rates =", dropRateOptions[ dropRateIDX ])
	print("    number of epochs =", numEpochsOptions[ numEpochsIDX ])

	writeTimesToCSV()


	'''
	//have 4 options per parameter

	//2 dropout layers, currently rates are 0.25 and 0.5
	
	//options: 0.25 & 0.5, 0.5 & 0.25, 0.15 & 0.4, 0.35 & 0.6
	
	//2 conv layers, currently filters are 3x3
	
	//options: 1x1, 2x2, 3x3, 4x4
	
	//currently 12 epochs
	
	//options: 8, 10, 12, 14
	'''
if __name__ == "__main__":
	main()