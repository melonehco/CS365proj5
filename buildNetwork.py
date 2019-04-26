"""
buildNetwork.py
builds and trains a network to do digit recognition using the MNIST data base
adapted from example: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

Zena Abulhab and Melody Mao
CS365 Spring 2019
Project 5
"""

import numpy
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

def loadData():
	# input image dimensions
	img_rows, img_cols = 28, 28
	
	# split data between train and test sets
	# x arrays are uint8 grayscale image data with shape (num_samples, 28, 28)
	# y arrays are uint8 arrays of num_samples digit labels (integers in range 0-9)
	(x_train, y_train), (x_test, y_test) = mnist.load_data( )
	
	# uncomment to display first two digits in OpenCV windows
	# print("Displaying first two digits in dataset, press any key to close")
	# cv2.imshow("digit 1", x_train[0])
	# cv2.imshow("digit 2", x_train[1])
	# cv2.waitKey(0)
	# cv2.destroyAllWindows( )
	
	# check how backend (i.e. tensorflow) stores image data
	if K.image_data_format( ) == 'channels_first':
		x_train = x_train.reshape( x_train.shape[ 0 ], 1, img_rows, img_cols )
		x_test = x_test.reshape( x_test.shape[ 0 ], 1, img_rows, img_cols )
		input_shape = (1, img_rows, img_cols)
	else:
		x_train = x_train.reshape( x_train.shape[ 0 ], img_rows, img_cols, 1 )
		x_test = x_test.reshape( x_test.shape[ 0 ], img_rows, img_cols, 1 )
		input_shape = (img_rows, img_cols, 1)
	
	# convert values from uint8
	x_train = x_train.astype( 'float32' )
	x_test = x_test.astype( 'float32' )
	x_train /= 255
	x_test /= 255
	# print( 'x_train shape:', x_train.shape )
	# print( x_train.shape[ 0 ], 'train samples' )
	# print( x_test.shape[ 0 ], 'test samples' )
	
	return (x_train, y_train, x_test, y_test, input_shape)

def main():
	# remove this line to create different networks (otherwise will randomize the same way)
	numpy.random.seed(42)
	
	batch_size = 128
	num_classes = 10 # 10 digits
	epochs = 12
	
	x_train, y_train, x_test, y_test, input_shape = loadData()
	
	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical( y_train, num_classes )
	y_test = keras.utils.to_categorical( y_test, num_classes )
	
	# init model and build up stack of layers
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
					 activation='relu',
					 input_shape=input_shape)) # 32 3x3 filters
	model.add(Conv2D(32, kernel_size=(3, 3),
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2))) # 2x2 window
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu')) # 128 nodes
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax')) # 10 output nodes
	
	# set up model for training
	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer=keras.optimizers.Adam(),
				  metrics=['accuracy'])
	
	# train model for given # epochs
	model.fit(x_train, y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  verbose=1,
			  validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	
	# uncomment to use in place of above section
	# train model and print out evaluation on training and testing sets every epoch
	# for i in range(epochs):
	# 	model.fit( x_train, y_train,
	# 			  batch_size=batch_size,
	# 			  epochs=1,
	# 			  verbose=1,
	# 			  validation_data=(x_test, y_test))
	# 	scoreTrain = model.evaluate(x_train, y_train, verbose=0)
	# 	scoreTest = model.evaluate(x_test, y_test, verbose=0)
	# 	print('Epoch ', i, ' train:', scoreTrain[0], ' ', scoreTrain[1])
	# 	print('Epoch ', i, '  test:', scoreTest[0], ' ', scoreTest[1])
	
	# save model to a file
	model.save("./mnistModel.h5")

if __name__ == "__main__":
	main()