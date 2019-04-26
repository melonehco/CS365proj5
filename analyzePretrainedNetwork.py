"""
gaborNetwork.py
creates a network with Gabor filters for its first layer, held constant

Zena Abulhab and Melody Mao
CS365 Spring 2019
Project 5
"""

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy
import matplotlib
matplotlib.use('Agg') # for saving plot to file
from matplotlib import pyplot
from analyzeNetwork import runPartialNetwork

'''
visualizes the given filters and saves the result to a PDF
'''
def visualize3ChannelFilters(weights):
	# display weights as color-coded grids
	pyplot.figure( )
	
	# ----- adapted from: https://stackoverflow.com/questions/49775515/visualise-filters-in-keras-cnn
	# normalize these filters first, otherwise they won't be in a suitable range for plotting:
	maxVal = weights.max( )
	minVal = weights.min( )
	absMax = max( abs( minVal ), abs( maxVal ) )
	
	normWeights = (weights / absMax) * 255
	
	numFilters = weights.shape[3]
	plotRows = numFilters / 4
	plotCols = 4
	for filterNum in range(numFilters):
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
		pyplot.subplot( plotRows, plotCols, filterNum + 1 )
		pyplot.imshow( redBlueGrid )
	# -----------------------------------------------------------------------------------------------
	
	pyplot.savefig( "pdfs/filtersVGG16.pdf" )
	print( "Drew filters into PDF..." )

def main():
	model = VGG16( weights='imagenet', include_top=False )
	weights = model.layers[ 1 ].get_weights( )[ 0 ]  # index 0 for weights, 1 for bias

	print( "Loaded in pre-trained model..." )
	model.summary()
	print( "first layer weights dims:", weights.shape )
	
	img_path = 'images/ruby.JPG'
	img = image.load_img( img_path, target_size=(224, 224) )
	img_data = image.img_to_array( img )
	img_data = numpy.expand_dims( img_data, axis=0 )
	img_data = preprocess_input( img_data )
	
	runPartialNetwork( model, img_data, "pdfs/vgg16layer1.pdf", 1 )
	runPartialNetwork( model, img_data, "pdfs/vgg16layer2.pdf", 2 )
	runPartialNetwork( model, img_data, "pdfs/vgg16layer3.pdf", 3 )

if __name__ == "__main__":
	main()