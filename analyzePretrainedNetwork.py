"""
analyzePretrainedNetwork.py
downloads a pre-trained network from Keras and analyzes its first couple of layers

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
def visualize3ChannelFilters(filters):
	# display weights as color-coded grids
	pyplot.figure(figsize=(8, 11)) # figure size in inches
	
	weights = numpy.array(filters) # make a copy to avoid changing the original
	minVal = weights.min()
	
	# normalize values for display
	if minVal < 0:
		weights = weights + -minVal
	maxVal = weights.max()
	weights = weights * 255.0 / maxVal
	
	numFilters = weights.shape[3]
	plotRows = numFilters / 4
	plotCols = 4
	for filterNum in range(numFilters):
		# plot image here
		pyplot.subplot( plotRows, plotCols, filterNum + 1 )
		pyplot.imshow( weights[:, :, :, filterNum] )
	# -----------------------------------------------------------------------------------------------
	
	pyplot.savefig( "pdfs/filtersVGG16.pdf" )
	print( "Drew pre-trained filters into PDF..." )

def main():
	model = VGG16( weights='imagenet', include_top=False )
	weights = model.layers[ 1 ].get_weights( )[ 0 ]  # index 0 for weights, 1 for bias

	print( "Loaded in pre-trained model..." )
	model.summary()
	print( "first layer weights dims:", weights.shape )
	
	visualize3ChannelFilters(weights)
	
	# read in input image
	img_path = 'images/ruby.JPG'
	img = image.load_img( img_path, target_size=(224, 224) )
	img_data = image.img_to_array( img )
	img_data = numpy.expand_dims( img_data, axis=0 )
	img_data = preprocess_input( img_data )
	
	runPartialNetwork( model, img_data, "pdfs/vgg16layer1.pdf", 1, figsize=(8, 11) ) # figure size in inches
	runPartialNetwork( model, img_data, "pdfs/vgg16layer2.pdf", 2, figsize=(8, 11) ) # figure size in inches

if __name__ == "__main__":
	main()