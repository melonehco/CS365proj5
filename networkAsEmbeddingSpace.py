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
from loadNetwork import readInImagesFromDir

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
	
def main(argv):
	if len(argv) < 2:
		print("Please give an image directory name.")
		exit()
	
	writeOutLetterData(argv[1])

if __name__ == "__main__":
	main( sys.argv )