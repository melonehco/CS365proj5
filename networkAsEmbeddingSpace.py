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
		
	unique, mapping = numpy.unique( numpy.array( labels ), return_inverse=True )
	
	# write data out to file
	# use flatten method to flatten each (image) np array to 1d
	# combine them into 1 np array (# images x 784)
	# then use numpy.savetxt, passing in a header row, to write out to file
	
def main(argv):
	if len(argv) < 2:
		print("Please give an image directory name.")
		exit()
	
	writeOutLetterData(argv[1])

if __name__ == "__main__":
	main( sys.argv )