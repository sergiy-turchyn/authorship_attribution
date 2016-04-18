import os
import random

import numpy as np

'''
import nltk.classify.util, nltk.metrics
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
'''

from sklearn import linear_model
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix

# Top N most frequent words/characters go here after the feature 
wordList = []
characterList = []

# Classifiers to compare
svmClassifier = svm.SVC()
logRegClassifier = linear_model.LogisticRegression()

# Class to represent all features about given text
class TextFeatures:
	wordFreqencies = {} # pairs word->frequency
	characterFrequencies = {}
	# any additional features go here
		
# Convert TextFeatures to list for ML algorithms
def featuresToList(features):
	# Append features to an empty list in given order
	# Take care of features not needed or not included
	pass
	
# Select features to be used given files
# Goal: set global variables to proper values
# For example, find N most used words/characters
def selectFeatures(filenames):
	pass
	
# Produces a list of features for a file
def getFeatures(filename):
	features = TextFeatures()
	# fill in all the fields, create separate functions for each feature if possible
	# use wordList and characterList
	return featuresToList(features)
	
# Train the classifier
# texts - list of filenames with text
# labels - author of each text
def train (filenames, labels):
	# process the files to get most frequent words and characters, putting the result into the global variables
	# for each file, get features as a list
	# use features and labels for all files to train the classifiers
	pass
	
# Evaluate classifiers using N-fold cross validation
# Input: classifier (one of global vars), features and labels for each text
def evaluate (classifier, filenames, labels, numberOfFolds=5):
	skf = StratifiedKFold(labels, numberOfFolds)
	confusion_matrices = []
	for train, test in skf:
		trainingFilenames = []
		trainingLabels = []
		testingFilenames = []
		actualLabels = []
		# Separate training and test data
		for i in range(len(labels)):
			if i in train:
				#trainingData.append(features[i])
				trainingFilenames.append(filenames[i])
				trainingLabels.append(labels[i])
			elif i in test:
				#testingData.append(features[i])
				testingFilenames.append(filenames[i])
				actualLabels.append(labels[i])
		# Figure out most frequent words/characters in the training dataset
		selectFeatures(trainingFilenames)
		# Get features
		trainingData = []
		testingData = []
		for fn in trainingFilenames:
			trainingData.append(getFeatures(fn))
		for fn in testingFilenames:
			testingData.append(getFeatures(fn))
		# Train on training data
		classifier.fit(trainingData, trainingLabels)
		# Classify testing data
		predictedLabels = classifier.predict(testingData)
		# Produce confusion matrix
		cf_mat = confusion_matrix(actualLabels, predictedLabels)
		confusion_matrices.append(cf_mat)
	# Print confusion matrix with average values
	confusion_matrices_numpy = np.array(confusion_matrices)
	mean_cf_mat = np.mean(confusion_matrices_numpy, axis=0)
	mean_cf_mat = mean_cf_mat/mean_cf_mat.sum(axis=1)[:,None]
	print mean_cf_mat
	
if __name__ == '__main__':
	# This code is run by the program. Put testing code here.
	# In the final version, 1) get filenames and labels for training data, 2) call evaluate on different classifiers
	print 'it works'