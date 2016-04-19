import os
import random
import re

import numpy as np

from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
'''
import nltk.classify.util, nltk.metrics
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
'''

from sklearn import linear_model
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix

stemmer = PorterStemmer()
letterPattern = re.compile('[^a-zA-Z0-9 ]+')

# Top N most frequent words/characters go here after the feature 
wordList = []
characterList = []

# Class to represent all features about given text
class TextFeatures:
	wordsOccurringOnce = None # words_occurring_once/total_words
	lexicalRichness = None # distinct_words/total_words
	# any additional features go here
		
# Convert TextFeatures to list for ML algorithms
def featuresToList(features):
	# Append features to an empty list in given order
	# Take care of features not needed or not included
	result = []
	result.append(features.wordsOccurringOnce)
	result.append(features.lexicalRichness)
	return result
	
# Select features to be used given files
# Goal: set global variables to proper values
# For example, find N most used words/characters
def selectFeatures(filenames, labels):
	pass
	
# Produces a list of features for a file
def getFeatures(filename):
	features = TextFeatures()
	# read the file
	with open(filename) as fp:
		raw = fp.read().decode('utf8').lower()
	# print len(set(raw.split())) # how many different words before processing
	# remove non-letters
	text = letterPattern.sub('', raw)
	# stem
	text = [stemmer.stem(word) for word in text.split()]
	# print len(set(text)) # how many different words after processing
	# print sorted(set(text)) # show individual words
	#features.wordsOccurringOnce = 
	freqDist = FreqDist(text)
	#print freqDist.most_common(50)
	wordsOccurringOnce = [w for w in freqDist.keys() if freqDist[w]==1]
	features.wordsOccurringOnce = len(wordsOccurringOnce)/float(len(text))
	features.lexicalRichness = len(set(text))/float(len(text))
	return featuresToList(features)
	
# Evaluate classifiers using N-fold cross validation
# Input: classifier (one of global vars), features and labels for each text
def evaluate (classifier, filenames, labels, numberOfFolds=5):
	skf = StratifiedKFold(labels, numberOfFolds)
	confusion_matrices = []
	for train, test in skf:
		print 'Training on ' + str(len(train)) + ' samples. Testing on ' + str(len(test)) + ' samples.'
		trainingFilenames = []
		trainingLabels = []
		testingFilenames = []
		actualLabels = []
		# Separate training and test data
		for i in range(len(labels)):
			if i in train:
				trainingFilenames.append(filenames[i])
				trainingLabels.append(labels[i])
			elif i in test:
				testingFilenames.append(filenames[i])
				actualLabels.append(labels[i])
		# Figure out most frequent words/characters in the training dataset
		selectFeatures(trainingFilenames, trainingLabels)
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
	# Get filenames and labels for training data
	dataFolder = 'data/'
	filenames = [os.path.join(dataFolder, f) for f in os.listdir(dataFolder) if f.endswith('.txt') and os.path.isfile(os.path.join(dataFolder, f))]
	labels = [os.path.basename(f).split('-')[0] for f in filenames]
	numAuthors = len(set(labels))
	print 'Overall dataset:'
	for author in set(labels):
		print 'Author: ' + str(author) + '    Files: ' + str(labels.count(author))
	# Define classifiers to compare
	svmClassifier = svm.SVC()
	logRegClassifier = linear_model.LogisticRegression()
	kMeans = KMeans(n_clusters=numAuthors)
	# Produce the results
	print 'Testing Logistic Regression'
	#print getFeatures(filenames[0])
	evaluate (logRegClassifier, filenames, labels)
	print 'Testing SVM'
	evaluate (svmClassifier, filenames, labels)
