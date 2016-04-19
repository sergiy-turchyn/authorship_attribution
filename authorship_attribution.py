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

# How many most used words/characters we take from each author
wordsPerAuthor = 30

'''
What features we can easily add (main task for tomorrow):
- word and character n-gram frequencies. nltk makes it easy to extract n-grams, so we can also add top N bigrams (bigrams are two words or characters following each other).
- frequencies of specific characters (alphabetic, numeric, punctiation, uppercase, etc.)
'''

# Class to represent all features about given text
class TextFeatures:
	wordsOccurringOnce = None # words_occurring_once/total_words
	wordsOccurringTwice = None # words_occurring_twice/total_words
	lexicalRichness = None # distinct_words/total_words
	avgWordLength = None
	medianWordLength = None
	stdWordLength = None
	wordFrequencies = {}
	author = None # cheating feature
	# any additional features go here
		
# Convert TextFeatures to list for ML algorithms
def featuresToList(features):
	# Append features to an empty list in given order
	# Take care of features not needed or not included
	result = []
	
	result.append(features.wordsOccurringOnce)
	result.append(features.wordsOccurringTwice)
	result.append(features.lexicalRichness)
	result.append(features.avgWordLength)
	result.append(features.medianWordLength)
	result.append(features.stdWordLength)
	result.append(features.author)
	
	for word in wordList:
		if (word in features.wordFrequencies.keys()):
			result.append(features.wordFrequencies[word])
		else:
			result.append(0.0)
	
	return result
	
# Select features to be used given files
# Goal: set global variables to proper values
# For example, find N most used words/characters
def selectFeatures(filenames, labels):
	# Select 10 most used words from each author
	for author in set(labels):
		authorFilenames = [f for f in filenames if os.path.basename(f).split('-')[0]==author]
		wordFreq = FreqDist()
		for filename in authorFilenames:
			with open(filename) as fp:
				raw = fp.read().decode('utf8').lower()
			# remove non-letters
			text = letterPattern.sub('', raw)
			# stem
			text = [stemmer.stem(word) for word in text.split()]
			freqDistText = FreqDist(text)
			#print freqDistText.most_common(5)
			for word in freqDistText.keys():
				if word in wordFreq.keys():
					wordFreq[word] = wordFreq[word] + freqDistText[word]
				else:
					wordFreq[word] = freqDistText[word]
		mostCommon = wordFreq.most_common(wordsPerAuthor)
		for (word, count) in mostCommon:
			if word not in wordList:
				wordList.append(word)
	#print wordList
	#print len(wordList)
		
	
# Produces a list of features for a file
def getFeatures(filename):
	features = TextFeatures()
	features.author = float(os.path.basename(filename).split('-')[0])
	# read the file
	with open(filename) as fp:
		raw = fp.read().decode('utf8').lower()
	# print len(set(raw.split())) # how many different words before processing
	# remove non-letters
	text = letterPattern.sub('', raw)
	# get average/median/std of word lengths
	wordLengths = [len(word) for word in text.split()]
	avgWordLength = np.average(wordLengths)
	medianWordLength = np.median(wordLengths)
	stdWordLength = np.std(wordLengths)
	features.avgWordLength = avgWordLength
	features.medianWordLength = medianWordLength
	features.stdWordLength = stdWordLength
	# stem
	text = [stemmer.stem(word) for word in text.split()]
	# calculate frequency data
	freqDist = FreqDist(text)
	for word in wordList:
		if word in freqDist.keys():
			features.wordFrequencies[word] = freqDist[word]/float(len(text))
		else:
			features.wordFrequencies[word] = 0.0
	# calculate words occurring 1-2 times
	wordsOccurringOnce = [w for w in freqDist.keys() if freqDist[w]==1]
	wordsOccurringTwice = [w for w in freqDist.keys() if freqDist[w]==2]
	features.wordsOccurringOnce = len(wordsOccurringOnce)/float(len(text))
	features.wordsOccurringTwice = len(wordsOccurringTwice)/float(len(text))
	features.lexicalRichness = len(set(text))/float(len(text))
	return featuresToList(features)
	
# Display average features for each author
def displayAvgFeatures (filenames, labels):
	selectFeatures(filenames, labels)
	print 'Average feature values for authors: '
	for author in set(labels):
		authorFilenames = [f for f in filenames if os.path.basename(f).split('-')[0]==author]
		avgFeatures = getFeatures(authorFilenames[0])
		for filename in authorFilenames[1:]:
			features = getFeatures(filename)
			for i in range(len(features)):
				avgFeatures[i] += features[i]
		for i in range(len(features)):
			avgFeatures[i] = avgFeatures[i]/len(authorFilenames)
		print 'Author ' + str(author) + ' average features: ' + str(avgFeatures)	

# Evaluate classifiers using N-fold cross validation
# Input: classifier (one of global vars), features and labels for each text
def evaluate (classifier, filenames, labels, numberOfFolds=5):
	skf = StratifiedKFold(labels, numberOfFolds)
	confusion_matrices = []
	numCorrect = 0
	numIncorrect = 0
	
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
		correct = np.count_nonzero(np.array(predictedLabels)==np.array(actualLabels))
		numCorrect += correct
		numIncorrect += len(predictedLabels) - correct
		# Produce confusion matrix
		cf_mat = confusion_matrix(actualLabels, predictedLabels)
		confusion_matrices.append(cf_mat)
	# Print confusion matrix with average values
	confusion_matrices_numpy = np.array(confusion_matrices)
	mean_cf_mat = np.mean(confusion_matrices_numpy, axis=0)
	mean_cf_mat = mean_cf_mat/mean_cf_mat.sum(axis=1)[:,None]
	print mean_cf_mat
	# Print accuracy
	accuracy = numCorrect/float(numCorrect+numIncorrect)
	print 'Accuracy: ' + str(accuracy)
	
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
	#displayAvgFeatures (filenames, labels)
	
	print 'Testing Logistic Regression'
	#print getFeatures(filenames[0])
	evaluate (logRegClassifier, filenames, labels)
	
	print 'Testing SVM'
	evaluate (svmClassifier, filenames, labels)
	
	
	
