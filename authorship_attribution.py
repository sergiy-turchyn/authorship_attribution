import os
import random
import re

import numpy as np

from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

stemmer = PorterStemmer()
letterPattern = re.compile('[^a-zA-Z0-9 ]+')

# Top N most frequent words/characters go here after the feature 
wordList = []
charList = []

# How many most used words/characters we take from each author
wordsPerAuthor = 100
charsPerAuthor = 100
# How many features to select in feature selection
numBestFeatures = 50

featureSelection = SelectKBest(chi2, k=numBestFeatures)

'''
Features to add:
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
	charFrequencies = {}
	author = None # cheating feature
	# any additional features go here
	
def getFeatureName(number):
	result = []
	result.append('Words occurring once')
	result.append('Words occurring twice')
	result.append('Lexical richness (distinct words used)')
	result.append('Average word length')
	result.append('Median word length')
	result.append('Standard deviation of word lengths')
	#result.append('Author')
	for word in wordList:
		result.append('Frequency of word "'+word+'"')
	for char in charList:
		result.append('Frequency of character "'+char+'"')
	return result[number]
		
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
	#result.append(features.author)
	
	for word in wordList:
		if (word in features.wordFrequencies.keys()):
			result.append(features.wordFrequencies[word])
		else:
			result.append(0.0000001)
			
	for char in charList:
		if (char in features.charFrequencies.keys()):
			result.append(features.charFrequencies[char])
		else:
			result.append(0.0000001)
	return result
	
# Select features to be used given files
# Goal: set global variables to proper values
# For example, find N most used words/characters
def selectFeatures(filenames, labels):
	# Select most used words and characters from each author
	for author in set(labels):
		authorFilenames = [f for f in filenames if os.path.basename(f).split('-')[0]==author]
		wordFreq = FreqDist()
		charFreq = FreqDist()
		for filename in authorFilenames:
			with open(filename) as fp:
				raw = fp.read().decode('utf8')
			# get most used chars
			charDistText = FreqDist(raw)
			for char in charDistText.keys():
				if char in charFreq.keys():
					charFreq[char] = charFreq[char] + charDistText[char]
				else:
					charFreq[char] = charDistText[char]
			# remove non-letters
			text = letterPattern.sub('', raw)
			# stem
			text = [stemmer.stem(word) for word in text.lower().split()]
			# get most used words
			freqDistText = FreqDist(text)
			for word in freqDistText.keys():
				if word in wordFreq.keys():
					wordFreq[word] = wordFreq[word] + freqDistText[word]
				else:
					wordFreq[word] = freqDistText[word]
		mostCommonWords = wordFreq.most_common(wordsPerAuthor)
		for (word, count) in mostCommonWords:
			if word not in wordList:
				wordList.append(word)
		mostCommonChars = charFreq.most_common(charsPerAuthor)
		for (char, count) in mostCommonChars:
			if char not in charList:
				charList.append(char)
		
# Print the most useful features for given data
def showBestFeatures(filenames, labels):
	selectFeatures(filenames, labels)
	features = [getFeatures(fn) for fn in filenames]
	featureSelection.fit(features, labels)
	featuresSelected = featureSelection.get_support(indices=True)
	print str(numBestFeatures) + ' best features:'
	for index in featuresSelected:
		print getFeatureName(index)
	
		
# Produces a list of features for a file
def getFeatures(filename):
	features = TextFeatures()
	features.author = float(os.path.basename(filename).split('-')[0])
	# read the file
	with open(filename) as fp:
		raw = fp.read().decode('utf8')
	# print len(set(raw.split())) # how many different words before processing
	# remove non-letters
	text = letterPattern.sub('', raw)
	# get average/median/std of word lengths
	wordLengths = [len(word) for word in text.lower().split()]
	avgWordLength = np.average(wordLengths)
	medianWordLength = np.median(wordLengths)
	stdWordLength = np.std(wordLengths)
	features.avgWordLength = avgWordLength
	features.medianWordLength = medianWordLength
	features.stdWordLength = stdWordLength
	# stem
	text = [stemmer.stem(word) for word in text.split()]
	# calculate frequency data
	wordFreqDist = FreqDist(text)
	for word in wordList:
		if word in wordFreqDist.keys():
			features.wordFrequencies[word] = wordFreqDist[word]/float(len(text))
		else:
			features.wordFrequencies[word] = 0.0000001
	charFreqDist = FreqDist(raw)
	for char in charList:
		if char in charFreqDist.keys():
			features.charFrequencies[char] = charFreqDist[char]/float(len(raw))
		else:
			features.charFrequencies[char] = 0.0000001
	# calculate words occurring 1-2 times
	wordsOccurringOnce = [w for w in wordFreqDist.keys() if wordFreqDist[w]==1]
	wordsOccurringTwice = [w for w in wordFreqDist.keys() if wordFreqDist[w]==2]
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
		#print 'Training on ' + str(len(train)) + ' samples. Testing on ' + str(len(test)) + ' samples.'
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
		# Feature selection
		trainingData = featureSelection.fit_transform(trainingData, trainingLabels)
		testingData = featureSelection.transform(testingData)
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
	print ''
	
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
	svmClassifier = SVC(C=10)
	knnClassifier = KNeighborsClassifier(weights='distance')
	dTreeClassifier = DecisionTreeClassifier()
	rForestClassifier = RandomForestClassifier()
	nBayesClassifier = GaussianNB()
	# Produce the results
	#displayAvgFeatures (filenames, labels)
	#selectFeatures(filenames, labels)
	print ''
	showBestFeatures(filenames, labels)
	print ''

	print 'Testing SVM'
	evaluate (svmClassifier, filenames, labels)
	
	print 'Testing KNN'
	evaluate (knnClassifier, filenames, labels)
	
	print 'Testing decision tree'
	evaluate (dTreeClassifier, filenames, labels)
	
	print 'Testing random forest'
	evaluate (rForestClassifier, filenames, labels)
	
	print 'Testing naive Bayes'
	evaluate (nBayesClassifier, filenames, labels)
	
	
	
	
