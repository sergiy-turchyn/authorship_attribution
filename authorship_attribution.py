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

import matplotlib.pyplot as plt

stemmer = PorterStemmer()
letterPattern = re.compile('[^a-zA-Z0-9 ]+')

# Top N most frequent words/characters go here after the feature 
wordList = []
charList = []

# How many most used words/characters we take from each author
wordsPerAuthor = 200
charsPerAuthor = 200
# How many features to select in feature selection
numBestFeatures = 20

featureSelection = SelectKBest(chi2, k=numBestFeatures)

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
	uppercaseFreq = None # uppercase character frequency
	lowercaseFreq = None # lowercase character frequency
	numericFreq = None # numeric character frequency
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
	result.append('Frequency of uppercase characters')
	result.append('Frequency of lowercase characters')
	result.append('Frequency of numeric characters')
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
	result.append(features.uppercaseFreq)
	result.append(features.lowercaseFreq)
	result.append(features.numericFreq)
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
	# get frequency of uppercase, lowercase, and numeric characters
	numeric = [char for char in raw if char.isdigit()]
	uppercase = [char for char in raw if char.isupper()]
	lowercase = [char for char in raw if char.islower()]
	features.numericFreq = len(numeric)/float(len(raw))
	features.uppercaseFreq = len(uppercase)/float(len(raw))
	features.lowercaseFreq = len(lowercase)/float(len(raw))
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
def evaluate (classifier, filenames, labels, numberOfFolds=5, printMatrix=True):
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
	if printMatrix:
		print mean_cf_mat
	# Print accuracy
	accuracy = numCorrect/float(numCorrect+numIncorrect)
	print 'Accuracy: ' + str(accuracy)
	#print ''
	return accuracy
		
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
	svmClassifier = SVC(C=100000)
	knnClassifier = KNeighborsClassifier(weights='distance', n_neighbors=15)
	dTreeClassifier = DecisionTreeClassifier()
	rForestClassifier = RandomForestClassifier()
	nBayesClassifier = GaussianNB()
	# Produce the results
	#displayAvgFeatures (filenames, labels)
	#selectFeatures(filenames, labels)
	print ''
	#showBestFeatures(filenames, labels)
	print ''
	
	data = [[],[],[],[],[]]
	
	#global numBestFeatures
	#global featureSelection
	#print 'numBestFeatures = ' + str(numBestFeatures)
	#showBestFeatures(filenames, labels)
	'''
	numBestFeatures = 100
	featureSelection = SelectKBest(chi2, k=numBestFeatures)
	print 'Testing SVM'
	data[0].append(evaluate (svmClassifier, filenames, labels))
	
	numBestFeatures = 37
	featureSelection = SelectKBest(chi2, k=numBestFeatures)
	print 'Testing KNN'
	data[1].append(evaluate (knnClassifier, filenames, labels))
	
	numBestFeatures = 95
	featureSelection = SelectKBest(chi2, k=numBestFeatures)
	print 'Testing decision tree'
	data[2].append(evaluate (dTreeClassifier, filenames, labels))
	
	numBestFeatures = 94
	featureSelection = SelectKBest(chi2, k=numBestFeatures)
	print 'Testing random forest'
	data[3].append(evaluate (rForestClassifier, filenames, labels))
	
	numBestFeatures = 67
	featureSelection = SelectKBest(chi2, k=numBestFeatures)
	print 'Testing naive Bayes'
	data[4].append(evaluate (nBayesClassifier, filenames, labels))
	'''
	print data
	
	data[0] = [0.391304347826087, 0.4855072463768116, 0.5217391304347826, 0.5869565217391305, 0.6231884057971014, 0.6884057971014492, 0.7101449275362319, 0.6811594202898551, 0.7028985507246377, 0.7318840579710145, 0.7391304347826086, 0.7536231884057971, 0.7608695652173914, 0.7536231884057971, 0.7608695652173914, 0.7681159420289855, 0.7608695652173914, 0.782608695652174, 0.7898550724637681, 0.7898550724637681, 0.7971014492753623, 0.7971014492753623, 0.7971014492753623, 0.8043478260869565, 0.7971014492753623, 0.7971014492753623, 0.7681159420289855, 0.7971014492753623, 0.7898550724637681, 0.7898550724637681, 0.7898550724637681, 0.782608695652174, 0.7753623188405797, 0.7753623188405797, 0.782608695652174, 0.782608695652174, 0.7753623188405797, 0.7753623188405797, 0.782608695652174, 0.7898550724637681, 0.782608695652174, 0.7898550724637681, 0.782608695652174, 0.782608695652174, 0.782608695652174, 0.8115942028985508, 0.7971014492753623, 0.7971014492753623, 0.782608695652174, 0.7898550724637681, 0.7971014492753623, 0.8043478260869565, 0.8043478260869565, 0.8043478260869565, 0.7971014492753623, 0.7898550724637681, 0.7971014492753623, 0.8043478260869565, 0.7971014492753623, 0.7971014492753623, 0.8043478260869565, 0.8115942028985508, 0.8115942028985508, 0.8115942028985508, 0.8260869565217391, 0.8260869565217391, 0.8405797101449275, 0.8333333333333334, 0.8405797101449275, 0.8405797101449275, 0.8478260869565217, 0.8478260869565217, 0.8478260869565217, 0.8478260869565217, 0.855072463768116, 0.855072463768116, 0.8478260869565217, 0.8478260869565217, 0.855072463768116, 0.8623188405797102, 0.8405797101449275, 0.8405797101449275, 0.8405797101449275, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.8623188405797102, 0.8695652173913043]
	data[1] = [0.4420289855072464, 0.4927536231884058, 0.5434782608695652, 0.572463768115942, 0.5652173913043478, 0.6014492753623188, 0.6014492753623188, 0.6159420289855072, 0.6304347826086957, 0.6304347826086957, 0.6231884057971014, 0.6159420289855072, 0.6159420289855072, 0.6159420289855072, 0.6159420289855072, 0.6159420289855072, 0.6304347826086957, 0.6231884057971014, 0.6304347826086957, 0.6304347826086957, 0.6304347826086957, 0.6304347826086957, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.644927536231884, 0.644927536231884, 0.644927536231884, 0.644927536231884, 0.644927536231884, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.6376811594202898, 0.6304347826086957, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6304347826086957, 0.6304347826086957, 0.6304347826086957, 0.6304347826086957, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6304347826086957, 0.6304347826086957, 0.6304347826086957, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014, 0.6231884057971014]
	data[2] = [0.427536231884058, 0.463768115942029, 0.5072463768115942, 0.4927536231884058, 0.5434782608695652, 0.5869565217391305, 0.6159420289855072, 0.6521739130434783, 0.6376811594202898, 0.6304347826086957, 0.6956521739130435, 0.7101449275362319, 0.7028985507246377, 0.7463768115942029, 0.7753623188405797, 0.7971014492753623, 0.8043478260869565, 0.8115942028985508, 0.7971014492753623, 0.8188405797101449, 0.8115942028985508, 0.8043478260869565, 0.8260869565217391, 0.7608695652173914, 0.7753623188405797, 0.7536231884057971, 0.782608695652174, 0.782608695652174, 0.7391304347826086, 0.7608695652173914, 0.7463768115942029, 0.7391304347826086, 0.7536231884057971, 0.7681159420289855, 0.7753623188405797, 0.7608695652173914, 0.782608695652174, 0.782608695652174, 0.7463768115942029, 0.7608695652173914, 0.7681159420289855, 0.7608695652173914, 0.7608695652173914, 0.7753623188405797, 0.7608695652173914, 0.7681159420289855, 0.7681159420289855, 0.7898550724637681, 0.782608695652174, 0.782608695652174, 0.7753623188405797, 0.7971014492753623, 0.7608695652173914, 0.8188405797101449, 0.8043478260869565, 0.782608695652174, 0.7753623188405797, 0.7971014492753623, 0.7753623188405797, 0.7753623188405797, 0.7463768115942029, 0.8188405797101449, 0.7898550724637681, 0.782608695652174, 0.782608695652174, 0.7971014492753623, 0.8260869565217391, 0.8043478260869565, 0.7753623188405797, 0.8115942028985508, 0.7971014492753623, 0.782608695652174, 0.8188405797101449, 0.8188405797101449, 0.7753623188405797, 0.8260869565217391, 0.8043478260869565, 0.8478260869565217, 0.855072463768116, 0.8405797101449275, 0.8115942028985508, 0.8188405797101449, 0.7898550724637681, 0.8043478260869565, 0.8043478260869565, 0.8115942028985508, 0.8188405797101449, 0.855072463768116, 0.855072463768116, 0.8478260869565217, 0.8478260869565217, 0.8695652173913043, 0.8478260869565217, 0.8478260869565217, 0.8840579710144928, 0.855072463768116, 0.8768115942028986, 0.8623188405797102, 0.8478260869565217, 0.8405797101449275]
	data[3] = [0.42028985507246375, 0.427536231884058, 0.572463768115942, 0.5797101449275363, 0.6521739130434783, 0.6521739130434783, 0.6811594202898551, 0.6521739130434783, 0.6521739130434783, 0.6521739130434783, 0.6521739130434783, 0.644927536231884, 0.6956521739130435, 0.7463768115942029, 0.782608695652174, 0.7681159420289855, 0.7463768115942029, 0.8478260869565217, 0.782608695652174, 0.8478260869565217, 0.7898550724637681, 0.8333333333333334, 0.7898550724637681, 0.8260869565217391, 0.782608695652174, 0.7898550724637681, 0.7971014492753623, 0.7898550724637681, 0.7898550724637681, 0.7971014492753623, 0.7681159420289855, 0.8043478260869565, 0.8115942028985508, 0.8043478260869565, 0.782608695652174, 0.8188405797101449, 0.8115942028985508, 0.8478260869565217, 0.782608695652174, 0.8260869565217391, 0.7971014492753623, 0.782608695652174, 0.8115942028985508, 0.782608695652174, 0.7898550724637681, 0.8188405797101449, 0.782608695652174, 0.8188405797101449, 0.8333333333333334, 0.8623188405797102, 0.7971014492753623, 0.7971014492753623, 0.7971014492753623, 0.8405797101449275, 0.7971014492753623, 0.8043478260869565, 0.7971014492753623, 0.8478260869565217, 0.8188405797101449, 0.8623188405797102, 0.8405797101449275, 0.8260869565217391, 0.8478260869565217, 0.8333333333333334, 0.855072463768116, 0.8115942028985508, 0.8405797101449275, 0.8695652173913043, 0.8695652173913043, 0.8623188405797102, 0.8333333333333334, 0.8478260869565217, 0.8405797101449275, 0.8623188405797102, 0.8768115942028986, 0.7971014492753623, 0.8115942028985508, 0.8478260869565217, 0.8115942028985508, 0.8333333333333334, 0.8405797101449275, 0.855072463768116, 0.8695652173913043, 0.8405797101449275, 0.8913043478260869, 0.8115942028985508, 0.8768115942028986, 0.8840579710144928, 0.8913043478260869, 0.8840579710144928, 0.855072463768116, 0.8405797101449275, 0.8478260869565217, 0.9202898550724637, 0.8405797101449275, 0.8768115942028986, 0.8478260869565217, 0.8913043478260869, 0.8478260869565217, 0.8260869565217391]
	data[4] = [0.4420289855072464, 0.5362318840579711, 0.6014492753623188, 0.5869565217391305, 0.6014492753623188, 0.644927536231884, 0.6666666666666666, 0.7028985507246377, 0.6811594202898551, 0.6956521739130435, 0.7028985507246377, 0.7318840579710145, 0.7753623188405797, 0.782608695652174, 0.8115942028985508, 0.7898550724637681, 0.7898550724637681, 0.7898550724637681, 0.8115942028985508, 0.8043478260869565, 0.8115942028985508, 0.7898550724637681, 0.782608695652174, 0.782608695652174, 0.7753623188405797, 0.782608695652174, 0.7898550724637681, 0.782608695652174, 0.7971014492753623, 0.8115942028985508, 0.8043478260869565, 0.8043478260869565, 0.8115942028985508, 0.8043478260869565, 0.782608695652174, 0.7898550724637681, 0.7898550724637681, 0.7971014492753623, 0.8043478260869565, 0.7971014492753623, 0.7898550724637681, 0.7971014492753623, 0.7971014492753623, 0.782608695652174, 0.7971014492753623, 0.8043478260869565, 0.8115942028985508, 0.7898550724637681, 0.7753623188405797, 0.7681159420289855, 0.7608695652173914, 0.7608695652173914, 0.782608695652174, 0.7898550724637681, 0.7898550724637681, 0.7971014492753623, 0.7971014492753623, 0.8115942028985508, 0.8115942028985508, 0.8115942028985508, 0.8115942028985508, 0.8115942028985508, 0.8188405797101449, 0.8260869565217391, 0.8260869565217391, 0.8260869565217391, 0.8333333333333334, 0.8260869565217391, 0.8188405797101449, 0.8115942028985508, 0.8115942028985508, 0.8115942028985508, 0.8043478260869565, 0.8043478260869565, 0.8115942028985508, 0.8043478260869565, 0.8043478260869565, 0.8043478260869565, 0.7971014492753623, 0.8043478260869565, 0.8115942028985508, 0.8043478260869565, 0.7971014492753623, 0.8115942028985508, 0.8115942028985508, 0.8115942028985508, 0.8188405797101449, 0.8115942028985508, 0.8115942028985508, 0.8043478260869565, 0.8043478260869565, 0.8043478260869565, 0.7971014492753623, 0.8115942028985508, 0.8115942028985508, 0.8115942028985508, 0.8115942028985508, 0.8115942028985508, 0.8043478260869565, 0.8115942028985508]
	names = ['SVM', 'kNN', 'Decision Tree', 'Random Forest', 'Naive Bayes']
	for i in range(5):
		plt.plot(range(1,len(data[i])+1), data[i], label=names[i])
	plt.ylabel('accuracy')
	plt.xlabel('number of features')
	plt.legend(bbox_to_anchor=(0.5, 0.1), loc=3, borderaxespad=0.)
	plt.show()

	raw_input("Press Enter to continue...")
	
	
	
