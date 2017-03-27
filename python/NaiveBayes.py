import sys
import getopt
import os
import math
import operator
import collections


class NaiveBayes:
    class TrainSplit:
        """Represents a set of training/testing data. self.train is a list of Examples, as is self.test.
        """

        def __init__(self):
            self.train = []
            self.test = []

    class Example:
        """Represents a document with a label. klass is 'pos' or 'neg' by convention.
           words is a list of strings.
        """

        def __init__(self):
            self.klass = ''
            self.words = []

    def __init__(self):
        """NaiveBayes initialization"""
        self.FILTER_STOP_WORDS = False
        self.BOOLEAN_NB = False
        self.stopList = set(self.readFile('../data/english.stop'))
        self.numFolds = 10
        self.posDict = {}
        self.negDict = {}
        self.vocabulary = set()
        self.positiveWordCount = dict()
        self.negativeWordCount = dict()
        self.positiveCount = 0
        self.negativeCount = 0
        self.zeroPosProb = 0.0
        self.zeroNegProb = 0.0
        self.posClassCount = 0
        self.negClassCount = 0

    #############################################################################
    # TODO TODO TODO TODO TODO
    # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
    # Boolean (Binarized) features.
    # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
    # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
    # that relies on feature counts.
    #
    #
    # If any one of the FILTER_STOP_WORDS and BOOLEAN_NB flags is on, the
    # other one is meant to be off.

    def classify(self, words):
        probNegative = 0.0
        probPositive = 0.0
        guess = ''
        if self.FILTER_STOP_WORDS:
            words = self.filterStopWords(words)
        for token in words:
            probNegative += math.log10(self.negDict.get(token, self.zeroNegProb))
            probPositive += math.log10(self.posDict.get(token, self.zeroPosProb))
        probPositive += math.log10((self.posClassCount / float(self.posClassCount + self.negClassCount)))
        probNegative += math.log10((self.negClassCount / float(self.posClassCount + self.negClassCount)))
        guess = 'pos' if probPositive > probNegative else 'neg'
        return guess

    def addExample(self, klass, words):
        if self.FILTER_STOP_WORDS:
            words = self.filterStopWords(words)
        if klass == 'pos':
            self.posClassCount += 1
        else:
            self.negClassCount += 1

        if self.BOOLEAN_NB:
            posWordBooleanCountDoc = {}
            negWordBooleanCountDoc = {}
            for word in words:
                if klass == 'pos':
                    posWordBooleanCountDoc[word] = True
                elif klass == 'neg':
                    negWordBooleanCountDoc[word] = True
                self.vocabulary.add(word)
            if klass == 'pos':
                for word, bool in posWordBooleanCountDoc.items():
                    self.positiveWordCount[word] = self.positiveWordCount.get(word, 0) + 1
            else:
                for word, bool in negWordBooleanCountDoc.items():
                    self.negativeWordCount[word] = self.negativeWordCount.get(word, 0) + 1
        else:
            for word in words:
                if klass == 'pos':
                    self.positiveWordCount[word] = self.positiveWordCount.get(word, 0) + 1
                    self.positiveCount += 1
                elif klass == 'neg':
                    self.negativeWordCount[word] = self.negativeWordCount.get(word, 0) + 1
                    self.negativeCount += 1
                self.vocabulary.add(word)

    def prepareDictionaries(self):
        vocab_size = len(self.vocabulary)
        if self.BOOLEAN_NB:
            for word, count in self.positiveWordCount.items():
                self.positiveCount += self.positiveWordCount[word]
            for word, count in self.negativeWordCount.items():
                self.negativeCount += self.negativeWordCount[word]

            for token in self.vocabulary:
                self.posDict[token] = float(
                    (self.positiveWordCount.get(token, 0) + 1) / float((self.positiveCount + vocab_size + 1)))
                self.negDict[token] = float(
                    (self.negativeWordCount.get(token, 0) + 1) / float((self.negativeCount + vocab_size + 1)))
            self.zeroNegProb = float(1 / float((self.negativeCount + vocab_size + 1)))
            self.zeroPosProb = float(1 / float((self.positiveCount + vocab_size + 1)))


        else:
            for token in self.vocabulary:
                self.posDict[token] = float(
                    (self.positiveWordCount.get(token, 0) + 1) / float((self.positiveCount + vocab_size + 1)))
                self.negDict[token] = float(
                    (self.negativeWordCount.get(token, 0) + 1) / float((self.negativeCount + vocab_size + 1)))
            self.zeroNegProb = float(1 / float((self.negativeCount + vocab_size + 1)))
            self.zeroPosProb = float(1 / float((self.positiveCount + vocab_size + 1)))

    # END TODO (Modify code beyond here with caution)
    #############################################################################


    def readFile(self, fileName):
        """
         * Code for reading a file.  you probably don't want to modify anything here,
         * unless you don't like the way we segment files.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        result = self.segmentWords('\n'.join(contents))
        return result

    def segmentWords(self, s):
        """
         * Splits lines on whitespace for file reading
        """
        return s.split()

    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        for fileName in posTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            example.klass = 'pos'
            split.train.append(example)
        for fileName in negTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            example.klass = 'neg'
            split.train.append(example)
        return split

    def train(self, split):
        for example in split.train:
            words = example.words
            if self.FILTER_STOP_WORDS:
                words = self.filterStopWords(words)
            self.addExample(example.klass, words)

    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        # for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                example.klass = 'pos'
                if fileName[2] == str(fold):
                    split.test.append(example)
                else:
                    split.train.append(example)
            for fileName in negTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                example.klass = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(example)
                else:
                    split.train.append(example)
            splits.append(split)
        return splits

    def filterStopWords(self, words):
        """Filters stop words."""
        filtered = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                filtered.append(word)
        return filtered


def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB):
    nb = NaiveBayes()
    splits = nb.crossValidationSplits(args[0])
    avgAccuracy = 0.0
    fold = 0
    for split in splits:
        classifier = NaiveBayes()
        classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
        classifier.BOOLEAN_NB = BOOLEAN_NB
        accuracy = 0.0
        for example in split.train:
            words = example.words
            classifier.addExample(example.klass, words)

        classifier.prepareDictionaries()

        for example in split.test:
            words = example.words
            guess = classifier.classify(words)
            if example.klass == guess:
                accuracy += 1.0

        print(accuracy, len(split.test))
        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print '[INFO]\tAccuracy: %f' % avgAccuracy


def classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB, trainDir, testDir):
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    trainSplit = classifier.trainSplit(trainDir)
    classifier.train(trainSplit)
    testSplit = classifier.trainSplit(testDir)
    accuracy = 0.0
    classifier.prepareDictionaries()
    for example in testSplit.train:
        words = example.words
        guess = classifier.classify(words)
        if example.klass == guess:
            accuracy += 1.0
    accuracy = accuracy / len(testSplit.train)
    print '[INFO]\tAccuracy: %f' % accuracy


def main():
    FILTER_STOP_WORDS = False
    BOOLEAN_NB = False
    (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
    if ('-f', '') in options:
        FILTER_STOP_WORDS = True
    if ('-b', '') in options:
        BOOLEAN_NB = True

    if len(args) == 2:
        classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB, args[0], args[1])
    elif len(args) == 1:
        test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB)


if __name__ == "__main__":
    main()
