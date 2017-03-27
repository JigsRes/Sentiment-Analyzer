import sys
import getopt
import os
import math
import operator
import random


class Perceptron:
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
        """Perceptron initialization"""
        # in case you found removing stop words helps.
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
        self.weightVector = {}
        self.iterations = 10

    #############################################################################
    # TODO TODO TODO TODO TODO
    # Implement the Perceptron classifier with
    # the best set of features you found through your experiments with Naive Bayes.

    def classify(self, words):
        words = self.filterStopWords(words)
        sum = 0.0
        testWords = {}
        guess = ''
        for token in words:
            testWords[token] = testWords.get(token, 0) + 1
        for token, count in testWords.items():
            # print(token, count , self.weightVector.get(token, 0))
            sum += count * self.weightVector.get(token, 0)
        # print sum

        guess = 'pos' if sum > 0 else 'neg'
        return guess

    def addExample(self, klass, words):
        words = self.filterStopWords(words)
        if klass == 'pos':
            self.positiveWordCount[self.posClassCount] = {}
        else:
            self.negativeWordCount[self.negClassCount] = {}

        for word in words:

            if klass == 'pos':
                self.positiveWordCount[self.posClassCount][word] = self.positiveWordCount[self.posClassCount].get(word,
                                                                                                                  0) + 1
            elif klass == 'neg':
                self.negativeWordCount[self.negClassCount][word] = self.negativeWordCount[self.negClassCount].get(word,
                                                                                                                  0) + 1
            self.vocabulary.add(word)
        if klass == 'pos':
            self.posClassCount += 1
        else:
            self.negClassCount += 1

    def trainPerceptron(self):
        size = len(self.negativeWordCount.keys()) if len(self.negativeWordCount.keys()) < len(
            self.positiveWordCount.keys()) else  len(self.positiveWordCount.keys())
        for j in range(self.iterations):
            random_list = range(2 * size)
            random.shuffle(random_list)
            for i in random_list:
                choice = 'neg'
                if i // size == 1:
                    choice = 'pos'
                k = i % size
                sum = 0
                # print (choice,"coice", )
                if choice == 'neg':
                    for word, count in self.negativeWordCount[k].items():
                        sum += count * self.weightVector.get(word, 0)
                    if sum >= 0:
                        for word, count in self.negativeWordCount[k].items():
                            self.weightVector[word] = self.weightVector.get(word, 0) - count
                else:
                    sum = 0
                    for word, count in self.positiveWordCount[k].items():
                        sum += count * self.weightVector.get(word, 0)
                    if (sum <= 0):
                        for word, count in self.positiveWordCount[k].items():
                            self.weightVector[word] = self.weightVector.get(word, 0) + count

    def train(self, split, iterations):
        """
    * TODO
    * iterates through data examples
    * TODO
    * use weight averages instead of final iteration weights
    """
        self.iterations = iterations
        for example in split.train:
            words = example.words
            self.addExample(example.klass, words)

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


def test10Fold(args):
    pt = Perceptron()

    iterations = int(args[1])
    splits = pt.crossValidationSplits(args[0])
    avgAccuracy = 0.0
    fold = 0
    for split in splits:
        classifier = Perceptron()
        accuracy = 0.0
        classifier.train(split, iterations)

        classifier.trainPerceptron()

        for example in split.test:
            words = example.words
            guess = classifier.classify(words)
            if example.klass == guess:
                accuracy += 1.0

        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print '[INFO]\tAccuracy: %f' % avgAccuracy


def classifyDir(trainDir, testDir, iter):
    classifier = Perceptron()
    trainSplit = classifier.trainSplit(trainDir)
    iterations = int(iter)
    classifier.train(trainSplit, iterations)
    testSplit = classifier.trainSplit(testDir)
    # testFile = classifier.readFile(testFilePath)
    accuracy = 0.0
    classifier.trainPerceptron()
    for example in testSplit.train:
        words = example.words
        guess = classifier.classify(words)
        if example.klass == guess:
            accuracy += 1.0
    accuracy = accuracy / len(testSplit.train)
    print '[INFO]\tAccuracy: %f' % accuracy


def main():
    (options, args) = getopt.getopt(sys.argv[1:], '')

    if len(args) == 3:
        classifyDir(args[0], args[1], args[2])
    elif len(args) == 2:
        test10Fold(args)


if __name__ == "__main__":
    main()
