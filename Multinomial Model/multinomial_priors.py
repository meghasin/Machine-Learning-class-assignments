import os
import csv
import sys
#import numpy as np
#import matplotlib.pyplot as plt
import math
import random
import time
#import scipy.integrate as integrate
#import scipy.linalg
import subprocess


#Call bagofwords.py to get our parsing started:
subprocess.call("python bagofwords.py clintontrump.tweets.train clintontrump.tweets.dev clintontrump.tweets.test")

#Do the rest of the parsing:
train = []
dev = []
test = []
vocab = []
vocabID = []
trainLabels = []
devLabels = []
count = 0


with open('clintontrump.vocabulary', 'r') as csvfile:
    raw_data = csv.reader(csvfile, delimiter='\t', quotechar=None)
    for row in raw_data:
        for i in row:
            vocab.append(str(i))
            count += 1

#for i in range(0, count):
#    print vocab[i]

count = 0

with open('clintontrump.bagofwords.train', 'r') as csvfile:
    raw_data = csv.reader(csvfile, delimiter='\n', quotechar=None)
    for row in raw_data:
        for i in row:
            train.append(str(i))
            count += 1

#for i in range(0, count):
#    print train[i]

count = 0

with open('clintontrump.bagofwords.test', 'r') as csvfile:
    raw_data = csv.reader(csvfile, delimiter='\n', quotechar=None)
    for row in raw_data:
        for i in row:
            test.append(str(i))
            count += 1

#for i in range(0, count):
#    print test[i]

count = 0

with open('clintontrump.bagofwords.dev', 'r') as csvfile:
    raw_data = csv.reader(csvfile, delimiter='\n', quotechar=None)
    for row in raw_data:
        for i in row:
            dev.append(str(i))
            count += 1

#for i in range(0, count):
#    print dev[i]

count = 0

with open('clintontrump.labels.dev', 'r') as csvfile:
    raw_data = csv.reader(csvfile, delimiter='\n', quotechar=None)
    for row in raw_data:
        for i in row:
            devLabels.append(str(i))
            count += 1

#for i in range(0, count):
#    print devLabels[i]

count = 0

with open('clintontrump.labels.train', 'r') as csvfile:
    raw_data = csv.reader(csvfile, delimiter='\n', quotechar=None)
    for row in raw_data:
        for i in row:
            trainLabels.append(str(i))
            count += 1

#for i in range(0, count):
#    print trainLabels[i]

"""
Part 1: Naive Bayes classifier, Multinomial model.
"""
clintwordCount = {}
#Find the probabilities for each word being associated with each candidate:
def part2TrainingFunct(vocab, training, trainLabels, trumpwordCount, clintwordCount,trumpVocabtrueCount, clinVocabtrueCount):
    prob = 0 #//The probability of the word being in a the candidate's post
    wordCount = 0 #Frequency of words
    trumptrueCount=0  #Number of trump words
    clintrueCount = 0 #Number of clinton words
    """
    print "i = "
    print i
    print "len(vocab)= "
    print len(vocab)
    """
    clintprob=0
    trumpprob = 0
  
    for line, label in zip(training, trainLabels):
        #print line
        if label == "HillaryClinton":                                                                                         
            for word in line.split():
                clintwordCount[word]=clintwordCount.get(word,0)
                clintwordCount[word]+=1
                clintrueCount+=1
                #print clintwordCount[word]
        else:
            for word in line.split():
                trumpwordCount[word]=trumpwordCount.get(word,0)
                trumpwordCount[word]+=1
                trumptrueCount+=1
 
    #vocabClintonProb.append(clintwordCount)
    #print trumpwordCount
    trumpVocabtrueCount.append(trumptrueCount)
    clinVocabtrueCount.append(clintrueCount)

#p(x|y) = PI notation from i=1 to d of p(x sub i|y) -> log(p(x|y)) = PI notation from i=1 to d of log(p(x sub i|y)):
#Apply alpha Laplace Smoothing -> P() = PI notation from i=1 to d of (c(w) + alpha)/(C(x) + alpha*(|V|+1)):
def part2Classification(vocab, training,trainLabels, trumpwordCount, clintwordCount, clinVocabtrueCount, trumpVocabtrueCount, testTrumpProbs, testClintonProbs):
    
    #print clintwordCount;
    print "Training done!"
    alpha = 0.00001
    while alpha < 10:
        testTrumpProbs=[]
        testClintonProbs=[]
        for i in range(0, len(training)):
            trumpProb = 0.0
            clintProb=0.0

            for j in range(1, len(training[i].split())):
                wordFreqTrump=trumpwordCount.get(training[i].split()[j],0)
                wordFreqClint=clintwordCount.get(training[i].split()[j],0)
                #print wordFreqTrump;
                logvalTrump= (math.log(float(wordFreqTrump + alpha)/float(trumpVocabtrueCount[0]+alpha*(len(vocab)+1))))  
                logvalClint= (math.log(float(wordFreqClint + alpha)/float(clinVocabtrueCount[0]+alpha*(len(vocab)+1))))
                #print logval
                if(math.log(float(wordFreqTrump + alpha)/float(trumpVocabtrueCount[0]+alpha*(len(vocab)+1))) != 0):
                    trumpProb = trumpProb + logvalTrump#(c(w) + alpha)/(C(x) + alpha*(|V|+1))

                if(math.log(float(wordFreqClint + alpha)/float(clinVocabtrueCount[0]+alpha*(len(vocab)+1))) != 0):
                    clintProb = clintProb + logvalClint#(c(w) + alpha)/(C(x) + alpha*(|V|+1))

            if trumpProb >= clintProb:
                testClintonProbs.append(0)
                testTrumpProbs.append(1)
            else:
                testClintonProbs.append(1)
                testTrumpProbs.append(0)
        alpha=alpha*10
        print "Classification done for alpha!"+str(alpha)
        #print testTrumpProbs, testClintonProbs
        #Training and testing accuracy:
        trumpTrainAcc = part1Accuracy(training, testTrumpProbs, trainLabels, 1)
        print "Accuracy rating 1 done!"
        clintonTrainAcc = part1Accuracy(training, testClintonProbs, trainLabels, 0)
        print "Accuracy rating 2 done!"
        #trumpDevAcc = part1Accuracy(testing, trumpTestingResults, devLabels, 1)
        print "Accuracy rating 3 done!"
        #clintonDevAcc = part1Accuracy(testing, clintonTestingResults, devLabels, 0)
    
        print "Accuracy rating 4 done!"

        #Output:
    
        print "trumpTrainAcc = "
        print trumpTrainAcc
        print "clintonTrainAcc = "
        print clintonTrainAcc
        
#Accuracy = the number of correct predictions / total predictions:
def part1Accuracy(testing, testProbs, testLabels, candidate):
    numCorrect = 0
    
    for i in range(len(testProbs)):
        if candidate == 0:
            if testLabels[i][len(testLabels[i])-1] == "n" and testProbs[i] == 1: #Clinton
                numCorrect = numCorrect + 1
            else:
                if testLabels[i][len(testLabels[i])-1] == "p" and testProbs[i] == 0: #Trump
                    numCorrect = numCorrect + 1
        else:
            if testLabels[i][len(testLabels[i])-1] == "p" and testProbs[i] == 1: #Clinton
                numCorrect = numCorrect + 1
            else:
                if testLabels[i][len(testLabels[i])-1] == "n" and testProbs[i] == 0: #Trump
                    numCorrect = numCorrect + 1
    return float(numCorrect) / len(testProbs)

def part1Classifier(vocab, training, testing, trainLabels, devLabels):
    trainingTrumpVocabProbs = {}
    trainingClintonVocabProbs = {}
    testTrumpVocabProbs = []
    testClintonVocabProbs = []
    trumpTrainingResults = []
    clintonTrainingResults = []
    trumpTestingResults = []
    clintonTestingResults = []
    trumpVocabtrueCount=[]
    clinVocabtrueCount=[]

    #l = math.log1p(5)
    
    #Training phase:
    
    #part1TrainingFunct(vocab, training, trainLabels, trainingClintonVocabProbs, 0)

    # print trainingTrumpVocabProbs[4]
    # print trainingTrumpVocabProbs[5]
    # print trainingClintonVocabProbs[4]
    # print trainingClintonVocabProbs[5]
    part2TrainingFunct(vocab, train, trainLabels, trainingTrumpVocabProbs, trainingClintonVocabProbs, trumpVocabtrueCount, clinVocabtrueCount)
    
    
    #Training and testing classification phase:
    
    part2Classification(vocab, training, trainLabels,trainingTrumpVocabProbs, trainingClintonVocabProbs, clinVocabtrueCount, trumpVocabtrueCount, trumpTrainingResults, clintonTrainingResults)
    print "Classification 1 done!"
	#part1Classification(vocab, testing, testTrumpVocabProbs, testClintonVocabProbs, trumpTestingResults, clintonTestingResults)

    # print trumpTrainingResults[4]
    # print trumpTrainingResults[5]
    # print clintonTrainingResults[4]
    # print clintonTrainingResults[5]
    
    print "Classification 2 done!"

    
    #print "trumpTestAcc = "
    #print trumpDevAcc
    #print "clintonTestAcc = "
    #printclintonDevAcc



"""Program Start"""
part1Classifier(vocab, train, test, trainLabels, devLabels)