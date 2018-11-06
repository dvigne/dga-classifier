#!/usr/bin/python
from __future__ import division

import numpy as np
from tabulate import tabulate
import scipy.stats as stats
from progress.bar import Bar

# Numpy Configuration
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold='nan')

fileName = "data/sampledga.csv"
csvArray = [None]
progBar = Bar('Load CSV', max=5)
def readCSV():
    global csvArray
    file = open(fileName, "r")
    csvArray = file.read().replace('\n', ',').replace("\"", "").split(',')
    csvArray.pop()
    csvArray = np.delete(np.array(csvArray).reshape(-1,5),0,0)
    file.close()

def calculateEntropy():
    global csvArray
    entArray = np.zeros(csvArray.shape[0])
    for x in range(0, csvArray.shape[0]):
        charFreq = np.zeros(52)
        for y in range(0, charFreq.shape[0]//2):
            charFreq[y] = csvArray[x][1].count(chr(y+65)) / int(csvArray[x][5])
            charFreq[y+charFreq.shape[0]//2] = csvArray[x][1].count(chr(y+97)) / int(csvArray[x][5])
        entArray[x] = stats.entropy(charFreq)
    csvArray = np.column_stack([csvArray, entArray])

def calculateLength():
    global csvArray
    lenArray = []
    for x in range(0, csvArray.shape[0]):
        lenArray.append(len(csvArray[x][1]))
    csvArray = np.column_stack([csvArray, lenArray])

def calculateVCRatio():
    global csvArray
    vcArray = []
    for x in range(0, csvArray.shape[0]):
        vowelCount = 0
        vowelCount += csvArray[x][1].count('a')
        vowelCount += csvArray[x][1].count('e')
        vowelCount += csvArray[x][1].count('i')
        vowelCount += csvArray[x][1].count('o')
        vowelCount += csvArray[x][1].count('u')
        consonantCount = int(csvArray[x][5]) - vowelCount
        vcArray.append([vowelCount, consonantCount])
    csvArray = np.column_stack([csvArray, vcArray])

def calculateNGram():
    global csvArray
    dupArray = np.zeros(csvArray.shape[0], dtype=np.int8)
    for x in range(0, csvArray.shape[0]):
        for y in range(0, len(csvArray[x][1])):
            dupArray[x] += csvArray[x][1].count(csvArray[x][1][y]) - 1
    csvArray = np.column_stack([csvArray, dupArray])

def writeCSV():
    file = open("finished.csv", "w")
    file.write('"host","domain","tld","class","subclass","length","vowels","consonants","ngram","entropy"\n')
    for x in range(0, csvArray.shape[0]):
        for y in range(0, csvArray.shape[1]):
            file.write(csvArray[x][y] + ",")
        file.write("\n")
    file.close()

readCSV()
# csvArray.reshape(6,-1)
progBar.message = "Calculate Length"
progBar.next()
calculateLength()
progBar.message = "Calculate Vowel-Consonant Ratio"
progBar.next()
calculateVCRatio()
progBar.message = "Calculate N-Gram"
progBar.next()
calculateNGram()
progBar.message = "Calculate Entropy"
progBar.next()
calculateEntropy()
progBar.message = "write Back"
progBar.next()
writeCSV()
progBar.writeln("Feature Selection and Extraction Complete")
progBar.finish()
print(csvArray)
