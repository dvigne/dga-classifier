#!/usr/bin/python
from __future__ import division

import numpy as np
import scipy.stats as stats
from progress.bar import Bar

# Numpy Configuration
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold='nan')

fileName = "data/sampledga.csv"
csvArray = [None]
progBar = Bar('Load CSV', max=6)
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
    vcArray = np.zeros(csvArray.shape[0])
    for x in range(0, csvArray.shape[0]):
        vowelCount = 0
        vowelCount += csvArray[x][1].count('a')
        vowelCount += csvArray[x][1].count('e')
        vowelCount += csvArray[x][1].count('i')
        vowelCount += csvArray[x][1].count('o')
        vowelCount += csvArray[x][1].count('u')
        consonantCount = int(csvArray[x][5]) - vowelCount
        vcArray[x] = vowelCount / consonantCount
    csvArray = np.column_stack([csvArray, vcArray])

def countLetters(letter, ngramCount, position):
    ngramCount = ngramCount
    if(position == len(letter) - 1):
        return ngramCount
    elif(letter[position] == letter[position + 1]):
        ngramCount = ngramCount + 1
    return countLetters(letter, ngramCount, position + 1)

def calculateNGram():
    global csvArray
    dupArray = np.zeros(csvArray.shape[0], dtype=np.int8)
    for x in range(0, csvArray.shape[0]):
        dupArray[x] = countLetters(csvArray[x][1], 0, 0)
    csvArray = np.column_stack([csvArray, dupArray])

def calculateNumbers():
    global csvArray
    capitalArray = np.zeros(csvArray.shape[0], dtype=np.int8)
    for x in range(0, csvArray.shape[0]):
        capitalCount = 0
        for y in range(0, len(csvArray[x][1])):
            if(ord(csvArray[x][1][y]) >= 48 and ord(csvArray[x][1][y]) <= 57):
                capitalCount += 1
        capitalArray[x] = capitalCount
    csvArray = np.column_stack([csvArray, capitalArray])

def writeCSV():
    file = open("finished.csv", "w")
    file.write('"host","domain","tld","class","subclass","length","vc-ratio","ngram","entropy","numbers"\n')
    for x in range(0, csvArray.shape[0]):
        for y in range(0, csvArray.shape[1]):
            file.write(csvArray[x][y] + ",")
        file.write("\n")
    file.close()

readCSV()
csvArray.reshape(5,-1)
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
progBar.message = "Calculate Number Count"
progBar.next()
calculateNumbers()
progBar.message = "write Back"
progBar.next()
writeCSV()
progBar.writeln("Feature Selection and Extraction Complete")
progBar.finish()
print(csvArray)
