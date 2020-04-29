# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:21:20 2020

@author: WAHLD
@author: Ashley
"""
import pandas as pd
import numpy as np
import torch



def main():
    file = "spam.csv"
    spam = readfile(file)
    hamWords, spamWords, sharedWords = createWordLists(spam)
    print(sorted(hamWords))
    print()
    print()
    print()
    print(sorted(sharedWords))
    print(len(hamWords), len(spamWords), len(sharedWords))


def createWordLists(array):
    hamWords = {}
    spamWords = {}
    sharedWords = {}
    for x in array:
        x[1] = x[1].split()
        if x[0] == 'ham':
            workingDict = hamWords
        elif x[0] == 'spam':
            workingDict = spamWords

        for w in x[1]:
            w = w.lower()
            value = workingDict.setdefault(w, 0) + 1
            workingDict.update({w : value})


        delete = []
        for x in hamWords.keys():
            if(spamWords.get(x, 0) != 0):
                value = spamWords.get(x)
                sharedWords.update({x: value})
                spamWords.pop(x)
                newValue = sharedWords.get(x) + hamWords.get(x)
                sharedWords.update({x: newValue})
                delete.append(x)

        for key in delete:
            hamWords.pop(key)


    return hamWords, spamWords, sharedWords


def readfile(file):
    spam = pd.read_csv(file, encoding='Windows-1252')
    spam = spam.to_numpy()
    spam = spam[..., :2]
    return spam

main()
