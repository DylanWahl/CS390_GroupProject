# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:21:20 2020

@author: WAHLD
@author: Ashley
"""
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter
from string import digits
from nltk import stem
import pandas as pd
import numpy as np
import string
import torch
import nltk
import re
#nltk.download('stopwords') # if you do not have all nltk library package

def main():
    file = "spam.csv"
    spam = readfile(file)
    hamWords, spamWords, sharedWords = createWordLists(spam)
    # Return the top 50 words in # just for demonstrating purposes
    # hamWords
    get_top(hamWords)
    # spamWords
    get_top(spamWords)
    # sharedWords
    get_top(sharedWords)



def readfile(file):
    # Read our file, to read properly, set encoding to Windows-1252.
    df = pd.read_csv(file, encoding='Windows-1252')
    # Rename our variables for clarity.
    df = df.rename(columns={'v1': 'label', 'v2': 'msg'})
    # Delete any unnamed columns present.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Pre-process the data through get_cleaned.
    df['msg'] = df['msg'].apply(lambda row: get_cleaned(row))
    # Convert our data frame to a numpy array to convert to dictionary.
    to_numpy = df.to_numpy()
    # return our converted numpy array.
    return to_numpy

# Pre-Processing:  data preparation to.
    # help the classifier do an optimal classification work.
# 1. Removes: punctuation and stopwords.
# 2. Performs: word stemming and removes numeric values.


def get_cleaned(text):
    # remove punctuation
    new_text = ''.join([char for char in text.lower()
                        if char not in string.punctuation])
    # tokenize
    tokens = re.split('\W+', new_text)
    # remove stopwords
    clean_text = [
        word for word in tokens if word not in stopwords.words('english')]
    # stemm
    ps = nltk.PorterStemmer()
    stemmed_text = [ps.stem(word) for word in clean_text]
    # return final un joined list to separate with in dict
    joined = " ".join([word for word in stemmed_text])
    # remove any numbers presetn
    final_text = "".join(filter(lambda x: not x.isdigit(), joined))
    return final_text
# 3. Frequent Words Identification of top 50 most used words.


def get_top(mail_dict):
    the = Counter(mail_dict)
    top = the.most_common(50)
    for i in top:
        print(i[0], ": ", i[1])

# Word Count: dictionary conversion that provide
    # the count of every word in the data set


def createWordLists(array):
    # Create empty sets of dictionaries to fill with values
    hamWords = {}
    spamWords = {}
    sharedWords = {}
    # split the numpy array into hamWordsand spamWords
    for x in array:
        x[1] = x[1].split()
        if x[0] == 'ham':
            workingDict = hamWords
        elif x[0] == 'spam':
            workingDict = spamWords
        # for the words in our messages, set the value
        for w in x[1]:
            # returns the vlaue of a key
            value = workingDict.setdefault(w, 0) + 1
            # Updates the dictionary with the elements from our array
            workingDict.update({w: value})
    # **** I will let you comment this Dyl ****
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


main()
