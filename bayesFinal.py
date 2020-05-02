# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:54:21 2020

@author: WAHLD
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter
from torch import tensor, nn
from string import digits
from nltk import stem
import pandas as pd
import numpy as np
import random
import string
import torch
import nltk
import re

# nltk.download('stopwords')# Uncomment if don't have this package


# Main function where all other functions are called.
def main():
    TRAIN_PERCENTAGE_ONE = .7
    TRAIN_PERCENTAGE_TWO = .8
    file = "spam.csv"
    df = readfile(file)
    numpy_df = convert_file(df)
    trainSet, testSet = split_set(numpy_df, TRAIN_PERCENTAGE_ONE)
#    trainSet2, testSet2 = split_set(numpy_df, TRAIN_PERCENTAGE_TWO)

    hamWords, spamWords, sharedWords = createWordLists(trainSet)
#    hamWords2, spamWords2, sharedWords2 = createWordLists(trainSet2)

    print('Baysian classifier with a 70:30 train/test split')
    correctCount, incorrectCount = get_results(testSet, hamWords, spamWords)
    correctPercentage = (correctCount / (correctCount + incorrectCount)) * 100
    print('incorrect: ', incorrectCount, ', correct: ', correctCount)
    print('The Baysian classifier correctly Identified the text ',
          correctPercentage, 'percent of the time!')

#    print('Baysian classifier with a 80:20 train/test split')
#    correctCount, incorrectCount = get_results(testSet2, hamWords2, spamWords2)
#    correctPercentage = (correctCount / (correctCount + incorrectCount)) * 100
#    print('incorrect: ', incorrectCount, ', correct: ', correctCount)
#    print('The Baysian classifier correctly Identified the text ',
#          correctPercentage, 'percent of the time!')
    
    
def get_results(testSet, hamWords, spamWords):
    correctCount = 0
    incorrectCount = 0
    for x in testSet:
        hamSimilarity, spamSimilarity = predict_class(hamWords, spamWords, x[1])
        if(spamSimilarity > hamSimilarity):
            guess = 'spam'
        else:
            guess = 'ham'

        if(guess == x[0]):
            correctCount += 1
        else:
            incorrectCount += 1
    return correctCount, incorrectCount


def predict_class(hamWords, spamWords, sentence):
    sentence = get_cleaned(sentence)

    hamSimilarity = 0
    spamSimilarity = 0
    for x in sentence:
        if(x in hamWords):
            hamSimilarity += 1
        if(x in spamWords):
            spamSimilarity += 1

    return hamSimilarity, spamSimilarity


def split_set(data, trainProp):
    random.shuffle(data)
    trainNumber = int(len(data) * trainProp)
    trainData = data[:trainNumber]
    testData = data[trainNumber:]

    return trainData, testData


def readfile(file):
    # Read our file, to read properly, set encoding to Windows-1252.
    df = pd.read_csv(file, encoding='Windows-1252')
    # Rename our variables for clarity.
    df = df.rename(columns={'v1': 'label', 'v2': 'msg'})
    # Delete any unnamed columns present.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Drop any duplicate data
    df = df.drop_duplicates()
    return df


def clean_df(df):
    df['msg'] = df['msg'].apply(lambda row: get_cleaned(row))
    return df


def np_separate(df):
    return np.array(df.loc[:, df.columns != 'label'])


def clean_split_data(df):
    return np.char.lower([get_cleaned(sentence)for sentence in df])

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
    lst = []
    for i in top:
        lst.append(i[0])
    return np.array(lst)

# Word Count: dictionary conversion that provide
    # the count of every word in the data set


def convert_file(data_frame):
    return data_frame.to_numpy()


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
        # removes the words that are shared by both the ham and spam lists andputs them in a shared list
        delete = []
        for x in hamWords.keys():
            # if the word is in the other list
            if(spamWords.get(x, 0) != 0):
                # retrieves the value to update shared words
                value = spamWords.get(x)
                sharedWords.update({x: value})
                # removes word from spam list
                spamWords.pop(x)
                # updates shared words a second time
                newValue = sharedWords.get(x) + hamWords.get(x)
                sharedWords.update({x: newValue})
                delete.append(x)

        #delete all of the shared words since i couldnt do it while iterating
        for key in delete:
            hamWords.pop(key)

    return hamWords, spamWords, sharedWords


def get_train_test(file, size_test):
    df = readfile(file)
    X_train, X_test, y_train, y_test = train_test_split(
        df["msg"], df["label"], test_size=size_test, random_state=10)
    return X_train, X_test, y_train, y_test
    
    
main()
    
    