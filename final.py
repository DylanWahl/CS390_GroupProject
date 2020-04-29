# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:21:20 2020

@author: WAHLD
@author: Ashley
"""
import pandas as pd
import numpy as np
import torch
#Ashley Added some libraries
import nltk
from nltk import stem
from nltk.corpus import stopwords



def main():
    file = "spam.csv"
    spam = readfile(file)
    hamWords, spamWords, sharedWords = createWordLists(spam)
    df = read_file('spam.csv')
    # I added a new column, dont have to
    #df['clean'] = df['msg'].apply(gen_ST)# OR
    df['clean'] = df['msg'].apply(gen_ST)
    #give us the length/word count
    df['length'] = df['clean'].apply(len)
# Ashley's Code ****
# Ash's work
        # Note: I do not think we need to have shared words for what he is asking

def read_file(strg):
    df = pd.read_csv(strg, encoding='Windows-1252')
    df = df.rename(columns={'v1': 'label', 'v2': 'msg'})
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]


# our stemming and lemninzation
stemmer = stem.SnowballStemmer('english')
stopwords = set(stopwords.words('english'))


def clean_data(doc):  # Rename for clarity
    lowercase = doc.lower()
    token = nltk.word_tokenize(lowercase)
    words = [word for word in token if not word in stopwords]
    # if there is punc. don't include
    new_words = [word for word in words if word.isalnum()]
    words = [stemmer.stem(words) for words in new_words]
    return words


def convert(df):
    str_to_num = {'ham': 0, 'spam': 1}
    new_df = df.replace(str_to_num)
    return new_df.to_dict('index')


convert(df)

# End of Ashley's Code ****

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
