# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:06:19 2020

@author: Ashley
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

# Steps to Classify With a Neural Network for Beginners

# 1. Pre-Processing:  data preparation to.
    # help the classifier do an optimal classification work.
# 1.a Removes: punctuation and stopwords.
# 1.b Performs: word stemming and removes numeric values.

# 2. Representing text as numerical data
# Reading a text-based dataset into pandas
# Vectorizing our dataset
# Building and evaluating a model
# Comparing our two models

def main():
    TRAIN_PERCENTAGE_ONE = .7
    TRAIN_PERCENTAGE_TWO = .8
    file = "spam.csv"
    df = readfile(file)
    numpy_df = convert_file(df)
    trainSet, testSet = split_set(numpy_df, TRAIN_PERCENTAGE_ONE)
#    trainSet2, testSet2 = split_set(numpy_df, TRAIN_PERCENTAGE_TWO)

    print(" Neural Networks Classifier with a 70: 30 train/test split ")
    trainSet1, testSet1 = split_set(numpy_df, TRAIN_PERCENTAGE_ONE)
    n_in, n_h, n_out, batch_size = 51, 25, 101, 10
    model = get_model(n_in, n_h, n_out)
    test = ['go', 'jurong', 'u', 'crazi', 'ok', 'bugi','u', 'crazi', 'ok', 'bugi'] # given tes

    ham,spam = get_top(df)

    x = transform(test,ham)
    y = transform(ham,test)

    bar_Top_ham(ham)
    bar_Top_spam(spam)

    combined = combine(ham,spam)

    NN1(n_in, n_h, n_out, batch_size)
    NN2(n_in, n_h, n_out, batch_size)







def bar_Top_ham(ham):
    ham.plot.bar(legend = False)
    y_pos = np.arange(len(ham[0]))
    plt.xticks(y_pos, ham[0])
    plt.title('Top 50 words in ham')
    plt.xlabel('Words')
    plt.ylabel('Count')
    plt.show()

def bar_Top_spam(spam):
    spam.plot.bar(legend = False)
    y_pos = np.arange(len(spam[0]))
    plt.xticks(y_pos, spam[0])
    plt.title('Top 50 words in spam')
    plt.xlabel('Words')
    plt.ylabel('Count')
    plt.show()


def clean_df(df):
    df['msg'] = df['msg'].apply(lambda row: get_cleaned(row))
    return df


def clean_split_data(df):
    return np.char.lower([get_cleaned(sentence)for sentence in df])

def combine(ham,spam):
    # Assumes both lists are of the same size.
    combined = []
    for i in range(len(ham)):
        combined.append(ham[i])
        combined.append(spam[i])
    return combined



def convert_file(data_frame):
    return data_frame.to_numpy()


#n_in, n_h, n_out, batch_size = 51, 25 , 101, 10
def get_dummy_data(n_in, n_h, n_out, batch_size):
    n_in, n_h, n_out, batch_size = 51, 25, 101, 10
    x = torch.randn(batch_size, n_in)  # 10 x 51
    x.size()
    y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [
                     0.0], [1.0], [1.0]])  # target tensor of size 10
    return x, y

# 3. Frequent Words Identification of top 50 most used words.


def get_top(df):
    # top 50 ham
    hamCountDict = Counter(" ".join(df[df['label']=='ham']["msg"]).split()).most_common(50)
    df_ham = pd.DataFrame.from_dict(hamCountDict)
    npd_ham = np.array(df_ham[0])
    # top 50 spam
    spamCountDict = Counter(" ".join(df[df['label']=='spam']["msg"]).split()).most_common(50)
    df_spam = pd.DataFrame.from_dict(spamCountDict)
    npd_spam = np.array(df_spam[0])
    # Return our top 50 ham and spam as np array (as suggested)
    return npd_ham, npd_spam
ham,spam = get_top(df)

def get_train_test(file, size_test):
    df = readfile(file)
    X_train, X_test, y_train, y_test = train_test_split(
        df["msg"], df["label"], test_size=size_test, random_state=10)
    return X_train, X_test, y_train, y_test




# layer definition
#n_in, n_h, n_out, batch_size = 51, 25, 2, 10
def NN1(n_in, n_h, n_out, batch_size,y):

    # Create dummy input and target tensors (data)
    x = torch.randn(batch_size, n_in) # 10 word sentence
    # Example sentence has these words that match in our list
    #y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]]) # target tensor of size 10
    print(y.shape)
    nn.Linear(n_in, n_out, bias=True)
    print(type(y))
    # Construct our model
    model = nn.Sequential(nn.Linear(n_in, n_h),
    nn.ReLU(),
    nn.Linear(n_h, n_out),
    nn.Sigmoid())


    # Construct the loss function
    criterion = torch.nn.MSELoss()
    # Construct the optimizer (Stochastic Gradient Descent in this case)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # lr=learning rate
    # Gradient Descent
    for epoch in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
        # Compute and print loss
        loss = criterion(y_pred, y)
        print('epoch: ', epoch,' loss: ', loss.item())
        # Zero gradients, perform a backward pass, and update the weights to zero
        # because PyTorch accumulates the gradients on subsequent backward passes.
        optimizer.zero_grad()
        # perform a backward pass (backpropagation)
        loss.backward()
        # Update the parameters
        optimizer.step()
    out.backward()




# n_in, n_h, n_out, batch_size = 101, 51 , 2, 10
def NN2(n_in, n_h, n_out, batch_size):
    x = torch.randn(batch_size, n_in) # 10 word sentence
    # Example sentence has these words that match in our list
    y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]]) # target tensor of size 10

    nn.Linear(n_in, n_out, bias=True)

    # Construct our model
    model = nn.Sequential(nn.Linear(n_in, n_h),
    nn.ReLU(),
    nn.Linear(n_h, n_out),
    nn.Sigmoid())


    # Construct the loss function
    criterion = torch.nn.MSELoss()
    # Construct the optimizer (Stochastic Gradient Descent in this case)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # lr=learning rate
    # Gradient Descent
    for epoch in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
        # Compute and print loss
        loss = criterion(y_pred, y)
        print('epoch: ', epoch,' loss: ', loss.item())
        # Zero gradients, perform a backward pass, and update the weights to zero
        # because PyTorch accumulates the gradients on subsequent backward passes.
        optimizer.zero_grad()
        # perform a backward pass (backpropagation)
        loss.backward()
        # Update the parameters
        optimizer.step()
    n_out.backward()



def pie_words(df):
    count_Class.plot(kind = 'pie',labels=df['label'],autopct='%1.0f%%')
    plt.ylabel('Distribution of words')
    plt.show()


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


def split_set(data, trainProp):
    random.shuffle(data)
    trainNumber = int(len(data) * trainProp)
    trainData = data[:trainNumber]
    testData = data[trainNumber:]

    return trainData, testData



def torch_conversion(strg):
    csv = pd.read_csv(strg)
    to_nump = csv.to_numpy()
    convert = torch.from_numpy(to_nump)
    return convert


def transform(text1, text2):
    lst = []
    vect = [i in text1 for i in text2]

    for i in vect:
        if i:
            lst.append(1)
        else:
            lst.append(0)
    print(lst)

main()
