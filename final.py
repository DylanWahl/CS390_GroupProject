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
    hamWords, spamWords, sharedWords = createWordLists(trainSet)
    print(hamWords)
    correctCount = 0
    incorrectCount = 0
    for x in testSet:
        hamSimilarity, spamSimilarity = predict_class(
            hamWords, spamWords, x[1])
        if(spamSimilarity > hamSimilarity):
            guess = 'spam'
        else:
            guess = 'ham'

        if(guess == x[0]):
            correctCount += 1
        else:
            incorrectCount += 1

    print('incorrect: ', incorrectCount, ', correct: ', correctCount)

    dfc = clean_df(df)
    numpy_df = convert_file(dfc)

    hamWords, spamWords, sharedWords = createWordLists(numpy_df)
    # Return the top 50 words in # just for demonstrating purposes
    # hamWords
#    print("Top 50 Ham")
#    get_top(hamWords)
    # spamWords
#    print("Top 50 Spam")
#    get_top(spamWords)
    # sharedWords
#    print("Top 50 Shared Words")
#    get_top(sharedWords)


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


def get_train_test(file, size_test):
    df = readfile(file)
    X_train, X_test, y_train, y_test = train_test_split(
        df["msg"], df["label"], test_size=size_test, random_state=10)
    return X_train, X_test, y_train, y_test

# ******** Ash's Naive Bayes Functions ******
# to get the prior


def get_prior(data):
    den = len(data)
    num = len(data) / 3
    return num / den


# Probability of continous data(Assume Gauss), use Gaussian from Prob Dist. Slide
def prob(val, mean, sd):
    var = sd**2
    denom = (2 * np.pi * var)**.5
    num = np.exp(-(val - mean)**2 / (2 * var))
    return num / denom


# Slide 11 from Naive Bayes Slides that we take the product(pi) of our probabilty
def prod_pi(data):
    return np.exp(sum(map(np.log, data)))


# Naive Gauss the MAP way from slide 15
def get_map(data, prior):
    return prod_pi(data) * prior


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

# option... may work better than my transform looked at sklearn.. CountVectorizer might work
#cv = CountVectorizer(max_features=max_words, stop_words='english')
#sparse_matrix = cv.fit_transform(df['label']).toarray()

#n_in, n_h, n_out, batch_size = 51, 25 , 101, 10


def get_dummy_data(n_in, n_h, n_out, batch_size):
    n_in, n_h, n_out, batch_size = 51, 25, 101, 10
    x = torch.randn(batch_size, n_in)  # 10 x 51
    x.size()
    y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [
                     0.0], [1.0], [1.0]])  # target tensor of size 10
    return x, y


def get_model(num_in, num_out, num_hidden):
    model = nn.Sequential(nn.Linear(num_in, num_hidden),
                          nn.ReLU(),
                          nn.Linear(num_hidden, num_out),
                          nn.Sigmoid())
    return model


def get_NN1(top_spam, num_in, num_out, num_hidden,x, y):
    model = get_model(num_in, num_out, num_hidden)
    
    # Gradient Descent

    criterion = torch.nn.MSELoss()
    # Construct the optimizer (Stochastic Gradient Descent in this case)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01)  # lr=learning rate

    loss_values = []
    # Gradient Descent
    # Gradient Descent
    for epoch in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
        # Compute and print loss
        loss = criterion(y_pred, y)
        loss_values.append(loss.item())

        print('epoch: ', epoch, ' loss: ', loss.item())
        # Zero gradients, perform a backward pass, and update the weights to zero
        # because PyTorch accumulates the gradients on subsequent backward passes.
        optimizer.zero_grad()
        # perform a backward pass (backpropagation)
        loss.backward()
        # Update the parameters
        optimizer.step()


def get_NN2(n_in, n_h, n_out, batch_size, top_spam, top_ham, x, y):
    model = get_model(n_in, n_h, n_out, batch_size)
    criterion = torch.nn.MSELoss()
    # Construct the optimizer (Stochastic Gradient Descent in this case)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01)  # lr=learning rate
    # Gradient Descent

    criterion = torch.nn.MSELoss()
    # Construct the optimizer (Stochastic Gradient Descent in this case)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01)  # lr=learning rate

    loss_values = []
    # Gradient Descent
    # Gradient Descent
    for epoch in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
        # Compute and print loss
        loss = criterion(y_pred, y)
        loss_values.append(loss.item())

        print('epoch: ', epoch, ' loss: ', loss.item())
        # Zero gradients, perform a backward pass, and update the weights to zero
        # because PyTorch accumulates the gradients on subsequent backward passes.
        optimizer.zero_grad()
        # perform a backward pass (backpropagation)
        loss.backward()
        # Update the parameters
        optimizer.step()


main()
