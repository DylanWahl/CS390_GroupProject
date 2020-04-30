from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
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
#nltk.download('stopwords')# Uncomment if don't have this package


# Main function where all other functions are called.
def main():
    file = "spam.csv"
    df = readfile(file)
    numpy_df = convert_file(df)
    hamWords, spamWords, sharedWords = createWordLists(numpy_df)
    # Return the top 50 words in # just for demonstrating purposes
    # hamWords
    print("Top 50 Ham")
    get_top(hamWords)
    # spamWords
    print("Top 50 Spam")
    get_top(spamWords)
    # sharedWords
    print("Top 50 Shared Words")
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
    # Drop any duplicate data
    df = df.drop_duplicates()
    return df


def convert_file(data_frame):
    return data_frame.to_numpy()

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
def get_train_test(file, size_test):
  df = readfile(file)
  X_train,X_test,y_train,y_test = train_test_split(df["msg"],df["label"], test_size = size_test, random_state = 10)
  return X_train,X_test,y_train,y_test

  # ******** Ash's Naive Bayes Functions ******
  #to get the prior
def get_prior(data):
  den = len(data)
  num = len(data)/3
  return num/den

# Probability of continous data(Assume Gauss), use Gaussian from Prob Dist. Slide
def prob(val, mean, sd):
  var = sd**2
  denom = (2*np.pi*var)**.5
  num = np.exp(-(val-mean)**2/(2*var))
  return num/denom

# Slide 11 from Naive Bayes Slides that we take the product(pi) of our probabilty
def prod_pi(data):
    return np.exp(sum(map(np.log, data)))

#Naive Gauss the MAP way from slide 15
def get_map(data,prior):
    return prod_pi(data) * prior
    

<<<<<<< HEAD
main()
=======

>>>>>>> 405c0e47dbcc8f9cad47f6f2dd2536cb2feab7fa


# **************** NOT FINAL: Still working on **********
def torch_conversion(strg):
    csv = pd.read_csv(strg)
    to_nump = csv.to_numpy()
    convert = torch.from_numpy(to_nump)
    return convert



# ! ! ! Need to figure out tokens
def get_dummy_data(batch_size,num_in):
    x = torch.randn(batch_size, n_in) # 10*10 tensor
    test = torch_conversion('dr.csv')
    # should be a token from the csv file 2 classes since 2 outputs?
    y = torch.tensor([[0.0], [1.0]]) # target tensor of size 10
    return x,y



def get_model(num_in,num_out,num_hidden):
    model = nn.Sequential(nn.Linear(num_in, num_hidden),
    nn.ReLU(),
    nn.Linear(num_hidden, num_out),
    nn.Sigmoid())
    return model



def get_hidden_num(num_in,num_out,n):
    a = 7
    return n/(a*(num_in + num_out))

def get_gradient_descent(x_train, y_train):
    for epoch in range(50):
        #To store our loss values
        loss_val = []
        # Define our criterion using MSELoss
        criterion = torch.nn.MSELoss()
        # optimiser uses the rate of loss function w.r.t. the parameters
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # Forward Propagation
        y_pred = model(x_train)
        # Compute and print loss
        loss = criterion(y_pred, y_train)
        # Add our loss items to the array
        loss_val.append(loss.item())
        # Zero the gradients
        optimizer.zero_grad()
        # Pass a predition function
        pred = torch.max(y_pred,1)[1].eq(y_train).sum()
        # Test the accuracy of the prediction
        accuracy = pred * 100.0 / len(x_train)
        # perform a backward pass (backpropagation)
        loss.backward()
        # Update the parameters
        optimizer.step()

        return pred


main()
