# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 22:57:25 2019

@author: zxj62
"""

#import and download the necessary libs
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
import numpy
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# the patterns for matching the 3 types of single quotes
pattern1 = r"(\B')(\w+)('\B)"
pattern2 = r"(\B')(\w+)"
pattern3 = r"(\w+)('\B)"

# pattern to match the negation tokens
negation = r"(\b)(not|never|cannot|\w+n't)(\b)"
# pattern to match the ending of negation
negation_ending_tokens = set(["but", "however", "nevertheless", ".", "?", "!"])

# opens the file at corpus path and reads in the data, which 
# return a list of tuples (snippet, label)
def load_corpus(corpus_path):
    pairs = []
    with open(corpus_path, 'r') as file:
        text = file.read().splitlines()
    for data in text:
        snippet, label = data.split('\t')
        pairs.append((snippet, int(label)))
    return pairs
        
# function to match single quote words
def tokenize(snippet):
    snippet = re.sub(pattern1, r'\1 \2 \3', snippet)
    snippet = re.sub(pattern2, r'\1 \2', snippet)
    snippet = re.sub(pattern3, r'\1 \2', snippet)   
    return snippet.split(" ")

# function to tag the bracketed words with EDIT_
def tag_edits(tokenized_snippet):
    start = False
    for i in range(len(tokenized_snippet)):
        word = tokenized_snippet[i]
        if word[0] == '[' and word[-1] == ']':
            word = "EDIT_" + word[1:-1]
        elif word[0] == '[':
            start = True
            word = "EDIT_" + word[1:]
        elif word[-1] == ']':
            word = "EDIT_" + word[:-1]
            start = False
        elif start:
            word = "EDIT_" + word
        if word != 'EDIT_' and word != '':
            tokenized_snippet[i] = word
    return tokenized_snippet

# function to detect and matche all negation words, marking'em with NOT_ tags
def tag_negation(tokenized_snippet):
    dummy = []
    savedTag = {}
    for i in range(len(tokenized_snippet)):
        word = tokenized_snippet[i]
        if "EDIT_" in word:
            savedTag[i] = "EDIT_"
            word = word[5:] 
        dummy.append(word)
    analysis = nltk.pos_tag(dummy)
    for i in range(len(analysis)):
        if (i in savedTag):
            word, pos = analysis[i]
            word = savedTag.get(i) + word
            analysis[i] = (word, pos)
    flag = False
    for i in range(len(analysis)):
        word, pos = analysis[i]
        check = word[:]
        if "EDIT_" in check:
            check = check[5:]
        if flag == False and re.match(negation, check):
            if check == "not" and i < len(analysis) - 1:
                nextCheck = analysis[i+1][0]
                if "EDIT_" in nextCheck:
                    nextCheck = nextCheck[5:]
                if (nextCheck != 'only'):
                    flag = True
            else:
                flag = True
        elif check in negation_ending_tokens or pos == 'JJR' or pos == 'RBR':
            flag = False
        elif flag == True:
            word = "NOT_" + word
            analysis[i] = (word, pos)  
    return analysis           # the return value is the nltk.pos_tag data


# function to create the vocabulary dictionary from the given address 
# of the file
def create_vocabulary(address):
    index = 0
    # here I also create a reverse_vocabulary so that 
    # I could easily fetch the top features
    vocabulary, reverse_vocabulary = {}, {}
    data = load_corpus(address)
    for snippet, label in data:
        tokens = tag_negation(tag_edits(tokenize(snippet)))
        for token, pos in tokens:
            if ("EDIT_" not in token and token not in vocabulary):
                vocabulary[token] = index
                reverse_vocabulary[index] = token
                index += 1
    return vocabulary, reverse_vocabulary


# funciton to returns a feature vector in a Numpy array
def get_features(preprocessed_snippet):
    array = numpy.zeros(len(vocabulary)+3)
    for word, pos in preprocessed_snippet:
        if "EDIT_" not in word and word in vocabulary:
            index = vocabulary.get(word)
            array[index] += 1
    act, plea, ima = score_snippet(preprocessed_snippet, dal)
    array[-1] = ima
    array[-2] = plea
    array[-3] = act
    return array


# function to normalize the feature array
def normalize(X):
    rows = len(X)
    columns = len(X[0])
    minVals = numpy.amin(X, axis = 0)
    maxVals = numpy.amax(X, axis = 0)
    for column in range(columns):
        minVal = minVals[column]
        maxVal = maxVals[column] 
        if not (minVal == maxVal):
            for row in range(rows):
                X[row][column] = (X[row][column]-minVal)/(maxVal-minVal)
    return X

# function to return precision, recall and F-measure        
def evaluate_predictions(Y_pred, Y_true):
    tp, fp, fn = 0, 0, 0
    for pl, tl in zip(Y_pred, Y_true):
        if pl == 1 and tl == 1:
            tp += 1
        if pl == 1 and tl == 0:
            fp += 1
        if pl == 0 and tl == 1:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = 2*((precision * recall) / (precision + recall))    
    return (precision, recall, fmeasure)    

# function to return the top k features in the model
def top_features(logreg_model, k):
    weights = logreg_model.coef_[0]
    pairs = []
    for i in range(len(weights)):
        pairs.append((i, weights[i]))
    pairs.sort(key=lambda tup:abs(tup[1]), reverse = True)
    res = []
    i = 0
    while len(res) < k:
        index, weight = pairs[i]
        if index not in reverse_vocabulary:
            i+=1
        else:
            word = reverse_vocabulary[index]
            res.append((word, weight))
            i+=1
    return res

# function to load dictionary of affect
def load_dal(dal_path):
    res = {}
    with open(dal_path, 'r') as file:
        next(file)
        for line in file:
            word, act, eva, ima = line.split('\t')
            act = float(act)
            eva = float(eva)
            ima = float(ima)
            res[word] = (act, eva, ima)
    return res

# function to convert NLTK POS tags into WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''
    
# function to return the snippet score, the function has been modified
def score_snippet(prepocessed_snippet, dal):
    act = 0
    plea = 0
    ima = 0
    count = 0
    for word, pos in preprocessed_snippet:
        if "EDIT_" not in word:
            flag = False
            if "NOT_" in word:
                word = word[4:]
                flag = True
            if word not in dal:
                synWords = set()
                antWords = set()
                synset = wn.synsets(word, get_wordnet_pos(pos))
                for syn in synset: 
                    for l in syn.lemmas(): 
                        synWords.add(l.name()) 
                        if l.antonyms(): 
                            antWords.add(l.antonyms()[0].name())
                for replace in synWords:
                    if replace in dal:
                        word = replace
                        break
                if word not in dal:
                    for replace in antWords:
                        if replace in dal:
                            word = replace
                            flag = not flag
                            break               
            if word in dal:
                aVal, pVal, iVal = dal.get(word)
                if flag:
                    act += -aVal
                    plea += -pVal
                    ima += -iVal
                else:
                    act += aVal
                    plea += pVal
                    ima += iVal
                count += 1
    if count == 0:
        return (0,0,0)
    else:
        return (act/count, plea/count, ima/count)


# the source data
dal = load_dal(r"dict_of_affect.txt")   
data = load_corpus(r"train.txt")
vocabulary, reverse_vocabulary = create_vocabulary(r"train.txt")

# arrays of features in the train data
X_train = numpy.empty([len(data), len(vocabulary)+3])
Y_train = numpy.empty(len(data))

# get the X_train, Y_train
index = 0
for snippet, label in data:
    preprocessed_snippet = tag_negation(tag_edits(tokenize(snippet)))
    X_train[index] = get_features(preprocessed_snippet)
    Y_train[index] = label
    index += 1

X_train = normalize(X_train)

test = load_corpus(r"test.txt")

# get the X_test data, Y_true data
X_test = numpy.empty([len(test), len(vocabulary)+3])
Y_true = numpy.empty(len(test))

index = 0
for snippet, label in test:
    preprocessed_snippet = tag_negation(tag_edits(tokenize(snippet)))
    X_test[index] = get_features(preprocessed_snippet)
    Y_true[index] = label
    index += 1
    
X_test = normalize(X_test)

# create and train the Gaussian-NB model, and evaluate the model
model = GaussianNB()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
preds = evaluate_predictions(Y_pred, Y_true)
print("The test for the GaussianNB model: ")
print("Precision:", preds[0])
print("Recall:", preds[1])
print("F-measure:", preds[2])

# create and train the Logistic-Regression model, and evaluate the model
lrModel = LogisticRegression()
lrModel.fit(X_train, Y_train)
Y_pred = lrModel.predict(X_test)
preds = evaluate_predictions(Y_pred, Y_true)
print("\nThe test for the LogisticRegression model: ")
print("Precision:", preds[0])
print("Recall:", preds[1])
print("F-measure:", preds[2])


# return the top 10 features
print('\n')
print("The top 10 features of the LogisticRegression model are: ")
print(top_features(lrModel, 10))