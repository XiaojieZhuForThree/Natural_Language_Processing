# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:53:00 2019

@author: zxj62
"""

import nltk
nltk.download('averaged_perceptron_tagger')
import numpy
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

pattern1 = r"(\B')(\w+)('\B)"
pattern2 = r"(\B')(\w+)"
pattern3 = r"(\w+)('\B)"
negation = r"(\b)(not|never|cannot|\w+n't)(\b)"
negation_ending_tokens = set(["but", "however", "nevertheless", ".", "?", "!"])

def load_corpus(corpus_path):
    pairs = []
    with open(corpus_path, 'r') as file:
        text = file.read().splitlines()
    for data in text:
        snippet, label =  data.split('\t')
        pairs.append((snippet, int(label)))
    return pairs
        
def tokenize(snippet):
    snippet = re.sub(pattern1, r'\1 \2 \3', snippet)
    snippet = re.sub(pattern2, r'\1 \2', snippet)
    snippet = re.sub(pattern3, r'\1 \2', snippet)   
    return snippet.split(" ")

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
    return analysis           
    
def create_vocabulary(address):
    index = 0
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

vocabulary, reverse_vocabulary = create_vocabulary(r"train.txt")
 
def get_features(preprocessed_snippet):
    array = numpy.zeros(len(vocabulary))
    for word, pos in preprocessed_snippet:
        if "EDIT_" not in word and word in vocabulary:
            index = vocabulary.get(word)
            array[index] += 1
    return array

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


def top_features(logreg_model, k):
    weights = logreg_model.coef_[0]
    pairs = []
    for i in range(len(weights)):
        pairs.append((i, weights[i]))
    pairs.sort(key=lambda tup:abs(tup[1]), reverse = True)
    res = []
    for i in range(k):
        index, weight = pairs[i]
        word = reverse_vocabulary[index]
        res.append((word, weight))
    return res

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

def score_snippet(preprocessed_snippet, dal):
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
data = load_corpus(r"train.txt")


X_train = numpy.empty([len(data), len(vocabulary)])
Y_train = numpy.empty(len(data))

index = 0
for snippet, label in data:
    preprocessed_snippet = tag_negation(tag_edits(tokenize(snippet)))
    X_train[index] = get_features(preprocessed_snippet)
    Y_train[index] = label
    index += 1

X_train = normalize(X_train)

test = load_corpus(r"test.txt")

X_test = numpy.empty([len(test), len(vocabulary)])
Y_true = numpy.empty(len(test))

index = 0
for snippet, label in test:
    preprocessed_snippet = tag_negation(tag_edits(tokenize(snippet)))
    X_test[index] = get_features(preprocessed_snippet)
    Y_true[index] = label
    index += 1
    
X_test = normalize(X_test)

model = GaussianNB()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
preds = evaluate_predictions(Y_pred, Y_true)
print("The test for the GaussianNB model: ")
print("Precision:", preds[0])
print("Recall:", preds[1])
print("F-measure:", preds[2])

lrModel = LogisticRegression()
lrModel.fit(X_train, Y_train)
Y_pred = lrModel.predict(X_test)
preds = evaluate_predictions(Y_pred, Y_true)
print("\nThe test for the LogisticRegression model: ")
print("Precision:", preds[0])
print("Recall:", preds[1])
print("F-measure:", preds[2])

print('\n')
print("The top 10 features of the LogisticRegression model are: ")
print(top_features(lrModel, 10))