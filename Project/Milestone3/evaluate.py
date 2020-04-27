# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:02:36 2019

@author: zxj62
"""
import numpy
import sys
import pickle

filename = 'baseline.model'
model = pickle.load(open(filename, 'rb'))

def load_test(filename):
    sentences = []
    vals = []
    with open(filename,'r') as file:
        sentenceFile = file.read().splitlines()
    for sentence in sentenceFile:
        
        sent, val = sentence.split("\t")
        
        sent.replace(",", "")
        sent.replace(".", "")
        sent.replace("!", "")
        sent.replace('"', "")
        sent.replace('?', "")
        
        sentences.append(sent.split(" "))
        vals.append([int(i) for i in val.strip().split(" ")])
    return sentences, vals

def build_Y(vals):
    Y = []      
    for val in vals:
        for value in val:
            Y.append(value)
    return numpy.array(Y)

def build_X(sentences):
    X = []      
    for sentence in sentences:
        for word in sentence:
            if word:
                X.append([0])
            else:
                X.append([1])
    return numpy.array(X)

def evaluate_predictions(Y_pred, Y_true):
    tp, fp, fn = 0, 0, 0
    for pl, tl in zip(Y_pred, Y_true):
        if pl == 0 and tl == 0:
            tp += 1
        if pl == 1 and tl == 0:
            fp += 1
        if pl == 0 and tl == 1:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = 2*((precision * recall) / (precision + recall))    
    return (precision, recall, fmeasure)

def main(args):
    filepath = args[0]
    sentences, vals = load_test(filepath)
    X = build_X(sentences)
    Y_true = build_Y(vals)
    Y_pred = model.predict(X)

    print('Precision: %f\nRecall:%f\nF-measure: %f' % evaluate_predictions(Y_pred, Y_true))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
