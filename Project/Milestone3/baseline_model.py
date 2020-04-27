# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:45:10 2019

@author: zxj62
"""

import numpy
import scipy
from scipy.sparse import csr_matrix

import sklearn
from sklearn.linear_model import LogisticRegression

import nltk
nltk.download('brown')
nltk.download('universal_tagset')
from nltk.corpus import brown

feature_dict = {}
tag_dict = {}
rare_words = set()

def word_ngram_features(i, words):
    features = []
    newWords = ["<s>", "<s>"] + words + ["</s>", "</s>"]
    j = i+2
    features.append('prevbigram-'+newWords[j-1])
    features.append('nextbigram-'+newWords[j+1])
    features.append('prevskip-'+newWords[j-2])
    features.append('nextskip-'+newWords[j+2])
    features.append('prevtrigram-'+newWords[j-1] + '-' + newWords[j-2])
    features.append('nexttrigram-'+newWords[j+1] + '-' + newWords[j+2])
    features.append('centertrigram-'+newWords[j-1] + '-' + newWords[j+1])
    return features

def word_features(word, rare_words):
    features = []
    if word not in rare_words:
        features.append("word-"+word)
    if word[0].isupper():
        features.append('capital')
    if any(char.isdigit() for char in word):
        features.append('number')
    if '-' in word:
        features.append('hyphen')
    for i in range(1,5):
        if len(word) >= i:
            features.append('prefix'+(str)(i)+'-'+word[:i])
    for i in range(1,5):
        if len(word) >= i:
            features.append('suffix'+(str)(i)+'-'+word[-i:])  
    return features

def get_features(i, words, prevtag, rare_words):
    features1 = word_ngram_features(i, words)
    features2 = word_features(words[i], rare_words)
    features = features1 + features2
    features.append('tagbigram-'+prevtag)
    return [feature.lower() for feature in features]

def remove_rare_features(features, n):
    count_features = {}
    for featureList in features:
        for wordFeatures in featureList:
            for feature in wordFeatures:
                count_features[feature] = count_features.get(feature, 0) + 1
    rare_features = set()
    non_rare_features = set()
    for feature in count_features:
        if count_features.get(feature) < n:
            rare_features.add(feature)
        else:
            non_rare_features.add(feature)
    newFeatures = [[[feature for feature in wordFeatures if feature not in rare_features] for wordFeatures in featureList] for featureList in features]
    return newFeatures, non_rare_features

def build_Y(tags):
    Y = []
    for sentenceTag in tags:
        for tag in sentenceTag:
            if tag in tag_dict:
                Y.append(tag_dict[tag])
    return numpy.array(Y)

def build_X(features):
    examples = []
    _features = []
    i = -1
    for sentenceFeature in features:
        for wordFeatures in sentenceFeature:
            i+=1
            for feature in wordFeatures:
                if feature in feature_dict:
                    examples.append(i)
                    _features.append(feature_dict[feature])    
    values = [1] * len(examples)
    examples = numpy.array(examples)
    _features = numpy.array(_features)
    values = numpy.array(values)
    return csr_matrix((values, (examples, _features)), shape=(i+1, len(feature_dict)))

def load_test(filename):
    sentences = []
    with open(filename,'r') as file:
        sentenceFile = file.read().splitlines()
    for sentence in sentenceFile:
        sentences.append(sentence.split(" "))
    return sentences

def get_predictions(test_sentence, model):
    Y_pred = numpy.empty((len(test_sentence[0])-1, len(tag_dict), len(tag_dict)))
    words = test_sentence[0]
    for i in range(1, len(words)):
        for tag in tag_dict:
            wordFeatures = get_features(i, words, tag, rare_words)
            X = build_X([[wordFeatures]])
            distribution = model.predict_log_proba(X)
            Y_pred[i-1,tag_dict[tag]] = distribution
            
    startWordFeatures = [[get_features(0, words, '<S>', rare_words)]]
    startX = build_X(startWordFeatures)
    startDistribution = model.predict_log_proba(startX)
    Y_start = numpy.array(startDistribution)
    
    return Y_pred, Y_start

def main():
    brown_sentences = brown.tagged_sents(tagset='universal')
    train_sentences = []
    train_tags = []
    training_features = []
       
    wordDict = {}
    rare_words = set()
    for sentence in brown_sentences:
        words = []
        tags = []
        for pair in sentence:
            words.append(pair[0])
            wordDict[pair[0]] = wordDict.get(pair[0], 0) + 1
            tags.append(pair[1])
        train_sentences.append(words)
        train_tags.append(tags)
    for word in wordDict:
        if wordDict.get(word) < 5:
            rare_words.add(word)
    
    for i in range(len(train_sentences)):
        words = train_sentences[i]
        tags = ['<S>']+train_tags[i]
        sentenceFeatures = []        
        for j in range(len(words)):
            wordFeatures = get_features(j, words, tags[j], rare_words)
            sentenceFeatures.append(wordFeatures)
        training_features.append(sentenceFeatures)
        
    newFeatures, non_rare_features = remove_rare_features(training_features, 5)
    training_features = newFeatures
    
    x = 0
    for word in non_rare_features:
        if word not in feature_dict:
            feature_dict[word] = x
            x+=1
            
    i = 0
    for tags in train_tags:
        for tag in tags:
            if tag not in tag_dict:
                tag_dict[tag] = i
                i+=1
                
    X_train = build_X(training_features)
    Y_train = build_Y(train_tags)

    LRModel = LogisticRegression(class_weight = 'balanced', solver = 'saga', multi_class = 'multinomial')
    LRModel.fit(X_train, Y_train)  
    
    testData = load_test(r'test.txt')
    for i in range(len(testData)):
        Y_pred, Y_start = get_predictions([testData[i]], LRModel)
        print("The highest-probability sequence of tags for sentence", i+1, "is: ")
        print(viterbi(Y_start, Y_pred))
    
    
if __name__ == '__main__':
    main()
    