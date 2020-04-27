# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:43:35 2019

@author: zxj62
"""
import numpy
import pickle
import sys

from sklearn.linear_model import LogisticRegression
import numpy
import scipy
from scipy.sparse import csr_matrix

import sklearn
from sklearn.linear_model import LogisticRegression

import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('brown')
nltk.download('universal_tagset')
from nltk.corpus import brown

feature_dict = {}
tag_dict = {}
pos_dict = {}  # previous 2
pos_uni_dict = {} # previous 1
pre_pos_dict = {} # previous + next

noun_the_dict = {}  # noun:[#the, #not the]
text_string = ""

rare_words = set()

def load_corpus(corpus_path):
    pairs = []
    text_string = open(corpus_path, 'r', encoding="utf-8", errors='ignore').read()
    with open(corpus_path, 'r', encoding="utf-8", errors='ignore') as file:
        text = file.read().split("\n\n")
    for data in text:
        if (data != "" and not data.startswith('<') and not data[0].isdigit()):
            pairs.append(data.replace("\n", " ").strip().lower())
    return pairs

def tokenize_sentence(snippet):
    return nltk.word_tokenize(snippet)

def detokenize_sentence(result):
    return TreebankWordDetokenizer().detokenize(result)

def pos_tag(tokenized_data):
    analysis = nltk.pos_tag(tokenized_data)
    return analysis

def create_pos_dict(token_tag_list):
    for i in range(len(token_tag_list)):
        pair = token_tag_list[i]
        if (pair[1] == "IN"):
            key = ""
            if (i == 0):
                key = "NULL-"
            else:
                key = token_tag_list[i-1][0] + "-"
            if (i == len(token_tag_list) - 1):
                key += "NULL"
            else:
                key += token_tag_list[i+1][0]
            if (pos_dict.get(key) == None):
                pos_dict[key] = {}
            if (pos_dict.get(key).get(pair[0]) != None):
                pos_dict.get(key)[pair[0]] += 1
            else:
                pos_dict.get(key)[pair[0]] = 1
    return

def create_pre_pos_dict(token_tag_list):
    for i in range(len(token_tag_list)):
        pair = token_tag_list[i]
        if (pair[1] == "IN"):
            key = ""
            if (i - 2 < 0):
                key = "NULL-"
            else:
                key = token_tag_list[i-2][0] + "-"
            if (i -1 < 0):
                key += "NULL"
            else:
                key += token_tag_list[i-1][0]               
            if (pre_pos_dict.get(key) == None):
                pre_pos_dict[key] = {}
            if (pre_pos_dict.get(key).get(pair[0]) != None):
                pre_pos_dict.get(key)[pair[0]] += 1
            else:
                pre_pos_dict.get(key)[pair[0]] = 1
    return

def create_uni_pos_dict(token_tag_list): 
    for i in range(len(token_tag_list)):
        pair = token_tag_list[i]
        if (pair[1] == "IN"):
            key = ""
            if (i - 1 < 0):
                key = "NULL"
            else:
                key = token_tag_list[i-1][0]
            if (pos_uni_dict.get(key) == None):
                pos_uni_dict[key] = {}
            if (pos_uni_dict.get(key).get(pair[0]) != None):
                pos_uni_dict.get(key)[pair[0]] += 1
            else:
                pos_uni_dict.get(key)[pair[0]] = 1
    return

def check_pos_dict(tokenized_data, token_tag_list): 
    for i in range(len(token_tag_list)):
        pair = token_tag_list[i]
        if (pair[1] == "IN"):
            key = ""
            if (i == 0):
                key = "NULL-"
            else:
                key = token_tag_list[i-1][0] + "-"
            if (i == len(token_tag_list) - 1):
                key += "NULL"
            else:
                key += token_tag_list[i+1][0]
            if (pos_dict.get(key) != None):
                if (pair[0] not in pos_dict.get(key)):
                    replace = ""
                    occ = 0
                    for cand in pos_dict.get(key):
                        if (pos_dict.get(key).get(cand) > occ):
                            occ = pos_dict.get(key).get(cand)
                            replace = cand
                    tokenized_data[i] = replace
    return


def check_pre_pos_dict(tokenized_data, token_tag_list): 
    for i in range(len(token_tag_list)):
        pair = token_tag_list[i]
        if (pair[1] == "IN"):
            key = ""
            if (i - 2 < 0):
                key = "NULL-"
            else:
                key = token_tag_list[i-2][0] + "-"
            if (i -1 < 0):
                key += "NULL"
            else:
                key += token_tag_list[i-1][0]
            if (pre_pos_dict.get(key) != None):
                if (pair[0] not in pre_pos_dict.get(key)):
                    replace = ""
                    occ = 0
                    for cand in pre_pos_dict.get(key):
                        if (pre_pos_dict.get(key).get(cand) > occ):
                            occ = pre_pos_dict.get(key).get(cand)
                            replace = cand
                    tokenized_data[i] = replace
    return

def check_uni_pos_dict(tokenized_data, token_tag_list):
    for i in range(len(token_tag_list)):
        pair = token_tag_list[i]
        if (pair[1] == "IN"):
            key = ""
            if (i - 1 < 0):
                key = "NULL"
            else:
                key = token_tag_list[i-1][0]
            if (pos_uni_dict.get(key) != None):
                if (pair[0] not in pos_uni_dict.get(key)):
                    replace = ""
                    occ = 0
                    for cand in pos_uni_dict.get(key):
                        if (pos_uni_dict.get(key).get(cand) > occ):
                            occ = pos_uni_dict.get(key).get(cand)
                            replace = cand
                    tokenized_data[i] = replace
    return

def prep_correction(sentence):
    backups = tokenize_sentence(sentence)
    tokenized_test = tokenize_sentence(sentence.lower())
    pos_t_t = pos_tag(tokenized_test)
    check_uni_pos_dict(tokenized_test, pos_t_t)
    for i in range(len(backups)):
        if (backups[i].lower() == tokenized_test[i]):
            tokenized_test[i] = backups[i]
    #check_pre_pos_dict(tokenized_test, pos_t_t)
    return detokenize_sentence(tokenized_test)

def create_noun_the_dict(tk_tg_list):
    if len(tk_tg_list) != 0:
        for i in range(1, len(tk_tg_list)):
            if tk_tg_list[i][1] == "NN":
                if tk_tg_list[i][0] not in noun_the_dict:
                    noun_the_dict[tk_tg_list[i][0]] = [0,1]
                else:
                    if tk_tg_list[i-1][0] == "the":
                        noun_the_dict[tk_tg_list[i][0]][0] += 1
                        noun_the_dict[tk_tg_list[i][0]][1] += 1
                    else:
                        noun_the_dict[tk_tg_list[i][0]][1] += 1
    return 0

def the_correction(str):
    str_lower = str.lower()
    tokens_original = tokenize_sentence(str)
    tokens = tokenize_sentence(str_lower)
    for i in range(len(tokens)):
        if (pos_tag([tokens[i]]))[0][1] == "NN":
            if noun_the_dict[tokens[i]][0] * 4 > noun_the_dict[tokens[i]][1] * 3:
                tokens_original[i] = "the " + tokens_original[i]

    return detokenize_sentence(tokens_original)

def adv_adj_correction(str):
    tokens = tokenize_sentence(str)
    for i in range(len(tokens) -1):
        if pos_tag([tokens[i]])[0][1] == "JJ" and pos_tag([tokens[i+1]])[0][1] == "JJ":
            if len(tokens[i])>=2 and tokens[i].endswith("ly"):
                tokens[i] = tokens[:len(tokens[i])-2] + "ily"
            elif len(tokens[i])>=2 and tokens[i].endswith("y"):
                tokens[i] = tokens[:len(tokens[i])-1] + "ily"
            elif len(tokens[i])>=2:
                if tokens[i].endswith("la") or tokens[i].endswith("le") or tokens[i].endswith("li") or tokens[i].endswith("lo") or tokens[i].endswith("lu"):
                    tokens[i] = tokens[i][:len(tokens[i]) - 2] + "ly"
                elif tokens[i].endswith("a") or tokens[i].endswith("e") or tokens[i].endswith("i") or tokens[i].endswith("o") or tokens[i].endswith("u"):
                    tokens[i] = tokens[i][:len(tokens[i])-1] + "ly"
                elif tokens[i].endswith("ll"):
                    tokens[i] = tokens[i] + "y"
                else:
                    tokens[i] = tokens[i] + "ly"

    return detokenize_sentence(tokens)

def verb_adv_correction(str):
    tokens = tokenize_sentence(str)
    for i in range(len(tokens) -1):
        if pos_tag([tokens[i]])[0][1].startswith("VB") and pos_tag([tokens[i+1]])[0][1] == "JJ":
            if len(tokens[i+1])>=2 and tokens[i+1].endswith("ly"):
                tokens[i+1] = tokens[i+1][:len(tokens[i+1])-2] + "ily"
            elif len(tokens[i+1])>=2 and tokens[i+1].endswith("y"):
                tokens[i+1] = tokens[i+1][:len(tokens[i+1])-1] + "ily"
            elif len(tokens[i+1])>=2:
                if tokens[i+1].endswith("la") or tokens[i+1].endswith("le") or tokens[i+1].endswith("li") or tokens[i+1].endswith("lo") or tokens[i+1].endswith("lu"):
                    tokens[i+1] = tokens[i+1][:len(tokens[i+1]) - 2] + "ly"
                elif tokens[i+1].endswith("a") or tokens[i+1].endswith("e") or tokens[i+1].endswith("i") or tokens[i+1].endswith("o") or tokens[i+1].endswith("u"):
                    tokens[i+1] = tokens[i+1][:len(tokens[i+1])-1] + "ly"
                elif tokens[i+1].endswith("ll"):
                    tokens[i+1] = tokens[i+1] + "y"
                else:
                    tokens[i+1] = tokens[i+1] + "ly"

    return detokenize_sentence(tokens)

def adv_verb_correction(str):
    tokens = tokenize_sentence(str)
    for i in range(len(tokens) -1):
        print(tokens[i])
        print(pos_tag([tokens[i]])[0][1])
        print(pos_tag([tokens[i+1]])[0][1])
        print("\n")
        if pos_tag([tokens[i]])[0][1] == "JJ" and pos_tag([tokens[i+1]])[0][1].startswith("VB"):
            if len(tokens[i])>=2 and tokens[i].endswith("ly"):
                tokens[i] = tokens[i][:len(tokens[i])-2] + "ily"
            elif len(tokens[i])>=2 and tokens[i].endswith("y"):
                tokens[i] = tokens[i][:len(tokens[i])-1] + "ily"
            elif len(tokens[i])>=2:
                if tokens[i].endswith("la") or tokens[i].endswith("le") or tokens[i].endswith("li") or tokens[i].endswith("lo") or tokens[i].endswith("lu"):
                    tokens[i] = tokens[i][:len(tokens[i]) - 2] + "ly"
                elif tokens[i].endswith("a") or tokens[i].endswith("e") or tokens[i].endswith("i") or tokens[i].endswith("o") or tokens[i].endswith("u"):
                    tokens[i] = tokens[i][:len(tokens[i])-1] + "ly"
                elif tokens[i].endswith("ll"):
                    tokens[i] = tokens[i] + "y"
                else:
                    tokens[i] = tokens[i] + "ly"
    return detokenize_sentence(tokens)

def load_test(filename):
    sentences = []
    with open(filename,'r') as file:
        sentenceFile = file.read().strip().split("\n")

    for sentence in sentenceFile:
        sentence.replace(",", "")
        sentence.replace(".", "")
        sentence.replace("!", "")
        sentence.replace('"', "")
        sentence.replace('?', "")
        if len(sentence) != 0 and not sentence.startswith("<"):
            sentences.append(sentence.split(" "))
    return sentences

def build_Y(sentences):
    Y = []      
    for sentence in sentences:
        correction = verb_adv_correction(sentence)
        correction = adv_verb_correction(correction)
        correction= adv_adj_correction(correction)
        correction = the_correction(correction)
        correction = prep_correction(correction)
        sentence_split = sentence.split(" ")
        correction_split = correction.split(" ")
        m = 0
        for i in range(len(correction_split)):
            word1 = correction_split[i]
            word2 = sentence_split[i-m]
            if word1 != word2:
                Y.append(1)
                if (word1 == "the"):
                    m += 1
            else:
                Y.append(0)
        for word in sentence:
            if word:
                Y.append(0)
            else:
                Y.append(1)
    return numpy.array(Y)

def build_X(sentences):
    X = []      
    for sentence in sentences:
        correction = verb_adv_correction(sentence)
        correction = adv_verb_correction(correction)
        correction= adv_adj_correction(correction)
        correction = the_correction(correction)
        correction = prep_correction(correction)
        for word in correction.split(" "):
            if word:
                X.append([0])
            else:
                X.append([1])
    return numpy.array(X)

def main(args):
    trainpath = args[0]
    testpath = args[1]
    g = load_corpus(trainpath)
    for x in g:
        m = tokenize_sentence(x)
        #y = tokenize_data(m)
        p = pos_tag(m)
        create_pos_dict(p)
        create_pre_pos_dict(p)
        create_uni_pos_dict(p)
        create_noun_the_dict(p)
        
    sentences_lists = load_test(testpath)
    X = build_X(sentences_lists)
    y= build_Y(sentences_lists)
    
    improve_model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
    improve_model.fit(X, y)
    
    filename = 'improve.model'
    pickle.dump(improve_model, open(filename, 'wb'))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))