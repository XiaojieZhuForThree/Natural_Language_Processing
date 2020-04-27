# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:36:59 2019

@author: xxz180012
"""
import math
   
# the generator function to provide the (word, context) data
def get_ngrams(n, text):
    
#    pad text with enough start tokens '<s>' to be able to make n-grams 
#    for the first n-1 words   
    
#    also add stop token '</s>'
    dup = ["<s>" for i in range(n-1)] + text + ["</s>"]
    num, total = n-1, len(dup)
    for num in range(n-1, total):
        yield((dup[num], tuple(dup[num - n + 1 : num])))


class NGramLM:    
    def __init__(self, n):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        # we use the set to save the vocabulary, since vocabulary contains no
        # duplicates
        self.vocabulary = set(["<s>"])
        
    def update(self, text):
        for word, context in get_ngrams(self.n, text):
#            for each n_gram, we add the ngram, the word, 
#            and the context up to 1 in the internal data structure
            self.ngram_counts[(word, context)] = self.ngram_counts.get((word, context), 0) + 1
            self.context_counts[context] = self.context_counts.get(context, 0) + 1
            self.vocabulary.add(word)
                
    def word_prob(self, word, context):
        if (context not in self.context_counts):
            return 1/len(self.vocabulary)
        return self.ngram_counts[(word, context)]/self.context_counts[context]

def create_ngramlm(n, corpus_path):
#    we create a NGramLM model and train it with the data in file corpos_path
    model = NGramLM(n)
#    this is how we process the data, simply split the 
#    sentence with '\n', then split the word with ' '
    with open(corpus_path, 'r') as file:
        text = file.read().split('\n')

    for sentence in text:
#        we might encounter empty sentences, and we should ignore them
        if sentence != "":
            model.update(sentence.split(" "))
    return model

def text_prob(model, text):
    ans = 0
    for word, context in get_ngrams(model.n, text):
#        add up the log probability of each ngram in the text
        ans += math.log(model.word_prob(word, context))
    return ans

# train the model and test the two sentences
model = create_ngramlm(3, r"warpeace.txt")

sentence1 = "God has given it to me, let him who touches it beware!".split(" ")
sentence2 = "Where is the prince, my Dauphin?".split(" ")

print("the sentence probability of sentence1 on the model is: ")
print(text_prob(model, sentence1))

print("\nthe sentence probability of sentence2 on the model is: ")
print(text_prob(model, sentence2))







