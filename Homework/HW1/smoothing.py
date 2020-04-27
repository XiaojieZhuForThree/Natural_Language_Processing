# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:06:30 2019

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
                
    def word_prob(self, word, context, delta = 0):
#        we need to replace the unseen word with the token "<unk>"
        dupWord = word
        dupContext = ()
        if (word not in self.vocabulary):
            dupWord = "<unk>"
        for i in context:
            if i not in self.vocabulary:
                dupContext += ("<unk>",)
            else:
                dupContext += (i,)
                
#        if delta = 0, we still use the old method
        if (delta == 0 and dupContext not in self.context_counts):
            return 1/len(self.vocabulary)
#        the change part: we now use Laplace smoothing to compute the probability
#        we also use dictionary.get() function to return 0 if the key is not found
        divident = self.ngram_counts.get((dupWord, dupContext), 0) + delta
        divisor = self.context_counts.get(dupContext, 0) + delta*len(self.vocabulary)
        return divident / divisor

def create_ngramlm(n, corpus_path):
#    we create a NGramLM model and train it with the data in file corpos_path
    model = NGramLM(n)
#    this is how we process the data, simply split the 
#    sentence with '\n', then split the word with ' '
    with open(corpus_path, 'r') as file:
#        but also this time we use the mask to process the data
        copy = mask_rare(file.read())
        text = copy.split('\n')
        
    for sentence in text:
#        we might encounter empty sentences, and we should ignore them
        if sentence != "":
            model.update(sentence.split(" "))
    return model

#we here add a variable delta so we can convenient change the delta values on the
#same model
def text_prob(model, text, delta = 0):
    ans = 0
    for word, context in get_ngrams(model.n, text):
#        add up the log probability of each ngram in the text
        ans += math.log(model.word_prob(word, context, delta))
    return ans



def mask_rare(corpus):
#    we use a dictionary to keep the counts of each word
    vocabulary = {}
#    we use the same method to process the data
    copy = corpus.split("\n")
    for i in copy:
        if i != "":
            sentence = i.split(" ")
            for word in sentence:
                vocabulary[word] = vocabulary.get(word, 0) + 1 
                
#    we then replace each word occurred only once by "<unk>"    
    for i in range(len(copy)):
        if copy[i] != "":
            sentence = copy[i].split(" ")
            for j in range(len(sentence)):
                if vocabulary[sentence[j]] == 1:
                    sentence[j]= "<unk>"
            copy[i] = " ".join(sentence)
    return "\n".join(copy)

# train the model and test the two sentences
model = create_ngramlm(3, r"warpeace.txt")

sentence1 = "God has given it to me, let him who touches it beware!".split(" ")
sentence2 = "Where is the prince, my Dauphin?".split(" ")

# different delta values
deltas = [0.001, 0.01, 0.05, 0.5, 0.75, 1]
print("the sentence probability of sentence1 on the model without smoothing is: ")
print(text_prob(model, sentence1))
print("\n")
for i in deltas:
    print("the text probability for sentence1 " + "with smoothing delta =", i , "is:", text_prob(model, sentence1, i))
    
print("\n\nthe sentence probability of sentence2 on the model without smoothing is: ")
print(text_prob(model, sentence2))
for i in deltas:
    print("the text probability for sentence2 " + "with smoothing delta =", i , "is:", text_prob(model, sentence2, i))