# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 23:06:01 2019

@author: xxz180012
"""

import math
import random
random.seed(1)

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
    
    def random_word(self, context, delta = 0):
#        sort the vocabulary, since we used set, we need to convert it to list
        pool = sorted(list(self.vocabulary))
#        generate a random value between [0,1)
        r = random.random()
#        assign the initial probability to 0, add each word's probability, then 
#        return the word soon as the prob is greater than r
        prob = 0
        for word in pool:
            prob += self.word_prob(word, context, delta)
            if r < prob:
                return word
            
    def likeliest_word(self, context, delta = 0):
        maxProb = 0 
        ngram = () # the target ngram to return
        
#        we iterate all words in vocabulary, and compute the probability of each word
#        under the context, then pick the largest one
        for word in self.vocabulary:
            prob = self.word_prob(word, context, delta)
#            keep the largest probability and the ngram
            if maxProb < prob:
                maxProb = prob
                ngram = word, context
        return ngram

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

class NGramInterpolator:
    def __init__(self, n, lambdas):
        self.n = n
        self.lambdas = lambdas
#        we use a list to save the NGramLMs
        self.NGramLMs = []
#        we create the NGramLM in descending orders, 
#        since the lambdas are also in descending orders
        for i in range(n, 0, -1):
            self.NGramLMs.append(NGramLM(i))

    def update(self, text):
#        update all of the internal NGramLMs
        for NGramLM in self.NGramLMs:
            NGramLM.update(text)

    def word_prob(self, word, context, delta = 0):
        prob = 0
#        use a zip function to match each lambda with each NGramLm
        for NGramLM, lam in zip(self.NGramLMs, self.lambdas):
            prob += lam * NGramLM.word_prob(word, context, delta)
        return prob

# a funciton to create and train a NGramInterpolator with lambdas and traning data
def create_NGramInterpolator(n, lambdas, corpus_path):
    model = NGramInterpolator(n, lambdas)
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


# a funciton to calculate the perplexity of the model against a test data
def perplexity(model, corpus_path, delta = 0):
    I = 0   #the entropy
    N = 0   # to record the total tokens
#    this is how we process the data, simply split the 
#    sentence with '\n', then split the word with ' '    
    with open(corpus_path, 'r') as file:
#        we do not need to mask the rare word for test data
        data = file.read().split("\n")
    for test in data:
#        we might encounter empty sentences, and we should ignore them        
        if test != "":
            text = test.split(" ")
            N += len(text)
#            the calculation of perplexity is to sum up the log probability of each sentence
#            and we can use text_prob function to help us do the work
            I += text_prob(model, text, delta)
#    we will use the average log to compute our result
    aveLog = I/N
#    since we used e as the base, we also need to use e to compute the exponential
    PP = math.exp(-aveLog)
    return PP


# the function to generate the sentence composed of randomly select words
def random_text(model, max_length, delta = 0):
#    initial context
    context = ["<s>" for i in range(model.n-1)]
#    used to store the words
    words = []
    while (len(words) < max_length):
        newWord = model.random_word(tuple(context), delta)
        words.append(newWord)
#        return the generated string immediately if the stop token '</s>' is generated.
        if newWord == '</s>':
            return " ".join(words)
#        update the context
        context = context[1:] + [newWord]
    return " ".join(words)

# the function to generate the sentence composed of the likeliest words
def likeliest_text(model, max_length, delta = 0):
#    initial context
    context = ["<s>" for i in range(model.n-1)]
#    used to store the words
    words = []
    while (len(words) < max_length):
        newWord = model.likeliest_word(tuple(context), delta)[0]
        words.append(newWord)
#        return the generated string immediately if the stop token '</s>' is generated.
        if newWord == '</s>':
            return " ".join(words)
#        update the context
        context = context[1:] + [newWord]
    return " ".join(words)



# train the model and generate the sentence
model = create_ngramlm(3, r"shakespeare.txt")
print("The five sentences generated by the randomly picked words: ")
for i in range(5):
    print(random_text(model, 10))
     
#Train four models - one bigram, one trigram, one 4-gram, and one 5-gram - on shakespeare.txt
bigram = create_ngramlm(2, r"shakespeare.txt")
trigram = create_ngramlm(3, r"shakespeare.txt")
fourgram = create_ngramlm(4, r"shakespeare.txt")
fifgram = create_ngramlm(5, r"shakespeare.txt")

print("\nThe sentences generated by the bigram model: ")
print(likeliest_text(bigram, 10))
print("\nThe sentences generated by the trigram model: ")
print(likeliest_text(trigram, 10))
print("\nThe sentences generated by the 4-gram model: ")
print(likeliest_text(fourgram, 10))
print("\nThe sentences generated by the 5-gram model: ")
print(likeliest_text(fifgram, 10))