import nltk
import re
import sys


# Fill in the pattern (see Part 2 instructions)
NP_grammar = 'NP: {<DT>?<JJ.*>*<NN.*>+}' 


# Fill in the other 4 rules (see Part 3 instructions)
hearst_patterns = [
    ('((NP_\w+ ?(, )?)+(and |or )?other NP_\w+)', 'after'),
    ('(NP_\w+ (, )?such as (NP_\w+ ? (, )?(and |or )?)+)', 'before'),
    ('(such NP_\w+ (, )?as (NP_\w+ ?(, )?(and |or )?)+)', 'before'),
    ('(NP_\w+ (, )?include (NP_\w+ ?(, )?(and |or )?)+)', 'before'),
    ('(NP_\w+ (, )?especially (NP_\w+ ?(, )?(and |or )?)+)', 'before')]


# Fill in the function (see Part 1 instructions)
# Argument type: path - string
# Return type: list of tuples
def load_corpus(path):
    sentences = []
    with open(path,'r',encoding="utf8") as file:
        sentenceFile = file.read().splitlines()
    for sentence in sentenceFile:
        sent, lemma = sentence.split("\t")
        sentList = [i.strip() for i in sent.split()]
        lemmaList = [j.strip() for j in lemma.split()]
        sentTup = (sentList, lemmaList)
        sentences.append(sentTup)
    return sentences


# Fill in the function (see Part 1 instructions)
# Argument type: path - string
# Return type: tuple of sets
def load_test(path):
    trueSet = set()
    falseSet = set()
    with open(path,'r',encoding="utf8") as file:
        sentenceFile = file.read().splitlines()
    for sentence in sentenceFile:
        hypo, hyper, label = sentence.split("\t")
        if label == "True":
            trueSet.add((hypo.strip(), hyper.strip()))
        elif label == "False":
            falseSet.add((hypo.strip(), hyper.strip()))
    return (trueSet, falseSet)


# Fill in the function (see Part 2 instructions)
# Argument type: sentence, lemmatized - list of strings; parser - nltk.RegexpParser
# Return type: string
def chunk_lemmatized_sentence(sentence, lemmatized, parser):
    sentence = nltk.pos_tag(sentence)
    tagLemma = []
    for i in range(len(lemmatized)):
        token, tag = lemmatized[i], sentence[i][1]
        tagLemma.append((token, tag))
    lemmatized = tagLemma
    lemmaTree = parser.parse(lemmatized)
    lemmaChunks = tree_to_chunks(lemmaTree)
    lemmaString = merge_chunks(lemmaChunks)
    return lemmaString

# Fill in the function (see Part 2 instructions)
# Helper function for chunk_lemmatized_sentence()
# Argument type: tree - nltk.Tree
# Return type: list of strings
def tree_to_chunks(tree):
    chunks = []
    for child in tree:
        if not isinstance(child, nltk.Tree):
            chunks.append(child[0])
        else:
            tokens = [subchild[0] for subchild in child]
            tstring = "_".join(tokens)
            tstring = "NP_" + tstring
            chunks.append(tstring)
    return chunks

# Fill in the function (see Part 2 instructions)
# Helper function for chunk_lemmatized_sentence()
# Argument type: chunks - list of strings
# Return type: string
def merge_chunks(chunks):
    buffer = []
    for chunk in chunks:
        if len(buffer) != 0 and buffer[-1].startswith("NP_") and chunk.startswith("NP_"):
            newChunk = buffer.pop() + chunk[2:]
            buffer.append(newChunk)
        else:
            buffer.append(chunk)
    return " ".join(buffer)


# Fill in the function (see Part 4 instructions)
# Argument type: chunked_sentence - string
# Yield type: tuple
def extract_relations(chunked_sentence):
    for pattern, pos in hearst_patterns:
        match = re.search(pattern, chunked_sentence)
        if match:
            text = match.group(0)
            tokenizedText = text.split()
            sievedTokens = [token for token in tokenizedText if token.startswith("NP_") ]
            postprocessedTokens = postprocess_NPs(sievedTokens)
            if pos == "before":
                hyper = postprocessedTokens[0]
                postprocessedTokens = postprocessedTokens[1:]
                for hypo in postprocessedTokens:
                    yield((hypo, hyper))
            elif pos == "after":
                hyper = postprocessedTokens.pop()
                for hypo in postprocessedTokens:
                    yield((hypo, hyper))

# Fill in the function (see Part 4 instructions)
# Helper function for extract_relations()
# Argument type: list of strings
# Return type: list of strings
def postprocess_NPs(NPs):
    newTokens = []
    for NP in NPs:
        newTokens.append(NP.replace("NP_", "").replace("_", " "))
    return newTokens


# Fill in the function (see Part 5 instructions)
# Argument type: extractions, gold_true, gold_false - set of tuples
# Return type: tuple
def evaluate_extractions(extractions, gold_true, gold_false):
    tp, fp, fn = 0, 0, 0
    for pair in extractions:
        if pair in gold_true:
            tp += 1
        elif pair in gold_false:
            fp += 1
    for gold_pair in gold_true:
        if gold_pair not in extractions:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = 2*((precision * recall) / (precision + recall))    
    return (precision, recall, fmeasure)


def main(args):
    corpus_path = args[0]
    test_path = args[1]

    wikipedia_corpus = load_corpus(corpus_path)
    test_true, test_false = load_test(test_path)

    NP_chunker = nltk.RegexpParser(NP_grammar)

    # Complete the line (see Part 2 instructions)
    wikipedia_corpus = [chunk_lemmatized_sentence(pair[0], pair[1], NP_chunker) for pair in wikipedia_corpus]

    extracted_pairs = set()
    for chunked_sentence in wikipedia_corpus:
        for pair in extract_relations(chunked_sentence):
            extracted_pairs.add(pair)

    print('Precision: %f\nRecall:%f\nF-measure: %f' % evaluate_extractions(extracted_pairs, test_true, test_false))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
