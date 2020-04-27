import numpy as np

import sentiment

sentiment.main()


total_points = 0

# Load_corpus (4 points)
print('Testing load_corpus()')
try:
    loaded_corpus = sentiment.load_corpus('tiny-corpus.txt')
    snippets_correct = loaded_corpus[0][0] == 'positive snippet .' and loaded_corpus[1][0] == 'negative snippet .'
    labels_correct = loaded_corpus[0][1] == 1 and loaded_corpus[1][1] == 0
    if snippets_correct:
        if labels_correct:
            print('4 points')
            total_points += 4
        else:
            print('2 points: failed to convert labels to int')
            total_points += 2
    else:
        print('0 points: failed to read data')
except Exception as e:
    print(e)

# Tokenize (8 points)
print('Testing tokenize()')
try:
    snippet1 = sentiment.tokenize('hello \'world')
    snippet2 = sentiment.tokenize('hello world\'')
    snippet3 = sentiment.tokenize('don\'t hello')
    snippet4 = sentiment.tokenize('\'hello\' world')
    snippet5 = sentiment.tokenize('\'em world')

    points = 0
    if snippet1 == ['hello', '\'', 'world']:
        points += 2
    if snippet2 == ['hello', 'world', '\'']:
        points += 2
    if snippet3 == ['don\'t', 'hello']:
        points += 1
    if snippet4 == ['\'', 'hello', '\'', 'world']:
        points += 2
    if snippet5 == ['\'em', 'world']:
        points += 1
    total_points += points
    print('%d points' % points)
except Exception as e:
    print(e)

# Tag_edits (4 points)
print('Testing tag_edits()')
try:
    snippet1 = sentiment.tag_edits(['[hello]', 'world'])
    snippet2 = sentiment.tag_edits(['[hello', 'world]'])
    snippet3 = sentiment.tag_edits(['[hello', 'world'])

    points = 1
    if snippet1 == ['EDIT_hello', 'world']:
        points +=1
    if snippet2 == ['EDIT_hello', 'EDIT_world']:
        points += 1
    if snippet3 == ['EDIT_hello', 'EDIT_world']:
        points += 1
    total_points += points
    print('%d points' % points)
except Exception as e:
    print(e)

# Tag_negation (8 points)
print('Testing tag_negation()')
try:
    snippet1 = sentiment.tag_negation(['cannot', 'like', '.', 'lol'])
    snippet2 = sentiment.tag_negation(['not', 'good', 'but', 'EDIT_okay'])
    snippet3 = sentiment.tag_negation(['couldn\'t', 'be', 'clearer', 'that'])
    snippet4 = sentiment.tag_negation(['not', 'only', 'bad', 'never', 'worse'])

    points = 0
    if snippet1[1] == 'NOT_like' and snippet1[3] == 'lol':
        points += 2
    if snippet2[1] == 'NOT_good' and snippet2[4] == 'EDIT_okay':
        points += 2
    if snippet3[1] == 'NOT_be' and snippet3[3] == 'that':
        points += 2
    if snippet4[2] == 'bad' and snippet4[4] == 'NOT_worse':
        points += 2
    total_points += points
    print('%d points' % points)
except Exception as e:
    print(e)

# Get_features (8 points)
print('Testing get_features()')
try:
    features1 = sentiment.get_features([('i', '-'), ('i', '-'), ('the', '-')])
    features2 = sentiment.get_features([('i', '-'), ('EDIT_am', '-'), ('a', '-')])
    points = 0
    if np.sum(features1) == 3:
        points += 4
    if np.sum(features2) == 2:
        points += 4
    total_points += points
    print('%d points' % points)
except Exception as e:
    print(e)

# Normalize (4 points)
print('Testing normalize()')
try:
    test_matrix = np.array([[1, 2], [3, 4]], dtype='float')
    normalized = sentiment.normalize(test_matrix)
    if np.array_equal(test_matrix, np.array([[0, 0], [1, 1]], dtype='float')):
        total_points += 4
        print('4 points')
    else:
        print('0 points')
except Exception as e:
    print(e)

# Evaluate_predictions (8 points)
print('Testing evaluate_predictions()')
try:
    p, r, f = sentiment.evaluate_predictions(
        np.array([0, 1, 0, 1]), 
        np.array([0, 1, 1, 0]))
    points = 0
    if p == 0.5:
        points += 3
    if r == 0.5:
        points += 3
    if f == 0.5:
        points += 2
    total_points += points
    print('%d points' % points)
except Exception as e:
    print(e)

# Load_dal (4 points)
print('Testing load_dal()')
try:
    dal = sentiment.load_dal('dict_of_affect.txt')
    points = 0
    if dal['a'] == ('-1.380115', '0.132305', '-2.170380'):
        points += 2
    elif dal['a'] == (-1.380115, 0.132305, -2.170380):
        points += 4
    if not 'Word' in dal:
        points -= 1
    total_points += points
    print('%d points' % points)
except Exception as e:
    print(e)

# Score_snippet (4 points)
print('Testing score_snippet')
try:
    dal = sentiment.load_dal('dict_of_affect.txt')
    a1, e1, i1 = sentiment.score_snippet([('abandon', 'VB')], dal)
    if a1 == 1.786400 and e1 == -3.644431 and i1 == 0.538300:
        total_points += 4
        print('4 points')
    else:
        print('0 points')
except Exception as e:
    print(e)

# Score_snippet part 2 (5 points)
print('Testing score_snippet again')
try:
    dal = sentiment.load_dal('dict_of_affect.txt')
    a2, e2, i2 = sentiment.score_snippet([('astounded', 'JJ')], dal)
    if a2 == 2.080232 and e2 == 2.645480 and i2 == 0.010639:
        total_points += 5
        print('5 points')
    else:
        print('0 points')
except Exception as e:
    print(e)

print("#"*20)   
print("total_points:", total_points)
print("deducted_points:",57-total_points)
print("#"*20)  