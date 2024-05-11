'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    dict= {}
    sol = {}
    num = {}
    
    tag_num = {}
    for pairs in train:
        for p in pairs:
            if p[0] not in dict.keys():
                dict[p[0]] = {}     
            if p[1] not in dict[p[0]].keys():
                dict[p[0]][p[1]] = 0
            
            dict[p[0]][p[1]] += 1
            
            if p[0] not in sol.keys() or num[p[0]] < dict[p[0]][p[1]]:
                sol[p[0]] = p[1]
                num[p[0]] = dict[p[0]][p[1]]
            
            if p[1] not in tag_num.keys():
                tag_num[p[1]] = 0
            tag_num[p[1]] += 1
     
    max_tag = train[0][0][1]
    for k in tag_num.keys():
        if tag_num[k] > tag_num[max_tag]:
            max_tag = k
    
    res = []
    for sentence in test:
        sent_res = []
        for word in sentence:
            if word in sol.keys():
                sent_res.append((word,sol[word]))
            else:
                sent_res.append((word, max_tag))
        res.append(sent_res)
            
    return res



def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    p_word_tag = {}
    p_tag_prev = {}
    p_tag = {}
    size = 0
    for pairs in train:
        prev = ""
        for p in pairs:
            if (p[0],p[1]) not in p_word_tag.keys():
                p_word_tag[(p[0],p[1])] = 0  
            p_word_tag[(p[0],p[1])] += 1

            if p[1] not in p_tag.keys():
                p_tag[p[1]] = 0
            p_tag[p[1]] += 1

            if prev != "":
                if (p[1], prev) not in p_tag_prev.keys():
                    p_tag_prev[(p[1], prev)] = 0
                p_tag_prev[(p[1], prev)] += 1
            
            prev = p[1]
            
            size += 1
    
    
    for p in p_word_tag.keys():
        p_word_tag[p] = log(p_word_tag[p] / p_tag[p[1]])
    p_word_tag["UN"] = log(1 / size)

    for p in p_tag_prev.keys():
        p_tag_prev[p] = log(p_tag_prev[p] / p_tag[p[1]])
    p_tag_prev["UN"] = log(1 / size)
    
    for t in p_tag.keys():
        p_tag[t] = log(p_tag[t] / size)

    # print(p_tag)
    # print(p_word_tag)
    # print(p_tag_prev)

    res = []
    for sentence in test:
        sent_res = []
        d = len(sentence)
        v = []
        psi = []
        for i in range(d):
            v.append({})
            psi.append({})
        
        for i in p_tag.keys():
            if (sentence[0], i) in p_word_tag.keys():
                v[0][i] = p_tag[i] + p_word_tag[(sentence[0], i)]
            else:
                v[0][i] = p_tag[i] + p_word_tag["UN"]
        
        for t in range(1,d):
            for j in p_tag.keys():
                v[t][j] = 0
                besti = ""
                bestv = 0
                for i in p_tag.keys():
                    if (j,i) in p_tag_prev.keys() and (sentence[t], j) in p_word_tag.keys():
                        val = v[t-1][i] + p_tag_prev[(j,i)] + p_word_tag[(sentence[t], j)]
                    elif (j,i) in p_tag_prev.keys() and (sentence[t], j) not in p_word_tag.keys():
                        val = v[t-1][i] + p_tag_prev[(j,i)] + p_word_tag["UN"]
                    elif (j,i) not in p_tag_prev.keys() and (sentence[t], j) in p_word_tag.keys():
                        val = v[t-1][i] + p_tag_prev["UN"] + p_word_tag[(sentence[t], j)]
                    else:
                        val =  v[t-1][i] + p_tag_prev["UN"] + p_word_tag["UN"]
                    
                    if besti == "" or val > bestv:
                        bestv = val
                        besti = i
                        #print(besti)
                        
                
                v[t][j] = bestv
                psi[t][j] = besti
                
            
        y = ""
        bestv = 0
        for i in p_tag.keys():
            if y == "" or bestv < v[d-1][i]:
                bestv = v[d-1][i]
                y = i
            
        ys = [y]
        for t in range(d-2, -1, -1):
            ys.insert(0, psi[t+1][ys[0]])

        for i in range(d):
            sent_res.append((sentence[i],ys[i]))
        
        res.append(sent_res)

    return res


def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")



