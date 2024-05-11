'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np

def joint_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X1=m,X2=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    '''
    num_texts = len(texts)
    word0_count = np.zeros(num_texts)
    word1_count = np.zeros(num_texts)
    max_m = 0
    max_n = 0
    for i in range(num_texts):
      for word in texts[i]:
        if word0 == word:
          word0_count[i] += 1
        if word1 == word:
          word1_count[i] += 1
      max_m = max(word0_count[i], max_m)
      max_n = max(word1_count[i], max_n)
    Pjoint = np.zeros((int(max_m+1), int(max_n+1)))

    for i in range(num_texts):
      Pjoint[int(word0_count[i])][int(word1_count[i])] += 1/num_texts
    return Pjoint

def marginal_distribution_of_word_counts(Pjoint, index):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other) 

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    '''
    if index == 0:
      Pmarginal = np.zeros(len(Pjoint))
      for i in range(len(Pjoint)):
        Pmarginal[i] = sum(Pjoint[i])
      return Pmarginal

    Pmarginal = np.zeros(len(Pjoint[0]))
    for i in range(len(Pjoint[0])):
      Pmarginal[i] = sum(Pjoint[:,i])
    return Pmarginal
    
def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs: 
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    '''
    Pcond = Pjoint.copy()
    for i in range(len(Pjoint)):
      for j in range(len(Pjoint[0])):
        Pcond[i][j] = Pcond[i][j] / Pmarginal[i]
    return Pcond

def mean_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    mu (float) - the mean of X
    '''
    mu = 0
    for i in range(len(P)):
      mu += i*P[i]
    return mu

def variance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    var (float) - the variance of X
    '''
    var = 0
    mu = mean_from_distribution(P)
    for i in range(len(P)):
      var += P[i]*((i-mu)**2)
    return var

def covariance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)
    
    Outputs:
    covar (float) - the covariance of X0 and X1
    '''
    P0 = marginal_distribution_of_word_counts(P, 0)
    P1 = marginal_distribution_of_word_counts(P, 1)

    mu0 = mean_from_distribution(P0)
    mu1 = mean_from_distribution(P1)

    covar = 0
    for i in range(len(P0)):
      for j in range(len(P1)):
        covar += P[i][j]*(i-mu0)*(j-mu1)
    return covar

def expectation_of_a_function(P, f):
    '''
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    '''
    expected = 0
    for i in range(len(P)):
      for j in range(len(P[0])):
        expected += f(i,j) * P[i][j]
    return expected
    
