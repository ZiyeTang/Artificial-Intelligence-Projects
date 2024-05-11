'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition_matrix(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    m = model.M
    n = model.N
    trans_mtx = np.zeros((m,n,4,m,n))
    for i in range(m):
        for j in range(n):
            if model.T[i][j]:
                continue
            if i+1 >= m or model.W[i+1, j]:
                trans_mtx[i,j,0,i,j] += model.D[i,j,1]
                trans_mtx[i,j,2,i,j] += model.D[i,j,2]
                trans_mtx[i,j,3,i,j] += model.D[i,j,0]
            else:
                trans_mtx[i,j,0,i+1,j] += model.D[i,j,1]
                trans_mtx[i,j,2,i+1,j] += model.D[i,j,2]
                trans_mtx[i,j,3,i+1,j] += model.D[i,j,0]
            
            if i-1 < 0 or model.W[i-1,j]:
                trans_mtx[i,j,0,i,j] += model.D[i,j,2]
                trans_mtx[i,j,1,i,j] += model.D[i,j,0]
                trans_mtx[i,j,2,i,j] += model.D[i,j,1]
            else:
                trans_mtx[i,j,0,i-1,j] += model.D[i,j,2]
                trans_mtx[i,j,1,i-1,j] += model.D[i,j,0]
                trans_mtx[i,j,2,i-1,j] += model.D[i,j,1]
                
                
            if j+1 >= n or model.W[i,j+1]:
                trans_mtx[i,j,1,i,j] += model.D[i,j,2]
                trans_mtx[i,j,2,i,j] += model.D[i,j,0]
                trans_mtx[i,j,3,i,j] += model.D[i,j,1]
            else:
                trans_mtx[i,j,1,i,j+1] += model.D[i,j,2]
                trans_mtx[i,j,2,i,j+1] += model.D[i,j,0]
                trans_mtx[i,j,3,i,j+1] += model.D[i,j,1]
            
            if j-1 < 0 or model.W[i,j-1]:
                trans_mtx[i,j,0,i,j] += model.D[i,j,0]
                trans_mtx[i,j,1,i,j] += model.D[i,j,1]
                trans_mtx[i,j,3,i,j] += model.D[i,j,2]
            else:
                trans_mtx[i,j,0,i,j-1] += model.D[i,j,0]
                trans_mtx[i,j,1,i,j-1] += model.D[i,j,1]
                trans_mtx[i,j,3,i,j-1] += model.D[i,j,2]
    return trans_mtx

def update_utility(model, P, U_current):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    m = U_current.shape[0]
    n = U_current.shape[1]
    U_next = np.zeros((m,n))
    
    for i in range(m):
        for j in range(n):
            mxu = 0
            for a in range(4):
                mxu = max(mxu, np.sum(P[i,j,a,:,:] * U_current))
            U_next[i,j] = model.R[i,j] + model.gamma * mxu

    return U_next

    

def value_iteration(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    epsilon = 1e-3
    trans_mtx = compute_transition_matrix(model)

    U = np.zeros((model.M,model.N))
    U_nxt = update_utility(model, trans_mtx, U)
    
    while np.sum(abs(U-U_nxt)<epsilon) < model.M * model.N:
        U = U_nxt
        U_nxt = update_utility(model, trans_mtx, U)
    return U



if __name__ == "__main__":
    import utils
    model = utils.load_MDP('models/small.json')
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
