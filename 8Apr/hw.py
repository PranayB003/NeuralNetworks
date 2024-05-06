import numpy as np

def max_ratio(u,v):
    '''
    This function takes u and v as input and checks finds all non-zero entries and
    does element-wise division on those indices,it returns the maximum ratio and
    the position at which it occurs.
    In the case of all elements being zero in the 2nd vector,a log regarding the same
    is done.
    '''
    u=np.array(u) 
    v=np.array(v) 
    # Indices of v containing non-zero elements
    nz=np.array(np.nonzero(v)) 
    # Handle edge case where all elements of v are zero
    if(nz.shape[1]==0):
        print("Either the 2nd vector is empty or all elements are zero")
        return (-1,-1)
    # Get the index where maximum ratio occurs
    ind=nz[0,np.argmax(u[nz[0]]/v[nz[0]])]
    return u[ind]/v[ind],ind

def make_zero(M,l): 
    '''
    This function takes M and l as inputs and removes the elements from l which are
    out of bounds for M and then makes those particular columns zero.
    '''
    l=np.array(l)
    M=np.array(M)
    # Remove those indices from l which are out-of-bounds for M
    ind=np.argwhere(l>=M.shape[1])
    l=np.delete(l,ind)
    # Modify the matrix M in-place and return it
    M[:,l]=0
    return M

if __name__ == "__main__":
    print("Q1, Test-1")
    print(max_ratio([1,2,3], [0,1,2]), end="\n\n")

    print("Q1, Test-2")
    print(max_ratio([1,2,3], [0,0,0]), end="\n\n")

    print("Q2, Test-1")
    print(make_zero([[0,1,2,3],[2,3,4,4]], [0,2]), end="\n\n")

    print("Q2, Test-2")
    print(make_zero([[0,1,2,3],[2,3,4,4]], [0,4]), end="\n\n")
