'''
    NumPy and The Matrix
'''

import numpy as np

def task1(matrix):
    '''i,j= np.shape(matrix)
    for k in range(i):
        for l in range(j):
            if k>l:
                matrix[k][l]=0'''
    
    upper=np.triu(matrix)
    return np.transpose(upper)

def task2(matrix):
    '''return mean, median, std (precision 2), all along x, determinant, inverse, pseudo-inverse'''
    mean = np.mean(matrix,axis=0)
    median = np.median(matrix,axis=0,overwrite_input=False,keepdims=False,out=None)
    std = np.std(matrix,axis=0)
    det = np.linalg.det(matrix)
    if(det!=0):
        inv=np.linalg.inv(matrix)
    else:
        inv=None
    pseudoinv=np.linalg.pinv(matrix)

    return mean, median, std, det, inv, pseudoinv

def task3(matrix, num = 0, padding = 3):
    padded=np.pad(matrix,pad_width=padding,mode='constant',constant_values=num)
    return padded

if __name__ == '__main__':

    matrix = np.array([
        [5,5,84,3,9],
        [6,11,1,55,58],
        [1,20,48,12,36],
        [8,4,41,93,98],
        [6,17,64,0,13]
    ])

    # you can call the functions here
    # Uncomment the following lines to test your code

    # TASK 1
    #print(task1(matrix))

    # TASK 2
    #mean, median, std, det, inv, pseudoinv = task2(matrix)
    #print("Mean: ", mean)
    #print("Median: ", median)
    #print("Standard Deviation: ", std)
    #print("Determinant: ", det)
    #print("Inverse: ", inv)
    #print("Pseudo-Inverse: ", pseudoinv)

    # TASK 3
    #print(task3(matrix)) # default padding
