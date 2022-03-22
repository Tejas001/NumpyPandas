from matplotlib.pyplot import axis
import numpy as np
from numpy.random import randint as ri
import matplotlib.pyplot as plt
# l = [1,2,3]
# ar = np.array(l)
# print(f'Array is {ar} and its type is {type(ar)}')

# l = [[1,2,3],[4,5,6],[7,8,9]]
# ar = np.array(l)
# print(ar, type(ar))
# print(ar.ndim,sep='')
# print(ar.size,sep='')
# print(ar.shape,sep='')
# print(ar.dtype,sep='',)

# l1 = [[1,2.3,3],[4,5.5,6],[7.1,8,9]]
# ar1 = np.array(l1)
# print(ar1, type(ar),sep='')
# print(ar1.dtype,sep='')

# print(np.array(([1,2,3],[4,5,6],[7,8,9])))

# arange and linspace
# print('series', np.arange(1,10))
# print('series', np.arange(0,10,2))
# print('series', np.arange(0,10,2.5))
# print('series', np.arange(20,-1,-2))
# print('linearly spaced ', np.linspace(1,2,5))

# Zero, ones, empty, identity matrix
# print('Zeros',np.zeros(5))
# print('Zeros',np.zeros((4,5)))

# print('Ones',np.ones(4))
# print('Ones',np.ones((2,3)))
# print('Ones',2*np.ones((2,2)))

# print("Empty matrix\n-------------\n", np.empty((3,5)))

# Random number generation
# print('Random',np.random.rand(2,3))
# print(np.random.rand(4,3))
# print(np.random.randint(1,100,10))
# print(np.random.randint(1,100,(2,4)))
# print(np.random.randint(1,5,10))

# Reshaping, min, max, sort
# a = ri(1,100,30)
# b = a.reshape(2,3,5)
# c = a.reshape(6,5)
# print("\na looks like\n",'-'*20,"\n",a,"\n",'-'*20,a.shape,"\n",'-'*20)
# print("\nb looks like\n",'-'*20,"\n",b,"\n",'-'*20,b.shape,"\n",'-'*20)
# print("\nc looks like\n",'-'*20,"\n",c,"\n",'-'*20,c.shape,"\n",'-'*20)
# print(a.max())
# print(b.max())
# print(a.argmax())
# print(b.argmax())
# print(c.argmax())

# A = ri(1,100,10)
# print(A)
# print('-'*20)
# print(np.sort(A,kind='mergesort'))
# print('-'*20)

# Indexing and slicing
# a = np.arange(1,10)
# print('Array',a)
# print(a[2])
# print(a[::-1])
# print(a[1:7])
# print(a[-1:-7:-2])
# print(a[[2,3,5]])

# mat = np.array(ri(1,20,15)).reshape(3,5)
# print(mat)
# print(mat[1,1])
# print(mat[0,1])
# print(mat[1:3,3:5])
# print(mat[mat>10])

# mat = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(mat)
# mat_slice = mat[:2,:2]
# print(mat_slice)
# mat_slice[0,0] = 100
# print(mat_slice)
# print(mat)

# mat = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(mat)
# mat_slice = np.array(mat[:2,:2])
# print(mat_slice)
# mat_slice[0,0] = 100
# print(mat_slice)
# print(mat)

# Array operations (array-array, array-scalar, universal functions)
# mat1 = np.array(ri(1,10,9)).reshape(3,3)
# mat2 = np.array(ri(1,10,9)).reshape(3,3)
# print("\n1st Matrix of random single-digit numbers\n----------------------------------------\n",mat1)
# print("\n2nd Matrix of random single-digit numbers\n----------------------------------------\n",mat2)

# print("\nAddition\n------------------\n", mat1+mat2)
# print("\nMultiplication\n------------------\n", mat1*mat2)
# print("\nDivision\n------------------\n", mat1/mat2)
# print("\nLineaer combination: 3*A - 2*B\n-----------------------------\n", 3*mat1-2*mat2)

# print("\nAddition of a scalar (100)\n-------------------------\n", 100+mat1)

# print("\nExponentiation, matrix cubed here\n----------------------------------------\n", mat1**3)
# print("\nExponentiation, sq-root using pow function\n-------------------------------------------\n",pow(mat1,0.5))

# create a rank 1 ndarray with 3 values
# start = np.zeros((4,3))
# print(start)
# add_rows = np.array([1, 0, 2])
# print(add_rows)
# y = start + add_rows
# print(y)

# a = np.array([[1,2,3,4]])
# a1 = a.T
# print(a1)

# print(np.array([100]))

# Numpy mathematical functions
# a = np.array(ri(1,10,9)).reshape(3,3)
# b = np.array(ri(1,10,9)).reshape(3,3)
# print(a)
# print(b)
# print(np.sqrt(a))
# print(np.sqrt(b))
# print(np.fmod(a,b))

# print("\nCombination of functions by shwoing exponetial decay of a sine wave\n",'-'*70)
# A = np.linspace(0,12*np.pi,1001)
# plt.scatter(x=A,y=100*np.exp(-A/10)*(np.sin(A)))
# plt.title("Exponential decay of sine wave: exp(-x)*sin(x)")
# plt.show()

# NumPy basic statistics on array
# a = np.array(ri(1,10,9)).reshape(3,3)
# print(a)
# print('@@@@@@@@@@@@@')
# print(np.sum(a))
# print(np.sum(a,axis=0))
# print(np.sum(a,axis=1))
# print(np.prod(a,axis=0))
# print(np.prod(a,axis=1))
# print(np.mean(a))
# print(np.std(a))
# print(np.var(a))
# print(np.median(a))
# print(np.sort(a.reshape(3,3)))
# print(np.percentile(a,50))

# Correlation and covariance
# A = ri(1,5,20) # 20 random integeres from a small range (1-10)
# B = 2*A+5*np.random.randn(20) # B is twice that of A plus some random noise
# print("\nB is twice that of A plus some random noise")
# plt.scatter(A,B) # Scatter plot of B
# plt.title("Scatter plot of A vs. B, expect a small positive correlation")
# plt.show()
# print(np.corrcoef(A,B)) # Correleation coefficient matrix between A and B

# A = ri(1,50,20) # 20 random integeres from a larger range (1-50)
# B = 100-2*A+10*np.random.randn(20) # B is 100 minus twice that of A plus some random noise
# print("\nB is 100 minus twice that of A plus some random noise")
# plt.scatter(A,B) # Scatter plot of B
# plt.title("Scatter plot of A vs. B, expect a large negative correlation")
# plt.show()
# print("")
# print(np.corrcoef(A,B)) # Correleation coefficient matrix between A and B

# a = np.arange(1,10).reshape(3,3)
# print(a)

# Dot product
# A = np.arange(1,10).reshape(3,3)
# B = ri(1,10,9).reshape(3,3)
# print("\n1st Matrix of 1-9 single-digit numbers (A)\n","-"*50,"\n",A)
# print("\n2nd Matrix of random single-digit numbers (B)\n","-"*50,"\n",B)

# print("\nDot product of A and B (for 2D arrays it is equivalent to matrix multiplication) \n","-"*80,"\n",np.dot(A,B))

# A = np.arange(1,6)
# B = ri(1,10,5)
# print("\n1st Vector of 1-5 numbers (A)\n","-"*50,"\n",A)
# print("\n2nd Vector of 5 random single-digit numbers (B)\n","-"*50,"\n",B)

# print("\nInner product of vectors A and B \n","-"*50,"\n",np.inner(A,B), "(sum of all pairwise elements)")
# print("\nOuter product of vectors A and B \n","-"*50,"\n",np.outer(A,B))

# Transpose
# a = ri(1,10,9).reshape(3,3)
# print(a)
# print(np.transpose(a))
# B = ri(1,10,6).reshape(3,2)
# print("\n3x2 Matrix of random single-digit numbers\n","-"*50,"\n",B)
# print("\n2x3 Matrix transpose\n","-"*50,"\n",np.transpose(B))
# print("\nMatrix multiplication of B and B-transpose\n","-"*50,"\n",np.dot(B, np.transpose(B)))

# Linear equation solving, matrix inverse, linear least suqare
# 2x + 5y + z = 14;
# 3x - 2y - z = -1;
# x - 3y + z = 4
# A = np.array([[2,5,1],[3,-2,-1],[1,-3,1]])
# B = np.array([14,-1,4])
# x = np.linalg.solve(A,B)
# print("The solutions are:",x)

# x = np.arange(1,11,1)
# A = np.vstack([x, np.ones(len(x))]).T
# b = np.hstack([x,np.ones(len(x))])
# print(A)
# print(b)

