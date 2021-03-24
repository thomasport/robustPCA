import numpy as np
from numpy.random import rand
from scipy.io import loadmat
from matplotlib import pyplot as plt
from math import sqrt
from utils import *

def stoc_rpca(D,nrank, lambda1=None,lambda2=None):
    ndim, nsample = D.shape
    if lambda1 is None:
        lambda1 = 1/sqrt(max(D.shape))
    if lambda2 is None:
        lambda2 = lambda1

    L = rand(ndim, nrank)
    A = np.zeros((nrank,nrank))
    B = np.zeros((ndim,nrank))

    S = np.zeros((ndim,nsample))
    Lr = np.zeros((ndim,nsample))
    # print("checkpoint 1")
    for idx, z in enumerate(D.T):
        print("{} face".format(idx))
        r,s = solve_proj2(z,L,lambda1,lambda2)
        # print("solved proj")
        S[:,idx] = s.flatten()
        z = z.reshape(z.shape[0],-1)
        A = A + np.matmul(r,r.T)
        B = B + np.matmul(z-s,r.T)
        # print("update_col")
        L = update_col(L,A,B,lambda1/nsample)
        Lr[:,idx] = L.dot(r).flatten()
    
    return Lr, S

def update_col(L,A,B,lambda1):
    ncol = L.shape[1]
    A = A + lambda1*np.eye(ncol,ncol)
    for idx, lj  in enumerate(L.T):
        bj = B[:,idx]
        aj = A[:,idx]
        temp = lj+(bj - np.matmul(L,aj))/A[idx,idx]
        L[:,idx] = temp/max(np.linalg.norm(temp),1)    

    return L

def solve_proj2(z,D,lambda1,lambda2):
    ndim, ncol = D.shape
    z = z.reshape(z.shape[0],-1)
    e = np.zeros((ndim,1))
    x = np.zeros((ncol,1))
    I = np.eye(ncol)
    converged = False
    maxIter = 100
    iter = 0    
    I_lambda1  = lambda1*I
    aux_D = np.matmul(D.T,D)
    aux_D_inv = np.linalg.inv(aux_D+I_lambda1)
    DDt = np.matmul(aux_D_inv,D.T)

    while(not converged):
        iter+=1

        xtemp = x.copy()

        x = np.matmul(DDt,z-e)
     
        etemp = e.copy()
        e = threshold(z-np.matmul(D,x),lambda2)

        stopc = max(np.linalg.norm(e-etemp), np.linalg.norm(x-xtemp))/ndim
        # if iter == maxIter:
        #   print("maxiter")
        if (stopc<10^-6 or iter>maxIter):
            converged=True

    return x,e  

def threshold(y,mu):
    aux = y-mu
    aux[aux<0] = 0
    aux2 = y+mu
    aux2[aux2>0] = 0
    x = aux+aux2
    return x





if __name__ == "__main__":
    data_path = "../data/allFaces.mat"
    raw_data = loadmat(data_path)
    print(raw_data.keys())
    n = raw_data['n'][0][0]
    m = raw_data['m'][0][0]
    faces = raw_data['faces']
    nfaces= raw_data['nfaces'].flatten()
    sub = raw_data['sub']
    
    selected_faces = sample_imgs(faces,10,nfaces)
    print("selected faces shape-> {}".format(selected_faces.shape))
    nrank = 100
    Lr, S = stoc_rpca(selected_faces,nrank)
    print("plotting")
    plot_LS(Lr, S ,selected_faces, n_faces = 3)