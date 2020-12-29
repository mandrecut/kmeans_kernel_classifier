# (c) 2020, M. Andrecut
import numpy as np
from scipy import sparse
from scipy.special import softmax


def read_data(imagefile,labelfile,N,M,fft):
    if fft:
        x = np.zeros((N,M+M//2))
        s2 = np.sqrt(2)
    else:
        x = np.zeros((N,M))        
    images = open(imagefile,'rb')
    images.read(16)  # skip the magic_number
    for n in range(N):
        z = np.frombuffer(images.read(M),dtype='uint8').astype(float)
        z = z - np.mean(z)
        if fft:
            f = np.abs(np.fft.fft(z))[:M//2]
            f = f - np.mean(f)
            f = f/np.linalg.norm(f)
            z = z/np.linalg.norm(z)
            x[n,:] = np.hstack((z,f))/s2
        else:
            x[n,:] = z/np.linalg.norm(z)
    images.close()
    labels = open(labelfile,'rb')
    labels.read(8)  # skip the magic_number
    xl = np.frombuffer(labels.read(N), dtype='uint8')
    labels.close()
    return (x, xl)


def kmeans(x,Q):
    np.random.shuffle(x)
    u,err = x[:Q,:],1
    rows = np.arange(len(x))
    data = np.ones(len(x))
    while err > 1e-6:
        r = np.dot(x,u.T)
        cols = np.argmax(r,axis=1)
        r = sparse.csr_matrix((data,(rows,cols)),shape=r.shape)
        v = sparse.csr_matrix.dot(r.T,x)
        err = 0
        for q in range(len(v)):
            v[q,:] = v[q,:]/np.linalg.norm(v[q,:])
            err += np.dot(u[q,:],v[q,:])
        u,err = v,1-err/len(v)
    return u


def lssvm(u, ul, K):
    (N,M) = u.shape
    q = np.ones((N+1,N+1))
    q[:-1,:-1] = np.dot(u,u.T)
    q = q**4
    v = np.zeros((N+1,K))
    for n in range(N):
        v[n,ul[n]] = 1.0
        q[n,n] += 1e-6
    q[N,N] = 0
    w = np.linalg.solve(q,v)
    return w


def lssvm_test(y,u,w):
    (J,M) = y.shape
    q = np.ones((J,len(u)+1))
    q[:,:-1] = np.dot(y,u.T)
    q = q**4
    g = softmax(np.dot(q,w))
    return g


def check_solution(g,yl):
    err = 0
    for n in range(len(yl)):
        k = np.argmax(g[n,:])
        if k != yl[n]:
            err += 1
    return err*100.0/len(yl)


if __name__ == '__main__':
    np.random.seed(12)
    fft = True # False/True, FFT selection switch
    print("Read data")
    (x, xl) = read_data("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte", 60000, 784, fft)
    (y, yl) = read_data("./data/t10k-images.idx3-ubyte", "./data/t10k-labels.idx1-ubyte", 10000, 784, fft)
    print("K-means")
    Q = 1000 # number of representative vectors per class
    u,ul,K = [],[],np.max(xl)+1
    for k in range(K):
        print("k=",k)
        v = kmeans(x[xl==k], Q)
        for i in range(len(v)):
            u.append(v[i,:].tolist())
            ul.append(k)
    u = np.array(u)
    print("LS-SVM")
    w  = lssvm(u, ul, K)
    g = lssvm_test(y, u, w)
    err = check_solution(g, yl)
    print("Test error =",round(err,3),"%")
        
