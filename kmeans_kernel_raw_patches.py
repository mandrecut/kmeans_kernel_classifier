# (c) 2020, M. Andrecut, mircea.andrecut@gmail.com
import numpy as np
from scipy import sparse
from scipy.special import softmax


def read_data(imagefile,labelfile,N,M):
    x = np.zeros((N,M),dtype='uint8')
    images = open(imagefile,'rb')
    images.read(16)  # skip the magic_number
    for n in range(N):
        x[n,:] = np.frombuffer(images.read(M),dtype='uint8')
    images.close()
    labels = open(labelfile,'rb')
    labels.read(8)  # skip the magic_number
    xl = np.frombuffer(labels.read(N),dtype='uint8')
    labels.close()
    return (x, xl)


def get_one_image(x,L,l):
    z = x.reshape((L,L))
    y = np.zeros(((L-l+1)*(L-l+1),l*l))
    n = 0
    for i in range(L-l+1):
        for j in range(L-l+1):
            y[n,:] = z[i:i+l,j:j+l].flatten()
            y[n,:] = y[n,:] - np.mean(y[n,:])
            y[n,:] = y[n,:]/np.linalg.norm(y[n,:])
            n = n + 1
    return y


def kmeans(z,Q):
    np.random.shuffle(z)
    u,err = z[:Q,:],1
    while err > 1e-6:
        r = np.dot(z,u.T)
        cols = np.argmax(r,axis=1)
        rows = np.array([i for i in range(len(r))])
        data = np.array([1 for i in range(len(r))])
        r = sparse.csr_matrix((data,(rows,cols)),shape=r.shape)
        v = sparse.csr_matrix.dot(r.T,z)
        err = 0
        for q in range(len(v)):
            v[q,:] = v[q,:]/np.linalg.norm(v[q,:])
            err += np.dot(u[q,:],v[q,:])
        u,err = v,1-err/len(v)
    return u


def features(x,L,l,Q):
    (N,M),LL,ll = x.shape,(L-l+1)*(L-l+1),l*l
    z = np.zeros((N*LL,ll))
    for n in range(N):
        z[n*LL:(n+1)*LL,:] = get_one_image(x[n,:],L,l)
    u = kmeans(z,Q)
    return u


def lssvm(u,ul,K):
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


def lssvm_test(w,y,yl,K,L,l):
    (J,M), LL = y.shape, (L-l+1)*(L-l+1)
    h = np.zeros((J,K))
    q = np.ones((LL,u.shape[0]+1))
    for n in range(J):
        z = get_one_image(y[n,:],L,l)
        q[:,:-1] = np.dot(z,u.T)
        q = q**4
        r = np.dot(q,w)
        h[n,:] = np.sum(r,axis=0)
        h[n,:] = softmax(h[n,:])
    return h


def check_solution(g,yl):
    err = 0
    for n in range(len(yl)):
        k = np.argmax(g[n,:])
        if k != yl[n]:
            err += 1
    return err*100.0/len(yl)


if __name__ == '__main__':
    np.random.seed(123)
    print("Read data")
    (x,xl) = read_data("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte", 60000, 784)
    (y,yl) = read_data("./data/t10k-images.idx3-ubyte", "./data/t10k-labels.idx1-ubyte", 10000, 784)
    print("K-means")
    Q = 3000 # number of representative patches per class
    L,l = 28,26 # image and patch size
    u,ul,K = [],[], np.max(xl)+1
    for k in range(K):
        print("k=",k)
        v = features(x[xl==k],L,l,Q)
        for i in range(len(v)):
            u.append(v[i,:].tolist())
            ul.append(k)
    u = np.array(u)
    print("LS-SVM")
    w  = lssvm(u,ul,K)
    g = lssvm_test(w,y,yl,K,L,l)
    err = check_solution(g,yl)
    print("Test error =", round(err, 3), "%")
