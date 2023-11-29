import numpy
import logging
import os

def bundle_adjust(set1, set2, printer = True):
    if set1.shape != set2.shape:
        raise Exception("Can only bundle adjust equally shaped sets of points")
    dim = set1.shape[1]
    m = numpy.hstack((set1, set2))
    mean = numpy.mean(m, axis=0)
    m -= mean
    H = numpy.matlib.zeros([dim,dim])
    for d in m:
        for i in range(dim):
            for j in range(dim):
                H[i,j] += d[j+dim]*d[i]
                
    U, s, V = numpy.linalg.svd(H)
    
    det = numpy.linalg.det(V.T*U.T)
    R = numpy.asarray(V.T*numpy.diag([1,1,det])*U.T).T
    T = -numpy.matmul(R, mean[dim:(2*dim)].T).T + mean[0:dim]
    adjusted = numpy.matmul(R, set2.T).T + T
    vol = numpy.linalg.norm(set1 - adjusted, axis=1)
    if printer:
        print("--- Bundle Adjust ---")
        print("Rotation")
        print(R)
        print("Translation")
        print(T)
        print("Before: {}".format(numpy.mean(numpy.linalg.norm(set1 - set2, axis=1))))
        print("After:  {}".format(numpy.mean(vol)))
        print("Maximum deviations: {}\t{}\t{}".format(*numpy.sort(vol)[-3:].tolist()))
        
    return R, T, adjusted

def quaternion(R, isprecise=False): #Gohlke
    M = numpy.array(R)
    if isprecise:
        q = numpy.empty((4, ))
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / numpy.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                        [m01+m10,     m11-m00-m22, 0.0,         0.0],
                        [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                        [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = numpy.linalg.eigh(K)
        q = V[[3, 0, 1, 2], numpy.argmax(w)]
    if q[0] < 0.0:
        numpy.negative(q, q)
    return q

def rotation_matrix(q):
    a, b, c, d = q
    return numpy.array( 
        [ 
            [a**2 + b**2 - c**2 - d**2, 2*(b*c - a*d), 2*(b*d + a*c)],
            [2*(b*c + a*d), a**2 -b**2  + c**2 - d**2, 2*(c*d - a*b)],
            [2*(b*d - a*c), 2*(c*d + a*b), a**2 - b**2 - c**2 + d**2]

        ]
    )

def euler(q):
    a, b, c, d = q

    alpha = numpy.arctan2(2*(a*b + c*d), a** - b**2 -c**2 + d**2)
    beta  = numpy.arcsin(2*(a*c - b*d))
    gamma = numpy.arctan2(2*(a*d + b*c), a**2 + b**2 - c**2 - d**2)

    return [alpha, beta, gamma]

def init_logging(log_level=None):
    if log_level is None and 'LOG_LEVEL' in os.environ:
        log_level = os.environ['LOG_LEVEL']
    else:
        log_level = logging.WARNING
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] [%(module)-10s] %(message)s", datefmt='%d/%m/%Y %H:%M:%S')
    rootLogger = logging.getLogger("SmoPa3D")

    fileHandler = logging.FileHandler(os.path.join(os.environ['ROOT'], "smopa3d.log"))
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.WARNING)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(log_level)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(log_level)
    return rootLogger

