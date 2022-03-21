import numpy as np
import numpy.random as rdm
import matplotlib.pyplot as plt; plt.style.use('dark_background')
from   ttictoc import tic,toc
from   scipy.stats import norm
import tensorflow as tf
import tensorflow.math as tfm
from   tensorflow import keras
from keras import losses 
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import Constant, RandomUniform
_n,_t = np.newaxis,tf.newaxis

#=================================#

def timeGrid(T,N,beta = 1): 
    """Time grid (beta = 1 -> regular grid)"""
    ts   = np.linspace(0,1,N+1)[:,_n]
    ts   = ts*(ts < 1/2) + (ts**beta + 1/2 - 2**(-beta))*(ts >= 1/2)
    ts  *= T/ts[-1]
    return ts, np.diff(ts,axis=0).astype("float32")

def simGBM(s0,r,delta,sig,T,N,d,nSim,dt = None,lbda = None,optType = "",nMax = 200): # 1000
    """Stock price simulation. 
       lbda: Girsanov parameter for importance sampling)"""
    if dt is None  : dt = T/N
    if lbda is None: lbda = np.zeros((d,1))
    dt_,rD = dt , 1
    if optType == "Lookback": 
        rD    = int(nMax/N) # Refine the partition if needed
        dt_  /= rD
    # Brownian Increments
    dW = tfm.sqrt(dt_) * tf.random.normal(shape=(nSim,N*rD,d))
    # Log Price
    X  = tfm.cumsum((r-delta-sig**2/2 - sig*lbda.T)*dt_ + sig*dW, axis=1)
    X  = tfm.log(s0) + tf.concat((np.zeros((nSim,1,d)), X),axis=1)
    # Log of density process (Importance Sampling)
    if d == 1: L = tfm.cumsum(-lbda**2*dt_/2 + lbda*dW   , axis=1)
    else:      L = tfm.cumsum((-lbda.T*dt_/2 + dW) @ lbda, axis=1)
    L = tf.concat((np.zeros((nSim,1,1)), L),axis=1)
    if optType == "Lookback": 
        X   = tf.concat([X, np.maximum.accumulate(X,axis=1)],axis=2)
        X,L = X[:,::rD,:], L[:,::rD,:]
    # Return stock prices and density process
    return tfm.exp(X), tfm.exp(L)

#=================================#

# Payoff function
def payoff(g,a,r): return lambda t,s: np.exp(-r*t) * g(a(s))
# Vanilla payoff
def vanilla(eta,K): return lambda a:  tf.maximum(eta*(a-K),0.) 
# Statistics of asset prices
def bskt(S, keepD = False): return tf.reduce_mean(S,np.ndim(S)-1,keepdims = keepD)
def max_(S, keepD = False): return tf.reduce_max(S, np.ndim(S)-1,keepdims = keepD)

#=================================#

def normalize(X,mu = None,sig = None,ax=0): 
    """Input normalization"""
    try: 
        return  (X - mu)/(sig + 1e-12)
    except:
        mu,sig = tf.reduce_mean(X,axis=ax), tf.math.reduce_std(X,axis=ax)
        return  (X - mu)/(sig + 1e-12)

def leakyReLU(x): return tf.nn.leaky_relu(x,alpha = 1e-2)
def xavierN(p,q): s = np.sqrt(6/(p+q)); return RandomUniform(minval = -s,maxval = s)

def newNN(f0,dX): 
    tf.keras.backend.clear_session()
    """Neural Network Architecture"""
    # Input dimension (add assets if d > 1)
    d_ = 1 + max(dX - 1,0) 
    # Number of nodes in hidden layers 
    qH = 20 + d_
    return   Sequential([ 
             Dense(qH,activation = leakyReLU,input_dim = d_), 
             Dense(qH,activation = leakyReLU),
             Dense(qH,activation = leakyReLU), 
             Dense(1, activation = "relu",bias_initializer = Constant(f0))])

def nnOut(NN,B,N,K,ts,Xi=None,xiInfo=None,train = False,d_ = 10):
    """Prepare input and return NN output."""
    # NN Input
    Y = np.tile(normalize(ts[:-1]),[B,1,1])
    # Add stock prices if d > 1
    if NN.input_shape[1] > 1: 
        # Sort the asset prices in decreasing order
        Xi = tf.sort(Xi,axis=-1,direction = 'DESCENDING')[:,:-1,1:d_]
        Y = tf.concat((normalize(Xi,xiInfo["mean"],xiInfo["std"]),Y),axis = 2)
    Y = tf.reshape(Y,(B*N,NN.input_shape[1]))
    # Threshold function
    f = tf.reshape(NN(Y,training = train),(B,N))
    # Add f(T,•) = K
    return K * tf.concat([f, tf.ones_like(f[:,0:1])],axis=1)

def stopFactor(dist_,k = 0.99):  
    """Stopping factors for each time point."""
    c = np.log(1/k-1) # Scaling coefficient
    return tf.sigmoid(c * dist_)

def stopFactor2(dist_):  
    """Stopping factors for each time point."""
    return tf.clip_by_value((1-dist_)/2,0.,1.)

def reward(NN,S,aS,xInfo,phi,K,eta,ts,train = False,Z = None,eps = 0.):
    # Dimensions
    B,N,d = np.shape(S); N -= 1 
    d_ = min(d,10)
    # Boundary
    if d > 1: f = nnOut(NN,B,N,K,ts,S/aS[:,:,_t],xInfo,train,d_)
    else    : f = nnOut(NN,B,N,K,ts,train = train)
    # Cap and floor the NN output for puts and calls, respectively 
    lim0 = K*(tf.ones_like(f) + eta*1e-6*tf.eye(N+1)[0,:])
    f    = tf.minimum(f,lim0) if eta == -1 else tf.maximum(f,lim0)
    # Signed distance to the boundary
    dist = eta * (f - aS)/K   
    if train: u = stopFactor2(dist/eps)      # Smooth Interface
    else    : u = tf.where(dist <= 0.,1.,0.) # Sharp  Interface
    # Stopping probabilities
    v = tf.concat([tf.ones_like(u[:,0:1]),1 - u[:,:-1] + 1e-10],axis=1)
    w = tf.concat([u[:,:-1], tf.ones_like(u[:,0:1])],axis=1) 
    U = w * tfm.cumprod(v,axis = 1) 
    if train: 
        # Payoff values and corresponding rewards
        phis = tf.concat([phi(ts[i],S[:,i:i+1]) for i in range(N+1)],axis=1)
        G    = tf.reduce_sum(Z[:,:,0] * U * phis,axis=1,keepdims = True)     
        return tf.reduce_mean(G)      
    else: 
        # Stopping times and stopped values
        tau = tf.reduce_sum(np.arange(0,N+1) * U,axis=1)
        s,z = tf.reduce_sum(U[:,:,_t]*S,axis=1),tf.reduce_sum(U[:,:,_t]*Z,axis=1)
        # Change scale
        tauDt = np.array([ts[int(t)] for t in tau]).T
        # No training -> simply use the values at stopping time
        if len(tau) > 0: G = tf.reshape(z,[-1]) * phi(tauDt,s)
        else           : G = 0. 
        return np.mean(G), f.numpy(), np.mean(G**2), tauDt
    
def trainFB(NN,params,show = False,rdmS0 = False): 
    """Train the free boundary."""
    s0,r,delta,sig,T,d,N,K,eta,a,phi   = params["Problem"]
    B,M,ts,dt,eps,zeta,lbda,xiInfo,opt = params["Training"]
    # Training Loop
    theta, rwds = NN.trainable_weights, []; tic()
    for m in range(M):
        if (m+1) % 50 == 0 and show: print("", end = f"\rIteration: {m+1}/{M}")
        # Increase batch size for the last training iterations
        if m == int(4*M/5): B *= 4 
        # Simulation
        if rdmS0: s0_ = K*tf.maximum(tf.random.normal(shape=(B,1,d),mean=1.,stddev=1/3),1e-10)
        else: s0_ = s0
        S,Z = simGBM(s0_,r,delta,sig,T,N,d,B,lbda=lbda)   #dt not given
        # Gradient Update  
        with tf.GradientTape() as tape: 
            rwd = reward(NN,S,a(S),xiInfo,phi,K,eta,ts,True,Z,eps);loss = -rwd 
        opt.apply_gradients(zip(tape.gradient(loss, theta),theta))  
        rwds.append(rwd.numpy())
        # Shrink the corridor for the last iterations
        if m > 2*M/3: eps *= zeta 
    runTime = toc()        
    if show: 
        print("\nTraining: %2.f seconds"%runTime)
        showRewards(rwds)
    return NN,runTime

def showRewards(rwds,burnIn=10):
    """Plot reward vs training iteration."""
    rwds = rwds[burnIn:] # Remove burn-in period (optional)
    plt.figure(figsize=(6,3))
    plt.plot(burnIn + np.arange(1,len(rwds)+ 1),rwds,color ="steelblue")
    plt.xlabel("Training Step"); plt.title("Reward vs Training Step")
    plt.show()