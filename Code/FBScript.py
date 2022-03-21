from FB import *
    
tf.config.list_physical_devices()
tf.config.set_visible_devices([], 'GPU')
import pandas as pd

_n,_t = np.newaxis,tf.newaxis

#==============================================#
#===== Problem parameters and functions =======#
#==============================================#
s0       = 100.   # Spot 
r        =    .05 # Interest rate
delta    =    .10 # Dividend
sig      =    .20 # Volatility
T        =   3.   # Maturity
d        =   100   # Number of assets
d_       = min(d,10) # Cutoff
N        =   9    # Number of time points
K        = 100.   # Strike
eta      =   1    # Vanilla type (1: call, -1: put) 
vaniType = "Call" if eta == 1 else "Put" 
optType  = "Max" # Max or Basket 
# Statistics of the asset prices
a        = bskt if optType == "Basket" else max_
# Chosen payoff
phi      = payoff(vanilla(eta,K),a,r)
#===============================================#
#===== Training parameters and functions =======#
#===============================================#
# Batch size and number of iterations
B,M      = 2**9, 4000
# Time grid and increment
ts,dt    = timeGrid(T,N,beta = 1)
# Initial boundary as percentage of strike (50% if put and 150% if call)
f0       = 1 + eta/2
# Corridor width
S,_ = simGBM(1.,r,delta,sig,T/N,1,d,int(1e5))
eps = np.std(a(S[:,-1]))
print("epsilon: %2.4f"%eps)
eps      = sig * np.sqrt(np.vstack((dt,dt[-1])).flatten())
# Shrink the corridor towards the end, NO
epsMin,zeta = eps, 1
# Importance Sampling: transform the stock price into (sub)martingales
mu  = 0.01*np.log(d)
print("Additional drift: %2.3f"%mu)
lbda     = (r-delta + mu)/sig * np.ones((d,1))
# Mean and std of simulations to normalize NN input
S,_      = simGBM(s0,r,delta,sig,T,N,d,int(1e5),dt,lbda)
X        = tf.sort(S/a(S)[:,:,_t],axis=-1,direction = 'DESCENDING')[:,:,1:d_]
xInfo    = {"mean": tf.reduce_mean(X,axis=0),"std": tfm.reduce_std(X,axis=0)}
# Optimizer
opt      = keras.optimizers.Adam(learning_rate = 1e-3)
# Group all parameters and functions
params   = {"Problem" : (s0,r,delta,sig,T,d,N,K,eta,a,phi),
            "Training": (B,M,ts,dt,eps,zeta,lbda,xInfo,opt)}
#===============================================#
# Create Table
# CHOOSE FILE NAME
fileName = "Tables/CsteFuzzyWidth %s %s, d = %d, N = %d.csv"%(optType,vaniType,d, N)

try:
    out = pd.read_csv(fileName,index_col=0)
except:
    out = pd.DataFrame(columns = ["Price","Std","Time (Train)","Time (Price)"])
    out.index.name = "Run"
#===============================================#    
I   = 5     # Number of runs
J   = 2**20 # Total number of MC simulations for the price (~1 million)
J1  = J; J2 = int(J/J1) # Split MC runs in smaller batches

V,V2 = np.zeros(J2),np.zeros(J2) # Array of prices
# Use regular grid to compute the price
ts,_ = timeGrid(T,N) 
i    = out.shape[0]
print("", end = f"\rRun: {i+1}/{I}")
# Training 
NN,runTime = trainFB(newNN(f0,d_),params) # Train
tic()
for j in range(J2):
    S,Z   = simGBM(s0,r,delta,sig,T,N,d,J1)  #,lbda = lbda)
    X     = tf.sort(S/a(S)[:,:,_t],axis=-1,direction = 'DESCENDING')[:,:,1:d_]
    xInfo = {"mean": tf.reduce_mean(X,axis=0),"std": tfm.reduce_std(X,axis=0)}
    V[j],_,V2[j],_ = reward(NN,S,a(S),xInfo,phi,K,eta,ts,False,Z) # Price
# First and second moment
v,v2       = np.mean(V),np.mean(V2)
out.loc[i] = [v,np.sqrt((v2 - v**2)/J),runTime,toc()]
out.to_csv(fileName)

if i+1 == I:
    out_ = out.astype(float).round({ "Price":4,"Std":4,"Time (Train)":1,"Time (Price)":1})
    out_.loc["Batch size"]       = [B,"","",""]
    out_.loc["# Iterations"]     = [M,"","",""]
    out_.loc["# MC Simulations"] = [J,"","",""] 
    out_.to_csv(fileName)
