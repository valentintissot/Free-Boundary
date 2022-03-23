from FB import *
    
# Benchmark
fStar  = lambda f0,alpha: lambda t:  tf.cast(K - (K-f0)* (1-t/T)**alpha,"float32")
# Dictionary
fStar_ = {1 : fStar(24.26, .28)}

def vizBdry(i,f,lbda = 0.,eps = [0.,0.],saveFig = False,path = "",titles = ""):
    """Plot boundary with trajectories."""
    
    fig, ax = plt.subplots(1,2,figsize=(10, 4))
    S,*_  = simGBM(s0,r,delta,sig,T,N,d,int(2e2),lbda = lbda)
    for k in range(2):
        bdry = pd.DataFrame(f[k],columns=ts.flatten()).T
        bdry.columns    = ["Neural Net"]
        bdry.index.name = "Time"; bdry.reset_index(inplace=True)
        # Plot the boundary
        ax[k].plot(ts,fStar_[1](ts),"-",color="cornflowerblue",lw=2)
        # Add fuzzy corridor
        bdry["eps"],bdry["-eps"] = bdry["Neural Net"] + K * eps[k], bdry["Neural Net"] - K * eps[k]
        bdry.plot("Time",["Neural Net","eps","-eps"],style=["-",':',':'],
          color=["indianred",'darkred','darkred'],ax=ax[k],lw=2,legend="None",zorder=-.5)
        ax[k].fill_between(bdry["Time"],bdry["eps"],bdry["-eps"], color = 'darkred',alpha = 0.4,zorder=-0.75)
        ax[k].plot(ts,a(S).numpy().T,color="steelblue",lw=0.25,zorder=-1,alpha=0.3)
        ax[k].hlines(K,0,T,alpha=0.5,lw=1)
        ax[k].set_xlabel(""); ax[k].set_yticks([0,K]);
        ax[k].set_xlim(0,T); ax[k].set_xticks([0,T]); ax[k].set_xlabel('t')
        ax[k].set_title(titles[k])
        if eta ==-1: ax[k].set_ylim(0., 2*K)
        else:        ax[k].set_ylim(K-5,np.maximum(2*K,np.amax(bdry["Neural Net"])))
        ax[k].get_legend().remove()
    fig.legend(["Optimal","Neural Net",r"$\epsilon$ corridor"],
               loc = "lower center",fontsize=11,ncol=3,bbox_to_anchor=(.5, -0.1))

    if saveFig:
        plt.savefig("Figures/Bdry %d, lbda=%2.3f, mu=%2.3f, N=%d.pdf"%(i,lbda,mu,N),dpi=600, bbox_inches='tight')
    #plt.show() 

tf.config.list_physical_devices()
tf.config.set_visible_devices([], 'GPU')
import pandas as pd

_n,_t = np.newaxis,tf.newaxis

#==============================================#
#===== Problem parameters and functions =======#
#==============================================#
s0       = 40.    # Spot 
r        =   .06  # Interest rate
delta    =   .0   # Dividend
sig      =   .40  # Volatility
T        =   1.   # Maturity
d        =   1    # Number of assets
N        =   50  # Number of time points 
K        =   40.  # Strike
eta      =   -1   # Vanilla type (1: call, -1: put) 
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
B,M      = 2**10, 3000
# Time grid and increment (regular if beta =1)
ts,dt    = timeGrid(T,N)#,beta = 1/2)
# Initial boundary as percentage of strike (50% if put and 150% if call)
f0       = 1 + eta/2
# Corridor width
eps      = sig * np.sqrt(np.vstack((dt,dt[-1])).flatten())
epsInit  = eps[0]
# DO NOT Shrink the corridor towards the end
epsFinal = epsInit
zeta     = 1. 
# Importance Sampling
mu       = -eta * 0.05 # Desired drift (> 0: downward, <0: upward)
lbda     = (r-delta + mu)/sig * np.ones((d,1)) 
# Mean and std of simulations to normalize NN input (not needed(
xInfo    = {"mean": 0.,"std": 1.}
# Optimizer
opt      = keras.optimizers.Adam(learning_rate = 1e-3)
# Group all parameters and functions
params   = {"Problem" : (s0,r,delta,sig,T,d,N,K,eta,a,phi), 
            "Training": (B,M,ts,dt,eps,zeta,lbda,xInfo,opt)}
#===============================================#
# Create Table
# CHOOSE FILE NAME
fileName = "Tables/Bdry bigger batch %s %s, d = %d, N = %d, mu = %2.2f.csv"%(optType,vaniType,d, N,mu)

try:
    out = pd.read_csv(fileName,index_col=0)
except:
    out = pd.DataFrame(columns = ["Price","Std","Time (Train)","Time (Price)"])
    out.index.name = "Run"
#===============================================#    
I   = 10    # Number of runs
J   = 2**22 # Total number of MC simulations for the price (~4 million)
J1  = min(J,2**19); J2 = int(J/J1) # Split MC runs in smaller batches

V,V2 = np.zeros(J2),np.zeros(J2) # Array of prices
# Use regular grid to compute the price
ts,_ = timeGrid(T,N) 
i    = out.shape[0]
print("", end = f"\rRun: {i+1}/{I}")
# Training 
NN,runTime = trainFB(newNN(f0,d),params) # Train
tic()
for j in range(J2):
    S,Z            = simGBM(s0,r,delta,sig,T,N,d,J1,lbda=lbda)
    V[j],_,V2[j],_ = reward(NN,S,a(S),xInfo,phi,K,eta,ts,False,Z) # Price
# First and second moment
v,v2       = np.mean(V),np.mean(V2)
out.loc[i] = [v,np.sqrt((v2 - v**2)/J),runTime,toc()]
out.to_csv(fileName)

# Visualize
#vizBdry(i,nnOut(NN,1,N,K,ts).numpy(),lbda=lbda, eps = [epsInit,epsFinal],saveFig=True)

if i+1 == I:

    out.loc["Mean"] = out.mean(axis=0).T
    out.loc["Std"]  = out.std(axis=0).T
    out_ = out.astype(float).round({ "Price":4,"Std":4,"Time (Train)":1,"Time (Price)":1})
    out_.loc["Batch size"]       = [B,"","",""]
    out_.loc["# Iterations"]     = [M,"","",""]
    out_.loc["# MC Simulations"] = [J,"","",""] 
    out_.loc["Mu"]               = [mu,"","",""] 
    out_.to_csv(fileName)

