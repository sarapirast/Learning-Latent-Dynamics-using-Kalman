import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split

class kalman:
    def __init__(self,A,C,Q,x_0,sig,R):
        self.A= A
        self.x=x_0
        self.Q= Q
        self.C=C
        self.R=R
        self.sig= sig
        self.I= np.eye(self.A.shape[0])

        self.x_pred,self.p_pred=[],[]
        self.x_filt,self.p_filt=[],[]
    def predict(self):
        self.x= self.A @self.x
        self.sig= self.A @ self.sig @ self.A.T+ self.Q
        self.x_pred.append(self.x.copy())
        self.p_pred.append(self.sig.copy())
        return self.x, self.sig

    def update(self,yk):
        inv= np.asarray(yk)-(self.C @ self.x)
        S= self.C @ self.sig @ self.C.T + self.R
        K= self.sig @ self.C.T@ np.linalg.inv(S)

        self.x= self.x +K @ inv
        self.sig= (self.I - K @self.C) @ self.sig
        self.x_filt.append(self.x.copy())
        self.p_filt.append(self.sig.copy())
        return self.x, self.sig,K,inv,S
    def smoother(self):
        x_filt=np.asarray(self.x_filt)
        p_filt=np.asarray(self.p_filt)
        x_pred=np.asarray(self.x_pred)
        p_pred=np.asarray(self.p_pred)

        N = x_filt.shape[0]
        x_s = np.empty_like(x_filt)
        p_s = np.empty_like(p_filt)

        x_s[-1] = x_filt[-1]
        p_s[-1] = p_filt[-1]

        AT = self.A.T
        for k in range(N-2, -1, -1):
            S = p_filt[k] @ AT @ np.linalg.inv(p_pred[k+1])
            x_s[k] = x_filt[k] + S @ (x_s[k+1] - x_pred[k+1])
            p_s[k] = p_filt[k] + S @ (p_s[k+1] - p_pred[k+1]) @ S.T
        return x_s, p_s

class EM:
    def __init__(self,A,C,Q,R,x0,v0):
        self.A= A.copy()
        self.C=C.copy()
        self.Q= Q.copy()
        self.R= R.copy()
        self.x0= x0.copy()
        self.v0= v0.copy()
        self.maxiter= 100
        self.tol= 1e-10
        self.mse=[]
        self.x_s=None

    def nrmse(self,y, y_pred):
        num = np.sum((y - y_pred) ** 2)
        den = np.sum(y ** 2)
        return np.sqrt(num / den)


    def E(self,y):
        T,p=y.shape
        kf=  kalman(self.A,self.C,self.Q,self.x0.copy(),self.v0.copy(),self.R)
        y_pred=np.zeros((T,p))
        for t in range(T):
            kf.predict()
            kf.update(y[t])
            y_pred[t]=self.C@kf.x
        x_s,P_s= kf.smoother()

        e= self.nrmse(y,y_pred)
        self.mse.append(e)
        return x_s,P_s,y_pred

    def M(self,y,x_s,P_s):
        T,p=y.shape
        d=x_s.shape[1]
        eps= 1e-6

        e=np.zeros((d,d))
        e_prev= np.zeros((d,d))
        ex= np.zeros((d,d))


        for t in range(T):
            e += P_s[t]+np.outer(x_s[t],x_s[t])
        for t in range(1,T):
            e_prev+= P_s[t-1] + np.outer(x_s[t-1],x_s[t-1])
            ex+= np.outer(x_s[t],x_s[t-1])

        self.A=ex@np.linalg.inv(e_prev+eps*np.eye(d))


        Q_new=np.zeros((d,d))
        for t in range(1,T):
            e_t= P_s[t] + np.outer(x_s[t],x_s[t])
            e_tprev= P_s[t-1] + np.outer(x_s[t-1],x_s[t-1])
            ext= np.outer(x_s[t],x_s[t-1])

            Q_new += (e_t-self.A@ext.T-ext@self.A.T+self.A@e_tprev@self.A.T)

        self.Q= Q_new/(T-1)
        x_sum= np.zeros((p,d))
        for t in range(T):
            x_sum += np.outer(y[t],x_s[t])
        self.C= x_sum @ np.linalg.inv(e+eps*np.eye(d))

        R_new= np.zeros((p,p))
        for t in range(T):
            y_t= y[t]
            ex_t= P_s[t]+np.outer(x_s[t],x_s[t])

            yy= np.outer(y_t,y_t)
            yx= np.outer(y_t,x_s[t])

            R_new += (yy-self.C@yx.T-yx@self.C.T+self.C@ex_t@self.C.T)
        self.R=R_new/T

        self.x0=x_s[0].copy()
        self.v0=P_s[0].copy()

    def fit(self,y):
        self.mse=[]
        prev= None
        for it in range(self.maxiter):
            x_s,P_s,y_pred=self.E(y)
            self.M(y,x_s,P_s)
            if prev is not None:
                if abs(prev-self.mse[-1])< self.tol:
                    break
            prev=self.mse[-1]
        self.x_s=x_s
        return self

def data_loader():
  y=[]
  mat= sio.loadmat('sentences.mat')
  ts= mat["neuralActivityTimeSeries"]
  numbins= mat["numTimeBinsPerSentence"].squeeze().astype(int)
  offset = 0
  for s_idx, T_s in enumerate(numbins):
      y.append(ts[offset:offset + T_s, :])
      offset += T_s  ##to split each sentence
  return y

def smoother(sentences, timestep, sigma=None):
  out=[]
  for sent in sentences:
    s= sent.astype(np.float32)
    if sigma is not None and sigma>0:
      s= gaussian_filter1d(s,sigma=sigma, axis=0, mode="nearest")
    s_ds= s[::timestep,:] #downsample
    out.append(s_ds)
  return out

def neurons(sentences, neuron_idx):
  return [sent[:,neuron_idx] for sent in sentences]

y= data_loader()
X_train1, X_valtest1 = train_test_split(y,test_size=0.2,random_state=42)
X_val1, X_test1 = train_test_split(X_valtest1,test_size=0.5,random_state=42)

timestep= 5
sig= 1.0 #for gauss smooth
X_train_sm= smoother(X_train1,timestep,sig)
X_val_sm= smoother(X_val1,timestep,sig)
X_test_sm= smoother(X_test1,timestep,sig)

print(X_train_sm)

neuron= 30 #keep 30 instead of 192
X_train_stack= np.vstack(X_train_sm).astype(np.float32)
neuronvar= X_train_stack.var(axis=0)
neuron_idx= np.argsort(neuronvar)[-neuron:]
X_train= neurons(X_train_sm,neuron_idx)
X_val= neurons(X_val_sm,neuron_idx)
X_test= neurons(X_test_sm,neuron_idx)

X_train

X_train_m= np.vstack(X_train) #concat
mean= X_train_m.mean(axis=0,keepdims=True)
X_c= X_train_m-mean
u,s,vt=np.linalg.svd(X_c,full_matrices=False) ##PCA
p= X_train[0].shape[1]
k= 20 #random

def init_model(mode,X_c,vt,p,k):
  if mode=="pca":
    C= vt[:k].T  ##(p,k)
    x= X_c@ C
    ##########A##########
    X_t= x[:-1] #past
    X_t1= x[1:] #future
    A= np.linalg.lstsq(X_t,X_t1,rcond=None)[0].T
    #########Q##########
    diff= X_t1-X_t@A.T
    Q=np.cov(diff,rowvar=False) #(k,k)
    Q += 1e-6*np.eye(k)

  elif mode=="random":
    C= np.random.randn(p,k)
    x= X_c@ C #linear proj
    ########rand A################
    rng= np.random.default_rng(0)
    A= 0.5 * np.eye(k) + 0.1 * rng.normal(size=(k, k))
    #######rand Q#########
    Q= np.eye(k)


##########R##############
  E_obs= X_c - (x@ C.T)
  R= np.cov(E_obs,rowvar=False)
  R += 1e-6 *np.eye(p)

  #######x_0,v_0###############
  x0_init= np.zeros(k)
  v0_init= np.eye(k)
  return C,A,Q,R,x0_init,v0_init

def run_kalman(A,C,Q,R,x0,v0,X,train_mean):
    Xc= X-train_mean
    T_s,p= Xc.shape
    x_pred= np.zeros_like(Xc)
    kf= kalman(A,C,Q,x0.copy(),v0.copy(),R)
    for t in range(T_s):
        kf.predict()
        kf.update(Xc[t])
        x_pred[t]=C@kf.x
    xs,ps=kf.smoother()
    num = np.sum((Xc - x_pred) ** 2)
    den = np.sum(Xc ** 2)
    nrmse= np.sqrt(num / den)
    X_pred= x_pred+train_mean
    return X_pred,xs,nrmse

def eval(X,mean,A,C,Q,R,x0,v0):
    pred,latent,error=[],[],[]
    for sent in X:
        x_pred,xs,e= run_kalman(A,C,Q,R,x0,v0,sent,mean)
        pred.append(x_pred)
        latent.append(xs)
        error.append(e)
    return pred,latent,error

C_pca,A_pca,Q_pca,R_pca,x0_pca,v0_pca= init_model("pca",X_c,vt,p,k)
C_rand,A_rand,Q_rand,R_rand,x0_rand,v0_rand= init_model("random",X_c,vt,p,k)

####Rand Kalman####
rand_train_pred, rand_train_lat, rand_train_err = eval(X_train, mean, A_rand, C_rand, Q_rand, R_rand, x0_rand, v0_rand)
rand_val_pred, rand_val_lat, rand_val_err = eval(X_val, mean, A_rand, C_rand, Q_rand, R_rand, x0_rand, v0_rand)
rand_test_pred, rand_test_lat, rand_test_err = eval(X_test, mean, A_rand, C_rand, Q_rand, R_rand, x0_rand, v0_rand)

####PCA Kalman####
pca_train_pred, pca_train_lat, pca_train_err = eval(X_train,mean,A_pca,C_pca,Q_pca,R_pca,x0_pca,v0_pca)
pca_val_pred, pca_val_lat, pca_val_err = eval(X_val,mean,A_pca,C_pca,Q_pca,R_pca,x0_pca,v0_pca)
pca_test_pred, pca_test_lat, pca_test_err = eval(X_test,mean,A_pca,C_pca,Q_pca,R_pca,x0_pca,v0_pca)

####EM Kalmna####
em= EM(A_rand,C_rand,Q_rand,R_rand,x0_rand,v0_rand)
em.fit(X_c)

A_h=em.A
C_h=em.C
Q_h=em.Q
R_h=em.R
x0_h=em.x0
v0_h=em.v0

EM_train_pred,EM_train_lat,EM_train_err= eval(X_train,mean,A_h,C_h,Q_h,R_h,x0_h,v0_h)
EM_val_pred,EM_val_lat,EM_val_err= eval(X_val,mean,A_h,C_h,Q_h,R_h,x0_h,v0_h)
EM_test_pred,EM_test_lat,EM_test_err= eval(X_test,mean,A_h,C_h,Q_h,R_h,x0_h,v0_h)

def plot_reconstruction(sent, mean, A, C, Q, R, x0, v0, neuron_idx=0):
    X_pred, xs, nrmse = run_kalman(A, C, Q, R, x0, v0, sent, mean)
    T = sent.shape[0]
    plt.figure(figsize=(12,4))
    plt.plot(sent[:, neuron_idx], label=f"Original neuron {neuron_idx}", alpha=0.8)
    plt.plot(X_pred[:, neuron_idx], label=f"Reconstructed neuron {neuron_idx}",alpha=0.8)
    plt.title(f"Neuron {neuron_idx} Reconstruction — NRMSE={nrmse:.4f}")
    plt.xlabel("Time")
    plt.ylabel("Activity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return nrmse, X_pred, xs

def plot_latent(xs):
    plt.figure(figsize=(10,4))
    for d in range(min(3, xs.shape[1])):
        plt.plot(xs[:, d], label=f"x[{d}]")
    plt.title("Latent State Trajectory")
    plt.xlabel("Time")
    plt.ylabel("Latent value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

def plot_acc():
    plt.figure(figsize=(8,4))
    plt.boxplot([rand_test_err, pca_test_err, EM_test_err], labels=["Random Init", "PCA Init", "EM"])
    plt.ylabel("NRMSE")
    plt.title("Test Reconstruction Accuracy")
    plt.grid(alpha=0.3)
    plt.show()

def plot_em_convergence(em):
    iters = np.arange(len(em.mse))
    plt.figure(figsize=(8,4))
    plt.plot(iters, em.mse, marker='o')
    plt.xlabel("EM iteration")
    plt.ylabel("NRMSE on training data")
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.title("EM Convergence")
    plt.grid(alpha=0.3)
    plt.show()

plot_em_convergence(em)

plot_reconstruction(X_test[0], mean, A_rand, C_rand, Q_rand, R_rand, x0_rand, v0_rand)
plot_reconstruction(X_test[0], mean, A_pca, C_pca, Q_pca, R_pca, x0_pca, v0_pca)
plot_reconstruction(X_test[0], mean, A_h, C_h, Q_h, R_h, x0_h, v0_h)

# plot_latent(test_lat[0])
plot_acc()

def plot_latent(xs):
    plt.figure(figsize=(10,4))
    for d in range(min(1, xs.shape[1])):
        plt.plot(xs[:, d], label=f"x[{d}]")
    plt.title("Latent State Trajectory")
    plt.xlabel("Time")
    plt.ylabel("Latent value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

plot_latent(EM_test_lat[0])
plot_latent(rand_test_lat[0])
plot_latent(pca_test_lat[0])