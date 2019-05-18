from librosa.feature import mfcc
from librosa import load
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import pickle

def create_ceps(fn):
    y,sr = load(fn)
    ceps = mfcc(y,sr)
    ceps = ceps.T
    np.save(fn+".ceps.npy",ceps)

def read_ceps(n_people):
    X = np.zeros(20)
    y = np.array([[0]])
    for i in range(int(n_people)):
        fn = "./group/%d.wav.ceps.npy"%i
        ceps = np.load(fn)
        X = np.vstack((X,ceps))
        t = i * np.ones((ceps.shape[0],1),dtype="int8")
        y = np.vstack((y,t))
    return np.array(X),np.array(y)

if __name__ == "__main__":
    #人数 を入れてください
    n_people = 3
    n_people = int(n_people)
    for i in range(n_people):
        create_ceps("./group/%d.wav"%i)
    x,y = read_ceps(n_people)

    x = x[1:,:]
    y = y[1:,:]
    y = y.reshape(-1,)

    x,y = resample(x,y,n_samples=len(y))
    sc = StandardScaler()
    sc.fit(x)
    x = sc.transform(x)
    svc = SVC(kernel="rbf",random_state=0,gamma=0.2,C=1.0)
    svc.fit(x,y)
    
    pickle.dump(svc,open("./group/model.sav","wb"))
    pickle.dump(sc,open("./group/sc.sav","wb"))

    """
    svc.fit(x[200:,:],y[200:])
    print(svc.score(x[:200,:],y[:200]))
    prediction = svc.predict(x[:200,:])
    """

