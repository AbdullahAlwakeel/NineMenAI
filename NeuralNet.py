#neural network
from numba import jitclass
from numba import njit
import numpy as np
import numpy.random as rng
import random
from numba import int32, int64, boolean, uint8, float64    # import the types

#shape = [5,2,3]
#biases = [[0,0,0,0,0], [0,0], [0,0,0]]
spec = [
    ("NInputs", int64),
    ("NMid", int64),
    ("NOutputs", int64),
    ("Biases", float64[:]),
    ("OBiases", float64[:]),
    ("InputMidWeights", float64[:, :]),
    ("MidOutWeights", float64[:, :]),
    ("LearningRate", float64)
]
@jitclass(spec)
class NeuralNet():
    def __init__(self, NInputs, NumMid, NOutputs):
        self.NInputs = NInputs
        self.NMid = NumMid
        self.NOutputs = NOutputs
        self.Biases = (2*rng.rand(NumMid))-1
        self.OBiases = (2*rng.rand(NOutputs))-1
        self.InputMidWeights = (2*rng.rand(NInputs, NumMid))-1
        self.MidOutWeights = (2*rng.rand(NumMid, NOutputs)-1)
        self.LearningRate = 0.1
    
    def Update(self, Inputs):
        Mids = np.tanh(np.dot(Inputs,self.InputMidWeights)+self.Biases)
        Outs = np.tanh(np.dot(Mids,self.MidOutWeights)+self.OBiases)
        return Outs
    
    def Update_GetLayers(self, Inputs):
        Mids = np.tanh(np.dot(Inputs,self.InputMidWeights)+self.Biases)
        Outs = np.tanh(np.dot(Mids,self.MidOutWeights)+self.OBiases)
        return Outs, Mids
    
    def Mutate(self):
        MutationRate = 0.02
        self.Biases += ((2*rng.rand(self.NMid))-1)*MutationRate
        self.OBiases += ((2*rng.rand(self.NOutputs))-1)*MutationRate
        self.InputMidWeights += ((2*rng.rand(self.NInputs, self.NMid))-1)*MutationRate
        self.MidOutWeights += ((2*rng.rand(self.NMid, self.NOutputs))-1)*MutationRate
    
    def Copy(self, Other):
        CopyRate = 0.1
        for b in range(self.NMid):
            if random.random() <= CopyRate:
                self.Biases[b] = Other.Biases[b]
        for b in range(self.NOutputs):
            if random.random() <= CopyRate:
                self.OBiases[b] = Other.OBiases[b]
        
        for i in range(self.NInputs):
            for m in range(self.NMid):
                if random.random() <= CopyRate:
                    self.InputMidWeights[i][m] = Other.InputMidWeights[i][m]
        
        for m in range(self.NMid):
            for o in range(self.NOutputs):
                if random.random() <= CopyRate:
                    self.MidOutWeights[m][o] = Other.MidOutWeights[m][o]
    
    def tanhPrime(self, n):
        return 1 - (np.tanh(n)**2)
    
    def Learn(self, I, O):
        NetOut, MidOut = self.Update_GetLayers(I)
        #using square error
        Err = np.sum(0.5 * ((NetOut-O)**2))
        dErr_dOut = (NetOut-O)
        dErr_dOutAct = dErr_dOut * self.tanhPrime(np.dot(MidOut,self.MidOutWeights)+self.OBiases) #dErr dOutBiases also
        tmp = np.reshape(dErr_dOutAct, (self.NOutputs,1))
        t_m = np.reshape(MidOut, (1,self.NMid))
        dErr_dWOut = np.dot(tmp,t_m).T #got emmm middle to out weights

        dErr_dMidOut = np.dot(self.MidOutWeights,dErr_dOutAct)
        dErr_dMidAct = dErr_dMidOut * self.tanhPrime(np.dot(I,self.InputMidWeights)+self.Biases) #also d err d biases
        tmp_m = np.reshape(dErr_dMidAct, (self.NMid, 1))
        t_i = np.reshape(I, (1, self.NInputs))
        dErr_dWMid = np.dot(tmp_m,t_i).T


        #gradient descent now:
        self.InputMidWeights -= (self.LearningRate*dErr_dWMid)
        self.Biases -= (self.LearningRate*dErr_dMidAct)
        self.MidOutWeights -= (self.LearningRate*dErr_dWOut)
        self.OBiases -= (self.LearningRate*dErr_dOutAct)

        return Err


import pickle
def SaveNet(N, Location):
    NetDict = {"NInputs":N.NInputs, "NMid":N.NMid, "NOutputs":N.NOutputs, "Biases":N.Biases, "OBiases":N.OBiases, "InputMidWeights":N.InputMidWeights, "MidOutWeights":N.MidOutWeights, "LearningRate":N.LearningRate}
    with open(Location, "wb") as f:
        pickle.dump(NetDict, f)

def LoadNet(Location):
    NetDict = {}
    with open(Location, "rb") as f:
        NetDict = pickle.load(f)
    
    N = NeuralNet(NetDict["NInputs"], NetDict["NMid"], NetDict["NOutputs"])
    N.Biases = NetDict["Biases"]
    N.OBiases = NetDict["OBiases"]
    N.InputMidWeights = NetDict["InputMidWeights"]
    N.MidOutWeights = NetDict["MidOutWeights"]
    N.LearningRate = NetDict["LearningRate"]
    return N


if __name__ == "__main__":
    N = NeuralNet(2,3,4)
    for i in range(100):
        print(N.Learn(np.array([0.0,1.0]),np.array([0,0.5,0.7,0])))
        #print(N.Update([0,1]))
    print(N.Update(np.array([0.0,1.0])))
    SaveNet(N, "testnet.net")

    D = LoadNet("testnet.net")

    print("------------")
    print(N.Update(np.array([0.0,1.0])))
    print(D.Update(np.array([0.0,1.0])))


