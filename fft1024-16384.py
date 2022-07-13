import torch
import numpy as np
import math
import cmath

class fft16(object):
    def __init__(self,F):
        self.F = F
        self.N = len(F)
        #self.pi = math.pi
        self.w = cmath.exp(2*math.pi*1j/self.N).conjugate()
    def partition(self):
        F = self.F
        x = np.zeros([1024,16], dtype='complex')
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for m in range(2):
                        temp = F[i::2][j::2][k::2][m::2]
                        first_res = torch.fft.fft(torch.tensor(temp))
                        #x[:, (i*8 + j*4 + k*2 + m)] = temp
                        x[:, (i*8 + j*4 + k*2 + m)] = first_res.resolve_conj().numpy()
        return x
    def merge(self, F_even, F_odd):
        #if len(F_even) != len(F_odd):
            #break
        w = self.w
        F_even = F_even.reshape(-1)
        F_odd = F_odd.reshape(-1)

        N_half = len(F_even)
        D_half = np.zeros(N_half,dtype='complex')
        for i in range(N_half):
            D_half[i] = w**i

        N = 2 * N_half
        res = np.zeros(N,dtype='complex')
        res[0:N_half] = F_even + D_half * F_odd
        res[N_half:2*N_half] = F_even - D_half * F_odd

        return res
    def get_res(self):
        x = self.partition()
        temp = np.zeros([2048, 8],dtype='complex')
        for i in range(8):
            temp[:,i] = self.merge(x[:,2*i], x[:,2*i+1]).reshape(-1)
        x = temp
        temp = np.zeros([4096, 4],dtype='complex')
        for i in range(4):
            temp[:,i] = self.merge(x[:,2*i], x[:,2*i+1]).reshape(-1)
        x = temp
        temp = np.zeros([8192, 2],dtype='complex')
        for i in range(2):
            temp[:,i] = self.merge(x[:,2*i], x[:,2*i+1]).reshape(-1)
        x = self.merge(temp[:,0], temp[:,1])
        return x/self.N
        
        

a = list(range(16384))
aa = fft16(a)
b = aa.get_res()
print(b)
