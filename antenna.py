import numpy as np
from numpy import linspace, sin, cos,exp, pi
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from numpy import random

class antenna:
    """
    docstring
    """

    def __init__(self, N = 10, n = 2000, d = 1/2, random_distr = False, antenna_type='non-directional'):
        """
        docstring
        """
        self.nfig = 1
        self.thetta_l = -90
        self.thetta_r = 90
        self.type = antenna_type
        self.Interference_list = []
        self.n = n
        self.__Grid(self.n)
        self.d = d
        self.N = N
        self.rand_distr= random_distr
        self.N_elements = np.matrix(np.linspace(0,N-1,N)).transpose()
        self.amplitude_distr = np.matrix(np.zeros([N,n]))
        self.d_shift = np.matrix(np.zeros([N])).transpose()
        for k in range(0,N):
            self.d_shift[k] = self.d * (self.N_elements[k] + np.random.randint(1,3))
        self.S_thetta = np.matrix(np.zeros([N,n]))
        pass

    def __Grid(self, n):
        """
        docstring
        """
        self.n = n
        self.Grid = np.matrix(np.linspace(self.thetta_l, self.thetta_r, n)*np.pi/180)
        pass


    def cosine_Amplitude_distr(self, power = 0, delta = 1):
        self.power = power
        self.delta = delta
        for k in range(self.n):
            for i in range(self.N):
                if self.rand_distr:
                    self.amplitude_distr[i,k] = delta - 1 + pow(cos(self.d_shift[i] * sin(self.Grid[0,k])), power)
                else:
                    self.amplitude_distr[i,k] = delta - 1 + pow(cos(i/2 * sin(self.Grid[0,k])), power)
        pass

        

    def axial_s(self):
        """
        docstring
        """
        if self.rand_distr:
            
            for i in range(self.n):
                for k in range(self.N):
                    self.S_thetta[k,i] = np.exp(1j * 2 * np.pi * self.d_shift[k,0] * sin(self.Grid[0,i]))
        else:
            self.S_thetta = np.exp(1j * 2 * np.pi * self.d * self.N_elements * sin(self.Grid))

        for i in range(0,self.n):
            for k in range(self.N):
                if self.rand_distr:
                    self.S_thetta[k,i] = self.S_thetta[k,i] * exp(1j * pi * self.d_shift[k,0] * sin(self.Grid[0,i])) / sin(self.Grid[0,i])
                else:
                    self.S_thetta[k,i] = self.S_thetta[k,i] * exp(1j * pi * 0.5 * self.N_elements[k] * sin(self.Grid[0,i])) / sin(self.Grid[0,i])

        return self.S_thetta

    def non_directional_s(self):
        """
        docstring
        """
        if self.rand_distr:
            #for i in range(0,self.n):
                #for k in range(self.N):
            self.S_thetta = exp(1j * 2 * np.pi * self.d_shift * sin(self.Grid))
        else:
            self.S_thetta = exp(1j * 2 * np.pi * self.d * self.N_elements * sin(self.Grid))
        return self.S_thetta

    def inclined_s(self):
        """
        docstring
        """
        #if self.rand_distr:
            #self.S_thetta = exp(1j * (2 * pi * self.d_shift * sin(self.Grid) - pi * self.d_shift) * sin(self.Grid))
        #else:
            #self.S_thetta = exp(1j * self.N_elements * (2 * pi * self.d * sin(self.Grid) - pi * self.d))
        #return self.S_thetta

    def calc_directional_diag(self):
        """
        docstring
        """
        
        self.s_switcher = {
            'non-directional' : self.non_directional_s(),

            'inclined'        : self.inclined_s(),

            'axial'           : self.axial_s()
        }

        try:
            self.amplitude_distr == None
        except AttributeError:
            self.cosine_Amplitude_distr()

            print('Amplitude distribution was set to 1')

        self.S_thetta = self.s_switcher.get(self.get_antenna_type())
        for i in range(0,self.n):
            for k in range(0,self.N):
                self.S_thetta[k,i] = self.S_thetta[k,i] * self.amplitude_distr[k,i]

        self.S_thetta_ = np.sum(self.S_thetta,axis = 0)
        #self.S_thetta_ = 10*np.log10(abs(self.S_thetta_))
        self.S_thetta_ = self.S_thetta_ / np.max(self.S_thetta_)
        self.S_thetta_db = 10*np.log10(abs(self.S_thetta_))

        pass

    def plot_dd(self):
        """
        docstring
        """
        
        try:
            self.S_thetta_ == None
        except AttributeError:
            self.calc_directional_diag()
        #self.calc_directional_diag()
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.plot(np.array(self.Grid).transpose(), np.array(self.S_thetta_db).transpose(), label='Directionl diagram', color = 'black')
        ax.set_title('Radiation diagram for ' + self.type + ' antenna with' + str(self.N) + ' elements')
        ax.legend(loc='upper left')
        ax.set_xlabel(r"$\dot{\Theta}$, rad")
        ax.set_ylabel('Pwr, dB')
        ax.grid()
        pass

    def plot_distr(self):
        """
        docstring
        """
        
        try:
            self.amplitude_distr == None
        except AttributeError:
            self.calc_directional_diag()

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.plot(np.array(self.Grid).transpose(), np.array(self.amplitude_distr).transpose(), label='amplitude distr')
        ax.set_title('Amplitude distr for ' + self.type + ' antenna with' + str(self.N) + ' elements')
        ax.legend(loc='upper left')
        ax.set_xlabel(r"$\dot{\Theta}$, rad")
        ax.set_ylabel('Pwr')
        ax.grid()
        pass

    def polar_plot(self):
        
        try:
            self.S_thetta_ == None
        except AttributeError:
            self.calc_directional_diag()

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111,projection='polar')
        ax.plot(self.Grid + pi/2, self.S_thetta_.transpose(), label='Directionl diagram')
        ax.set_title('Directional diagram for ' + self.type + ' antenna with' + str(self.N) + ' elements')
        ax.legend(loc='upper left')
        ax.set_xlabel(r"$\dot{\Theta}$, rad")
        ax.set_ylabel('Pwr')
        ax.grid(True)
        pass
        pass

    def set_interference(self, angular_loc,sigma):
        """
        docstring
        """
        self.placeholder = Interference(angular_loc,sigma,self.power, self.delta)
        if self.rand_distr:
            self.placeholder.set_params(self.d_shift,self.N_elements, self.Grid, self.N, self.type)
        else:
            self.placeholder.set_params(self.d,self.N_elements, self.Grid, self.N, self.type)

        self.Interference_list.append(self.placeholder)
        pass

    def reset_interference(self, angular_loc,sigma):
        """
        docstring
        """
        self.Interference_list = []
        self.placeholder = Interference(angular_loc,sigma,self.power, self.delta)
        if self.rand_distr:
            self.placeholder.set_params(self.d_shift,self.N_elements, self.Grid, self.N, self.type)
        else:
            self.placeholder.set_params(self.d,self.N_elements, self.Grid, self.N, self.type)
        self.Interference_list.append(self.placeholder)
        pass

    def get_interference_list(self):
        """
        docstring
        """
        self.dict_placeholder = {}
        for a in self.Interference_list:
            self.dict_placeholder.update({a.get_loc():a.get_sigma()})
        return self.dict_placeholder

    def get_d_param(self):
        return self.d

    def get_Grid(self):
        return self.Grid

    def get_N_elements(self):
        return self.N_elements

    def get_antenna_type(self):
        """
        docstring
        """
        return self.type
        
    def __calc_overall_cov_matrix(self):
        """
        docstring
        """
        self.cov_m = np.matrix(np.zeros((self.N,self.N)))
        for a in self.Interference_list:
            self.cov_m = self.cov_m +a.get_cov_matrix()
        self.cov_m = self.cov_m + np.matrix(np.eye(self.N))
        pass

    def return_overall_cov(self):
        self.__calc_overall_cov_matrix()
        return self.cov_m
    
    def adapt_Capon(self):
        """
        docstring
        """
        self.__calc_overall_cov_matrix() 
        self.P = np.matrix(np.zeros([self.n,1]))
        for i in range(0,max(np.shape(self.Grid))):
            a = self.S_thetta[:,i]
            self.P[i] = float(abs(a.transpose().conj()*np.linalg.inv(self.cov_m)*a))

        self.P = 1./self.P[:]
        self.P_db = 10*np.log10(abs(self.P))#/max(abs(self.P))
        self.P_db = self.P_db-max(self.P_db)
        return self.P_db

    def Thermal_noise(self):
        """
        docstring
        """
        self.__calc_overall_cov_matrix() 
        self.P = (np.zeros(self.n))
        for i in range(0,max(np.shape(self.Grid))):
            a = self.S_thetta[:,i]
            placeholder = np.linalg.matrix_power(self.cov_m,-2)
            self.P[i] = float(abs(a.transpose().conj()*placeholder@a))
            
        self.P = 1/self.P[:]
        self.P_db = 10*np.log10(abs(self.P))#/max(abs(self.P))
        self.P_db = self.P_db-max(self.P_db)
        return self.P_db

    def MUSIC(self):
        """
        docstring
        """
        self.__calc_overall_cov_matrix() 
        self.P = np.zeros(max(np.shape(self.Grid)))

        _, V = np.linalg.eig(self.cov_m)
        Qn  = V[:,3:self.N]
        for i in range(max(np.shape(self.Grid))):
            av = self.S_thetta[:,i]
            self.P[i] = 1/np.sum((av.conj().transpose()@(Qn@Qn.conj().transpose())@av))
        self.P_db = 10*np.log10(abs(self.P))
        self.P_db = self.P_db-max(self.P_db)
        return self.P_db

    pass

class Interference:
    """
    docstring
    """
    def __init__(self,angular_loc,sigma,power, delta):
        self.sigma = sigma
        self.UI = np.sin(angular_loc)
        self.power = power
        self.delta = delta
        pass
    
    def set_params(self, d, N_elements, Grid,N, antenna_type_ = 'non-directional'):
        """
        docstring
        """
        self.N_elements = N_elements
        self.Grid = Grid
        self.N = N
        if np.shape(d) == ():
            self.d=np.matrix(np.zeros([N])).transpose()
            for a in range(N):
                self.d[a] = d*N_elements[a]
        else:
            self.d = d
        self.antenna_type = antenna_type_
        self.covariance_matrix = np.zeros((self.N,self.N))
        self.VI = (np.matrix(np.zeros([N,1]),dtype=np.complex))
        self.amplitude_distr = np.matrix(np.zeros([N]))
        pass

    def axial_V(self):
        """
        docstring
        """
        for i in range(self.N):
            self.VI[i,0] = exp(1j * 2 * pi * (self.d[i,0]) * self.UI)
        return self.VI

    def non_directional_V(self):
        """
        docstring
        """
        for i in range(self.N):
            self.VI[i,0] = exp(1j * 2 * pi * (self.d[i,0]) * self.UI)
        return self.VI

    def inclined_V(self):
        """
        docstring
        """
        for i in range(self.N):
            self.VI[i,0] = sin((self.d[i,0]) * ( 2 * pi * self.UI -  pi))
        return self.VI

    def calc_cov_matrix(self):
        """
        docstring
        """
        self.v_switcher = {
            'non-directional' : self.non_directional_V(),

            'inclined'        : self.inclined_V(),

            'axial'           : self.axial_V()
        }

        self.VI = self.v_switcher.get(self.antenna_type)

        for k in range(self.N):
            self.amplitude_distr[0,k] = self.delta + pow(cos(pi/2 * self.d[k] * sin(self.UI)), self.power)# - pow(cos(pi/4),self.power)
            self.VI[k,0] = 10**(self.sigma/10)*self.VI[k,0] * self.amplitude_distr[0,k]

        self.covariance_matrix = 10**(self.sigma/10)*self.VI*(self.VI.transpose().conj())
        pass

    def get_cov_matrix(self):
        """
        docstring
        """
        self.calc_cov_matrix()
        return self.covariance_matrix

    def get_VI(self):
        """
        docstring
        """
        return self.VI

    def get_sigma(self):
        """
        docstring
        """
        return self.sigma    