import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from scipy import linalg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        


#from fasth_wrapper import Orthogonal as Orthogonal_Fast
def solve_sylvester(Mat):
    #print(Mat.shape)
    eM = np.eye(Mat.shape[0]) 
    #eM = np.ones(Mat.shape)
    commuter = linalg.solve_sylvester(Mat,-Mat,eM)
    return 1e-5*commuter/commuter.mean()



def solve_eig(Mat):
    dim = Mat.shape[0]
    try: 
        res = linalg.eig(Mat)
        eigenvalues = np.diag(res[0])
        U = res[1]     
    except:
        eigenvalues = np.zeros((dim,dim))
        U = np.eye(dim)
    
    try:
        U_inv = np.linalg.inv(U)
    except:
        U_inv = np.eye(dim) 
    

    return  [U , eigenvalues , U_inv]

class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape,self).__init__()
        
        self.shape = shape
        
        
    def forward(self,input):
        shape = (input.shape[0],self.shape[0],self.shape[1])
        return torch.reshape(input, shape)

      

class MultiplyVec(torch.nn.Module):
    def __init__(self, action_dim, latent_dim,device):
        super(MultiplyVec,self).__init__()
        
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.device = device
        
    def forward(self,vec1,vec2):
        out = torch.zeros((vec1.shape[0],self.latent_dim)).to(self.device)
        
        for i in range(0,self.action_dim):
            out += vec1[:,i:(i+1)]*vec2[:,i*self.latent_dim:(i+1)*self.latent_dim]
            
        return out

class MultiplyVecMat(torch.nn.Module):
    def __init__(self, action_dim, latent_dim,device):
        super(MultiplyVecMat,self).__init__()
        
        self.device = device
        
        self.action_dim = action_dim
        self.latent_dim = latent_dim
         
        
    def forward(self,vec,mat):
        out = torch.zeros((vec.shape[0],self.latent_dim,self.latent_dim)).to(self.device)
        
        for i in range(0,self.action_dim):
            Bi= torch.reshape(mat[i*self.latent_dim:(i+1)*self.latent_dim,:],(self.latent_dim,self.latent_dim,1))
            out += torch.transpose(torch.transpose(torch.matmul(Bi, torch.transpose(vec[:,i:(i+1)],0,1)),2,1) ,0,1)
            
        return out

#Koopman Encode - Decoder
class MLP_Koopman_Bilinear(torch.nn.Module): 
    def __init__(self, env, latent_dim,hidden_dim = 512, device =device): 
        super(MLP_Koopman_Bilinear, self).__init__()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        self.action_dim = action_dim
        
        self.latent_dim = latent_dim
        
        self.device = device

        self.layer1 = nn.Linear(state_dim,hidden_dim ,bias = True)
        self.layer2 = nn.Linear(hidden_dim ,hidden_dim ,bias = True)
        self.layer3 = nn.Linear(hidden_dim ,latent_dim,bias =  True)
        #self.layer4 = nn.Linear(latent_dim ,latent_dim,bias =  False)
          
        self.layerK = nn.Linear(latent_dim ,latent_dim,bias = False)
        
        self.layerBi = nn.Linear(latent_dim ,latent_dim*action_dim,bias = False)
        self.sum_a_B = MultiplyVec(action_dim,latent_dim,device)
        
        self.getK = MultiplyVecMat(1,self.latent_dim,device)
        self.getBis = MultiplyVecMat(self.action_dim,self.latent_dim,device)
        
        self.layer3inv = nn.Linear(latent_dim, hidden_dim, bias = True)
        self.layer2inv = nn.Linear(hidden_dim ,hidden_dim ,bias = True)
        self.layer1inv = nn.Linear(hidden_dim,state_dim ,bias = True)
        
             
        
        self.obs_upper_bound = float(env.observation_space.high[0]) #state space upper bound
        self.obs_lower_bound = float(env.observation_space.low[0])  #state space lower bound
        #self.max_action = float(env.action_space.high[0])
        
        self.activation = nn.Tanh() #aReLU() #nn.Tanh()#nn.ReLU()
        
    
    
    def forward(self,state, action,choice = True):
   
        
        act = self.activation
        #Function approximation to Koopman space
        y = act(self.layer1(state))
        y = act(self.layer2(y))
        gt = act(self.layer3(y))
        #gt = self.layer4(y)
        
        #gt = self.layer3(y)
        
        if choice:
            gtp1 = self.layerK(gt) + self.sum_a_B(action,self.layerBi(gt))
        else:
            gtp1 = gt
            
        
        #inverse Function approximation from  Koopman space to state space
        z = act(self.layer3inv(gtp1))
        #z = seaf.layer3.inverse(gtp1)
        z = act(self.layer2inv(z))
        z = self.layer1inv(z)
        
        next_state = z.clamp(self.obs_lower_bound ,self.obs_upper_bound)


        return next_state
    
    def Encoder_obs(self,state):
        
      
        act =  self.activation
        #Function approximation to Koopman space
        y = act(self.layer1(state))
        y = act(self.layer2(y))
        gt = act(self.layer3(y))
        #Agt =  self.layerA(gt)
        
        return gt

    
    def Decoder_obs(self,g_state):
        
        act =  self.activation
         #inverse Function approximation from  Koopman space to state space
        z = act(self.layer3inv(g_state))
        #z = seaf.layer3.inverse(gtp1)
        z = act(self.layer2inv(z))
        z = self.layer1inv(z)
        
        return z.clamp(self.obs_lower_bound ,self.obs_upper_bound)
    
    
    def Symmetry_Encoder_Decoder(self,state,symmetry_op,ep):
      
        gt = self.Encoder_obs(state)
        
        gt_shift = torch.nn.Flatten()(torch.matmul(symmetry_op, Reshape((self.latent_dim,1))(gt)))
        gt_shift = gt +  ep*gt_shift
        
        st_shift = self.Decoder_obs(gt_shift)
        
        return st_shift
    
    def Symmetry_Encoder_Decoder_Eigenspace(self,state,symmetry_op,ep):
      
        gt = self.Encoder_obs(state)
 
        U = symmetry_op[:,0]
        U_inv = symmetry_op[:,2]
        #the complex multiplication done on CPU
        sigma_a = torch.matmul(torch.matmul(U,ep),U_inv)
        #the floattensor redners the cmplex entries real.
        sigma_a = torch.FloatTensor(np.real(np.array(sigma_a))).to(state.device)
        gt_shift = torch.nn.Flatten()(torch.matmul(sigma_a, Reshape((self.latent_dim,1))(gt)))
        
        
        gt_shift = gt +  gt_shift
        
        st_shift = self.Decoder_obs(gt_shift)
        
        return st_shift
    
    def Symmetry_Encoder_Decoder_Gauged(self,state,symmetry_op,ep):
      
        gt = self.Encoder_obs(state)
        
        gt_shift = torch.nn.Flatten()(torch.matmul(symmetry_op, Reshape((self.latent_dim,1))(gt)))
        gt_shift = gt +  ep*gt_shift
        
        #Compute systematic error
        diff_gauge =  state - self.Decoder_obs(gt) 
        
        st_shift = self.Decoder_obs(gt_shift) + diff_gauge
        
        return st_shift.clamp(self.obs_lower_bound ,self.obs_upper_bound)
    
    
    def Symmetry_Generator(self,action):
      
        action_Bi_weight = self.getBis(action,self.layerBi.weight.data)
        
        
        K_weight =  self.getK(torch.ones((action.shape[0],1)).to(self.device),self.layerK.weight.data)
                                                     
        weight_data = K_weight + action_Bi_weight 
        weight_data = np.array(weight_data.cpu().detach())
        #print(weight_data.shape)
        symmetry_gens = np.array(list(map(solve_sylvester,weight_data)))
        

        return symmetry_gens
    

    def Symmetry_Generator_Eigenspace(self,action):
      
        action_Bi_weight = self.getBis(action,self.layerBi.weight.data)
        
        
        K_weight =  self.getK(torch.ones((action.shape[0],1)).to(self.device),self.layerK.weight.data)
                                                     
        weight_data = K_weight + action_Bi_weight 
        weight_data = np.array(weight_data.cpu().detach())
        #print(weight_data.shape)
        symmetry_gens = np.array(list(map(solve_eig,weight_data)))
        

        return symmetry_gens
    




