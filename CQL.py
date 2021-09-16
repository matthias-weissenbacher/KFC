import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CQL(object):
    def __init__(self, num_inputs, action_space,  args,device=device ):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        
        self.num_random = args.num_random
        
        self.policy_eval_start = args.policy_eval_start
        self.temp = args.temp
        self.min_q_weight = args.min_q_weight

        self.policy_type = args.policy_type
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = device

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.qf_lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.obs_upper_bound,self.obs_lower_bound = 0,0
        self.reward_lower_bound,self.reward_upper_bound=0,0
        
        self.with_lagrange = args.with_lagrange
        if self.with_lagrange:
            self.target_action_gap = args.lagrange_thresh
            self.log_alpha_prime = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_prime_optimizer = Adam([self.log_alpha_prime],lr=args.qf_lr)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.policy_lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.policy_lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.policy_lr)
            

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds1 , preds2 = network(obs_temp, actions)
        preds1 = preds1.view(obs.shape[0], num_repeat, 1)
        preds2 = preds2.view(obs.shape[0], num_repeat, 1)
        return preds1, preds2

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        new_obs_actions,new_obs_log_pi,_ = network.sample(obs_temp)

        return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)


    def update_parameters(self, batch, updates):
        # Sample a batch from memory
        state_batch = torch.FloatTensor(batch['observations']).to(self.device)
        action_batch = torch.FloatTensor(batch['actions']).to(self.device)
        next_state_batch = torch.FloatTensor(batch['next_observations']).to(self.device)
        reward_batch =  torch.FloatTensor(batch['rewards']).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(batch['terminals'].numpy()).to(self.device).unsqueeze(1)

        
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            
        
        
            
            
        qf1_pred, qf2_pred = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1_pred, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2_pred, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        
        ## add CQL
        random_actions_tensor = torch.FloatTensor(qf2_pred.shape[0] * self.num_random, action_batch.shape[-1]).uniform_(-1, 1).to(self.device)
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(state_batch, num_actions=self.num_random, network=self.policy)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_state_batch, num_actions=self.num_random, network=self.policy)
        q1_rand , q2_rand = self._get_tensor_values(state_batch, random_actions_tensor, network=self.critic)
        q1_curr_actions , q2_curr_actions = self._get_tensor_values(state_batch, curr_actions_tensor, network=self.critic)
        q1_next_actions , q2_next_actions = self._get_tensor_values(state_batch, new_curr_actions_tensor, network=self.critic)
       
        cat_q1 = torch.cat(
            [q1_rand, qf1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
        )
        cat_q2 = torch.cat(
            [q2_rand, qf2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
        )
        std_q1 = torch.std(cat_q1, dim=1)
        std_q2 = torch.std(cat_q2, dim=1)

   
        # importance sammpled version
        random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
        cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
            )
        cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
            )
            
        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
                    
        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - qf1_pred.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - qf2_pred.mean() * self.min_q_weight
        
        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5 
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
            

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss

        qf_loss = qf1_loss + qf2_loss

        """
        Update critic
        """
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()


        
        #Policy Loss
        pi, log_pi, _ = self.policy.sample(state_batch)
        
         #entropy tuning for alpha
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            #alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            #alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        #Policy Loss...
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        if updates < self.policy_eval_start:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k 
            gradient steps here, or not having it
            """
            policy_log_prob = self.policy.log_prob(state_batch, action_batch)
            policy_loss = (self.alpha * log_pi - policy_log_prob).mean()

        """
        Update policy
        """
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        
        
       
        #soft updates 
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item()

    # Save model parameters

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optim.state_dict(), filename + "_critic_optimizer")

        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optim.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        self.critic_optim.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.policy.load_state_dict(torch.load(filename + "_actor.pth"))
        self.policy_optim.load_state_dict(torch.load(filename + "_actor_optimizer"))
