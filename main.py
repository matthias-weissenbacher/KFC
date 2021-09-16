import argparse
import datetime
import gym
import numpy as np
import itertools
import os
import json
import pandas as pd
import torch
import CQL
import CQL_Noise
import time


import d4rl # Import required to register environments
from utils import load_replaymemory_dataloader



def eval_policy(policy, env_name, eval_episodes=10):
    eval_env = gym.make(env_name)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state),evaluate=True)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward



parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env', default="halfcheetah-medium-v0",
                    help='D4rl Mujoco Gym environment (default: halfcheetah-medium-v0)')
parser.add_argument('--policy_type', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--policy', default="CQL",
                    help='Policy name CQL or KFC_QL')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=5e-3, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--policy_lr', type=float, default=1e-4, metavar='G',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--qf_lr', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')

parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')

parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--sys_hidden_size', type=int, default=512, metavar='N',
                    help='sys_hidden_size (default: 512)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')

parser.add_argument("--eval_freq", default=5e3, type=int, help="evaluation frequency")
parser.add_argument("--training_mode", default="Offline", help="Online Training or Offline Training")
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

parser.add_argument("--save_model", default="False",help="Save training models")
parser.add_argument("--load_model", default="" ,help="Loding model or not")
parser.add_argument("--cuda_device" , default= 1)    
parser.add_argument("--comment" , default= "none")    
parser.add_argument("--pause_hour" , default= 0,type=int,help="pasue run of script for pause_hour hours.")      
    

#CQL specific hyperparamters
parser.add_argument("--policy_eval_start", default=40000, type=int)       # Defaulted to 20000 (40000 or 10000 work similarly)
parser.add_argument('--min_q_weight', default=10.0, type=float)    # the value of "alpha"  in CQL Paper, set to 5.0 or 10.0 if not 
parser.add_argument('--temp', default=1.0, type=float)    #
parser.add_argument('--num_random', default=10, type=int)    #     number of random samples for Q-minimizer

parser.add_argument('--with_lagrange', default=True, type=bool)   
parser.add_argument('--lagrange_thresh', default=10.0, type=float)   
parser.add_argument('--noise_sigma', type=float, default=3e-3, metavar='G',help='noise_sigma  (default: 0.003)')



args = parser.parse_args()


if args.pause_hour > 0: # Pause until a certain time
    time.sleep(3600.0*args.pause_hour)
    
policy_name = args.policy
if args.comment != "none":
    policy_name = args.policy + args.comment
        
file_name = f"{policy_name}_{args.env}_{args.training_mode}"
print("---------------------------------------")
print(f"Policy: {policy_name}, Env: {args.env},Training_mode: {args.training_mode} ")
print("---------------------------------------")

torch.cuda.set_device(int(args.cuda_device))
device_name = str("cuda:" + str(args.cuda_device))
print("The current device is: ", device_name )

device = torch.device( device_name   if torch.cuda.is_available() else "cpu")

if args.save_model == "True" and not os.path.exists("./models"):
    os.makedirs("./models")
    
# Loading Environment
env = gym.make(args.env)
env.seed(0)
env.action_space.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# Agent
if args.policy == 'CQL':
    agent = CQL.CQL(env.observation_space.shape[0], env.action_space, args, device =device)
elif args.policy == 'CQL_Noise':
    agent = CQL_Noise.CQL(env, env.observation_space.shape[0], env.action_space, args, device =device)

    
    
#load Replay memorey from d4rl into troch.Dataloader shuffle =True
Replay_memory_loader = load_replaymemory_dataloader(env,batch_size =  args.batch_size) # ReplayMemory(args.replay_size, 

# Training Loop
total_numsteps = 0
updates = 0
evaluations = [] #[eval_policy(agent, args.env)]
agent.update_sys = 0


#Load dataset and initiate dataloader
dataset = d4rl.qlearning_dataset(env)

data_arranged = []
obs= dataset['observations']
act = dataset['actions']
next_obs = dataset['next_observations']

if args.policy == "CQL":
    variant = dict(
        algorithm='CQL',
        env=args.env,
    )
elif args.policy == "CQL_Noise":
    variant = dict(
        algorithm='CQL_Noise',
        env=args.env,
    )


if not os.path.exists(f"./data/{args.env}/{policy_name}"):
    os.makedirs(f'./data/{args.env}/{policy_name}')

with open(f'./data/{args.env}/{policy_name}/variant.json', 'w') as outfile:
    json.dump(variant,outfile)
    


# ----------------------- Main training Loop # ----------------------- # -----------------------
training_step = 0
print("Training initiated...")
while training_step  < args.num_steps: 
            # Number of updates per step in environment
        for batch in Replay_memory_loader:
                training_step +=1
                # Update parameters of all the networks
                agent.update_parameters(batch, training_step)
                if (training_step) % args.eval_freq == 0:
                        print("---------------------------------------")
                        print("---------------------------------------")
                        print("Trainingstep:" , training_step)
                        eval_reward = eval_policy(agent, args.env)
                        #print(" - - - - - - - - - - - - - - - - - - - - - - - - - ")
                        #print("Trainingstep: %2.i  - policy reward returned: %3.f"%(training_step,eval_reward))
                        evaluations.append(eval_reward)
                        if args.save_model == "True":
                            agent.save(f"./models/{file_name}")

                        data = np.array(evaluations)
                        df = pd.DataFrame(data=data,columns=["Average Return"]).reset_index()
                        df['Timesteps'] = df['index'] * args.eval_freq
                        df['env'] = args.env
                        df['algorithm_name'] = args.policy
                        df.to_csv(f'./data/{args.env}/{policy_name}/progress.csv', index = False)


# ----------------------- # ----------------------- # -----------------------
env.close()
