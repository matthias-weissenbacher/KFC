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

import KFC_prime

import time


import d4rl # Import required to register environments
from utils import load_replaymemory_dataloader, load_replaymemory_dataloader_symmetry , load_replaymemory_dataloader_symmetry2 ,load_replaymemory_dataloader_train_test



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


def eval_policy2(policy, env_name, eval_episodes=10,seedNr = 1):
    total_avg_reward =0.
    for seed in range(0,seedNr):
        eval_env = gym.make(env_name)
        eval_env.seed(seed)
        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = policy.select_action(np.array(state),evaluate=True)
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes
        total_reward +=avg_reward
        
    total_avg_reward /=  seedNr   
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {total_avg_reward:.3f}")
    print("---------------------------------------")
    return total_avg_reward



parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env', default="halfcheetah-medium-v0",
                    help='D4rl Mujoco Gym environment (default: halfcheetah-medium-v0)')
parser.add_argument('--policy_type', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--policy', default="KFC_prime",
                    help='Policy name KFC_prime')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=5e-3, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--policy_lr', type=float, default=1e-4, metavar='G',
                    help='learning rate (default: 0.0003)')
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
parser.add_argument("--pause_hour" , default= 0,type=float,help="pasue run of script for pause_hour hours.")      
    

#CQL specific hyperparamters
parser.add_argument("--policy_eval_start", default=40000, type=int)       # Defaulted to 20000 (40000 or 10000 work similarly)
parser.add_argument('--min_q_weight', default=10.0, type=float)    # the value of alpha, set to 5.0 or 10.0 if not 
parser.add_argument('--temp', default=1.0, type=float)    #
parser.add_argument('--num_random', default=10, type=int)    #     number of random samples for Q-minimizer

parser.add_argument('--with_lagrange', default=True, type=bool)   
parser.add_argument('--lagrange_thresh', default=10.0, type=float)   

# for KFC_QL
parser.add_argument('--koopman_augmentation', default=True, type=bool)   
parser.add_argument('--policy_forwards', default=0, type=int)         
parser.add_argument('--gamma2', type=float, default=0.25, metavar='G',
                    help='discount factor for reward (default: 0.9)')
parser.add_argument('--shift_sigma', type=float, default=1.0, metavar='G',
                    help='shift for random vectoe symmetries(default: 1)')
parser.add_argument('--koopman_probability', type=float, default=0.8, metavar='G',
                    help=' probablity [0,1] of using koopman symmetry augmenatation of states(default: 0.5)')
parser.add_argument('--latent_dim', type=int, default=32, metavar='N',
                    help='latent dimension Koopman (default: 32)')
parser.add_argument('--epochs_forward_model', type=int, default=400, metavar='N',
                    help='Koopman Forward model epochs for training(default: 1000)')
parser.add_argument('--symmetry_type', default="Eigenspace",
                    help='symmetry_type: Eigenspace | Sylvester (default: Eigenspace)')
#parser.add_argument('--load_sysmodel', default=False, type=bool)   
parser.add_argument('--sysmodel_dic_name', default="None")   
args = parser.parse_args()


if args.pause_hour > 0: # Pause until a certain time
    time.sleep(3600.0*args.pause_hour)
    
policy_name = args.policy
if args.comment != "none":
    policy_name = args.policy + args.comment + args.symmetry_type
        
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

if args.policy == 'KFC_prime':
    agent = KFC_prime.KFC_QL(env, args, device =device)

    

# Training Loop
total_numsteps = 0
updates = 0
evaluations = [] #[eval_policy(agent, args.env)]
agent.update_sys = 0


#Load dataset and initiate dataloader
#dataset = d4rl.qlearning_dataset(env)




if args.policy == "KFC_prime":
    variant = dict(
        algorithm='KFC_prime',
        env=args.env,
    )

    
if not os.path.exists(f"./data/{args.env}/{policy_name}"):
    os.makedirs(f'./data/{args.env}/{policy_name}')

with open(f'./data/{args.env}/{policy_name}/variant.json', 'w') as outfile:
    json.dump(variant,outfile)
    
    


# ----------------------- Pre training Loop for Forcast Sysmodel/VAE # ----------------------- # -----------------------
Replay_memory_loader , test_memory_loader = load_replaymemory_dataloader_train_test(env,ratio = 0.7, batch_size =  args.batch_size)

epochs = args.epochs_forward_model
if args.sysmodel_dic_name !=  "None":
    epochs = 0
    print("Loading sysmodel from file: ", args.sysmodel_dic_name)
    try:
        agent.sysmodel.load_state_dict(torch.load("./pretrained_sysmodels/" +args.sysmodel_dic_name))
    except:
        print("Failed loading file - training model instead")
        epochs = args.epochs_forward_model
        
train_vs_test = []
train_vs_test_VAE = []
lr_rate  = agent.variable_lr 
for epoch in range(0,epochs): 
            # Number of updates per step in environment
    running_loss = 0.0
    running_loss_VAE = 0.0
    count = 0
    for batch in Replay_memory_loader:
        count += 1
        loss1, loss2 = agent.train_sysmodel(batch,epoch)
        running_loss += loss1
        running_loss_VAE += loss2
                      
    #opotimizer refinedmend
    if epoch > 100 and epoch%50 ==1:
            lr_rate = lr_rate / 2.0
            print("New learning rate: ",lr_rate )
            agent.sysmodel_optimizer = torch.optim.Adam(agent.sysmodel.parameters(), lr=lr_rate)
     
    epoch_avg_loss = running_loss/count
    epoch_avg_loss_VAE= running_loss_VAE/count
    test_avg_loss =  agent.eval_model(test_memory_loader )
    test_avg_loss_VAE =  agent.eval_model_VAE(test_memory_loader )
    train_vs_test.append(epoch_avg_loss/test_avg_loss )
    train_vs_test_VAE.append(epoch_avg_loss_VAE/test_avg_loss_VAE )
    #if epoch >= 74:  # 30
      #  score = sum(train_vs_test[-10:])/10
       # score_VAE =sum(train_vs_test_VAE[-10:])/10
       # print("score vs score VAE: %.2f / %.2f "%(score,score_VAE))
       # if score_VAE <= 0.80 and score <= 0.75:
       #     break
     #   elif score <= 0.70:
          #  break
       # elif score_VAE <= 0.75:
           # break 
    if epoch >= 75:
        #get averagr over last 15 episodes
        score = sum(train_vs_test[-5:])/5
        score_VAE =sum(train_vs_test_VAE[-5:])/5
        print("score vs score VAE: %.2f / %.2f "%(score,score_VAE))
        if score_VAE <= 0.90 and score <= 0.92: #  if score_VAE <= 0.90 and score <= 0.80:
            break
        elif score <= 0.90:# <= 0.78:
            break
        elif score_VAE <= 0.90: #
            break
        
    #test_avg_loss = 0
    print("epoch %d - train/test loss: %.6f / %.6f  - train/test loss VAE %.7f /  %.7f " %(epoch,epoch_avg_loss,test_avg_loss,epoch_avg_loss_VAE,test_avg_loss_VAE ))



        
#generate symmetry operation of data
if args.koopman_augmentation:
    print("Generating symmetries of dynamical system in Koopman space.")
    #load Replay memorey from d4rl into troch.Dataloader shuffle =True
    if args.symmetry_type == "Sylvester":
        Replay_memory_loader = load_replaymemory_dataloader_symmetry(env,agent.sysmodel,device,batch_size =  args.batch_size) 
    elif args.symmetry_type == "Eigenspace":
        Replay_memory_loader = load_replaymemory_dataloader_symmetry2(env,agent.sysmodel,device,batch_size =  args.batch_size) # 
else:
    #load Replay memorey from d4rl into troch.Dataloader shuffle =True
    Replay_memory_loader = load_replaymemory_dataloader(env,batch_size =  args.batch_size) # ReplayMemory(args.replay_size, 


        
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
