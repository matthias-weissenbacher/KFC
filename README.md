# KFC
Koopman Forward Conservative (KFC) Q-learning from the paper [Koopman Q-learning: Offline Reinforcement Learning via Symmetries of Dynamics](https://arxiv.org/abs/2111.01365) [ICML2022](https://proceedings.mlr.press/v162/weissenbacher22a/weissenbacher22a.pdf).

# CQL and KFC
Self-contained implementation of  [KFC](https://arxiv.org/abs/2111.01365), [CQL](https://arxiv.org/abs/2006.04779) and [S4RL](https://arxiv.org/abs/2103.06326) (CQL-Noise). Based on SAC. Supplemnatry code to the KFC-paper.

# Disclaimer
The CQL codebase differs from the one used in the paper; thus this code won't reproduce the reults in the KFC-paper. We may release the full CQL codebase in the future. Meanhwile have fun & stay foolish;) THe KFC part is indentical to the one in the paper so it may be integrated into your own propoerly benchmarked CQL-codebase.

# Requirements
Tested on [D4RL](https://github.com/rail-berkeley/d4rl) and [Mujoco](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://gym.openai.com/). 


# Usage
To run the CQL-"baseline" experiments run the following for CQL:
```
nohup python3 -u main.py \
        --env  "halfcheetah-medium-v0" \
        --policy "CQL" \
        --cuda_device 0 \
         &>  hc.out &
```
and for S4RL i.e. CQL-noise:
```
nohup python3 -u main.py \
        --env  "halfcheetah-medium-v0" \
        --policy "CQL_Noise" \
        --cuda_device 0 \
         &>  hc.out &
```


To run the main KFC experiments of this paper run e.g. for the halfcheetah-medium-v0 environment:
```
nohup python3 -u main_kfc.py \
                --env  "halfcheetah-medium-v0"   \
                --policy "KFC_prime" \
                --symmetry_type "Eigenspace" \
                --koopman_probability 0.8 \
                --cuda_device 0 \
                --shift_sigma 0.5 \
                &> hc_m.out &

```
Choose for the KFC'' algorithm: 
```
 --symmetry_type "Eigenspace" \
 ```
or for the KFC' algorithm:
 ```
 --symmetry_type "Sylvester" \
 ```
The standard deviation of the random variable is set by:
 ```
 --shift_sigma 0.5 \
 ```
 where it is in units of $3 x 10^{-3}$ and $1 x 10^{-5}$ for the KFC' and KFC'', respectively.
