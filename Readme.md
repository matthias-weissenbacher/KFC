# CQL and KFC
Self-contained implementation of  KFC, CQL and S4RL (CQL-Noise). Based on SAC. Copyright Matthias Weissenbacher. Please do not distribute.

# Requirements
Tested on [D4RL](https://github.com/rail-berkeley/d4rl) and [Mujoco](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://gym.openai.com/). 


# Usage
To run the baseline experiments run the following for CQL:
```
nohup python3 -u main.py \
        --env  "halfcheetah-medium-v0" \
        --policy "CQL" \
        --cuda_device 0 \
         &>  hc.out &
```
and for CQL-noise:
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
