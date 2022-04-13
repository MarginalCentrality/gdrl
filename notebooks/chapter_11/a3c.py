import warnings ; warnings.filterwarnings('ignore')
import os
# os.environ is a mapping object where keys and values are strings that represent the process environments. 

# order the GPU device by PCI_BUS_ID
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# set os.environ['CUDA_VISIBLE_DEVICES'] = '0' to only use '/gpu:0'
# set os.environ['CUDA_VISIBLE_DEVICES'] = '' to do not use gpu
os.environ['CUDA_VISIBLE_DEVICES']=''

# OMP_NUM_THREADS is an option for OpenMP, a c/c++/fortan api for doing multi-threading within a process 
# the option avoid python multiprocessing to call another process that is doing multiprocess
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
# No use in this python notebook
import threading
import numpy as np

# IPython(Interactive python) provides a rich toolkit to help you make the most of using Python interactively.
# IPython.display.Display is to display a python object in all frontends.
from IPython.display import display
from collections import namedtuple, deque
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab
from itertools import cycle, count
# wrap is used to truncate string
from textwrap import wrap

import matplotlib
import subprocess
import os.path
import tempfile
import random
import base64
import pprint
import glob
import time
import json
import sys
import gym
import io
import os
import gc

from gym import wrappers
from subprocess import check_output
from IPython.display import HTML

LEAVE_PRINT_EVERY_N_SECS = 30
# 清除光标所在行的所有字符
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')
SEEDS = (12, 34, 56, 78, 90)


# Try to replicate the styles from FiveThirtyEight.com.
plt.style.use('fivethirtyeight')
params = {
    'figure.figsize': (15, 8),
    'font.size': 24,
    'legend.fontsize': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
}
pylab.rcParams.update(params)
# suppress = True
# print floating point numbers using fixed point notation, in 
# which case numbers equal to zero in the current precision will print as zero.
np.set_printoptions(suppress=True)

# 将 make_env_fn 从 get_make_env_fn 中提到外面，以解决【错误】
#【错误】多进程运行 A3C 时出现 can't pickle local object 

def make_env_fn(env_name, seed=None, render=None, record=False,
                    unwrapped=False, monitor_mode=None, 
                    inner_wrappers=None, outer_wrappers=None):
        # Create a temporary directory in the most secure manner possible.
        mdir = tempfile.mkdtemp()
        env = None
        if render:
            # gym.make 只接受一个参数，此处必然产生异常
            try:
                env = gym.make(env_name, render=render)
            except:
                pass
        # 由于上面的异常，env is None 成立，执行 env = gym.make(env_name) 
        if env is None:
            env = gym.make(env_name)
        if seed is not None: env.seed(seed)
        env = env.unwrapped if unwrapped else env
        if inner_wrappers:
            for wrapper in inner_wrappers:
                env = wrapper(env)
        env = wrappers.Monitor(
            env, mdir, force=True, 
            mode=monitor_mode, 
            video_callable=lambda e_idx: record) if monitor_mode else env
        if outer_wrappers:
            for wrapper in outer_wrappers:
                env = wrapper(env)
        return env


def get_make_env_fn(**kargs):
    # def make_env_fn(env_name, seed=None, render=None, record=False,
    #                 unwrapped=False, monitor_mode=None, 
    #                 inner_wrappers=None, outer_wrappers=None):
    #     # Create a temporary directory in the most secure manner possible.
    #     mdir = tempfile.mkdtemp()
    #     env = None
    #     if render:
    #         try:
    #             env = gym.make(env_name, render=render)
    #         except:
    #             pass
    #     if env is None:
    #         env = gym.make(env_name)
    #     if seed is not None: env.seed(seed)
    #     env = env.unwrapped if unwrapped else env
    #     if inner_wrappers:
    #         for wrapper in inner_wrappers:
    #             env = wrapper(env)
    #     env = wrappers.Monitor(
    #         env, mdir, force=True, 
    #         mode=monitor_mode, 
    #         video_callable=lambda e_idx: record) if monitor_mode else env
    #     if outer_wrappers:
    #         for wrapper in outer_wrappers:
    #             env = wrapper(env)
    #     return env
    return make_env_fn, kargs

def get_videos_html(env_videos, title, max_n_videos=5):
    videos = np.array(env_videos)
    if len(videos) == 0:
        return
    
    n_videos = max(1, min(max_n_videos, len(videos)))
    idxs = np.linspace(0, len(videos) - 1, n_videos).astype(int) if n_videos > 1 else [-1,]
    videos = videos[idxs,...]

    strm = '<h2>{}<h2>'.format(title)
    for video_path, meta_path in videos:
        video = io.open(video_path, 'r+b').read()
        encoded = base64.b64encode(video)

        with open(meta_path) as data_file:    
            meta = json.load(data_file)

        html_tag = """
        <h3>{0}<h3/>
        <video width="960" height="540" controls>
            <source src="data:video/mp4;base64,{1}" type="video/mp4" />
        </video>"""
        strm += html_tag.format('Episode ' + str(meta['episode_id']), encoded.decode('ascii'))
    return strm

def get_gif_html(env_videos, title, subtitle_eps=None, max_n_videos=4):
    videos = np.array(env_videos)
    if len(videos) == 0:
        return
    
    n_videos = max(1, min(max_n_videos, len(videos)))
    idxs = np.linspace(0, len(videos) - 1, n_videos).astype(int) if n_videos > 1 else [-1,]
    videos = videos[idxs,...]

    strm = '<h2>{}<h2>'.format(title)
    for video_path, meta_path in videos:
        basename = os.path.splitext(video_path)[0]
        gif_path = basename + '.gif'
        if not os.path.exists(gif_path):
            ps = subprocess.Popen(
                ('ffmpeg', 
                 '-i', video_path, 
                 '-r', '7',
                 '-f', 'image2pipe', 
                 '-vcodec', 'ppm',
                 '-crf', '20',
                 '-vf', 'scale=512:-1',
                 '-'), 
                stdout=subprocess.PIPE)
            output = subprocess.check_output(
                ('convert',
                 '-coalesce',
                 '-delay', '7',
                 '-loop', '0',
                 '-fuzz', '2%',
                 '+dither',
                 '-deconstruct',
                 '-layers', 'Optimize',
                 '-', gif_path), 
                stdin=ps.stdout)
            ps.wait()

        gif = io.open(gif_path, 'r+b').read()
        encoded = base64.b64encode(gif)
            
        with open(meta_path) as data_file:    
            meta = json.load(data_file)

        html_tag = """
        <h3>{0}<h3/>
        <img src="data:image/gif;base64,{1}" />"""
        prefix = 'Trial ' if subtitle_eps is None else 'Episode '
        sufix = str(meta['episode_id'] if subtitle_eps is None \
                    else subtitle_eps[meta['episode_id']])
        strm += html_tag.format(prefix + sufix, encoded.decode('ascii'))
    return strm

class FCDAP(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dims=(32,32), 
                 activation_fc=F.relu):
        super(FCDAP, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x
        
    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)

    def full_pass(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logpa = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        is_exploratory = action != np.argmax(logits.detach().numpy())
        return action.item(), is_exploratory.item(), logpa, entropy

    def select_action(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()
    
    def select_greedy_action(self, state):
        logits = self.forward(state)
        return np.argmax(logits.detach().numpy())
    

class FCV(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dims=(32,32), 
                 activation_fc=F.relu):
        super(FCV, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)
    
    
class SharedAdam(torch.optim.Adam):
    # params: iterable of parameters to optimize or dicts defining parameter groups
    # lr: learning rate
    # betas: coefficients used for computing running averages of gradient and its square
    # eps: term added to the denominator to improve numberical stability
    # weight_decay: L2 penalty
    # amsgrad: whether to use the AMSGrad variant of this algorithm
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(SharedAdam, self).__init__(
            params, lr=lr, betas=betas, eps=eps, 
            weight_decay=weight_decay, amsgrad=amsgrad)
        # group is a param_group, {'params':[x,...,x], 'lr':x, 'betas':(x, x), 'weight_decay':x ...}
        # The param_group specifies what tensors should be optimized along with group
        # specific optimization options.
        for group in self.param_groups:
            for p in group['params']:
                # self.state = defaultdict(dict) [Optimizer Class]
                # Syntax: defaultdict(default_factory)
                # default_factory: A function returning the default value for the dictionary defined.
                state = self.state[p]
                
                # We need to have 'step' and 'shared_step',
                # because share_memory_() is a tensor's method;
                # torch.optim.Adam needs step to be an int.
                state['step'] = 0
                # share_memory_() : Moves the underlying storage to shared memory. 
                state['shared_step'] = torch.zeros(1).share_memory_()
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()
                if weight_decay:
                    state['weight_decay'] = torch.zeros_like(p.data).share_memory_()
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # BUG: 'steps' ---> 'step' 
                self.state[p]['steps'] = self.state[p]['shared_step'].item()
                self.state[p]['shared_step'] += 1
        super().step(closure)
        
        
class SharedRMSprop(torch.optim.RMSprop):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super(SharedRMSprop, self).__init__(
            params, lr=lr, alpha=alpha, 
            eps=eps, weight_decay=weight_decay, 
            momentum=momentum, centered=centered)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['shared_step'] = torch.zeros(1).share_memory_()
                state['square_avg'] = torch.zeros_like(p.data).share_memory_()
                if weight_decay:
                    state['weight_decay'] = torch.zeros_like(p.data).share_memory_()
                if momentum > 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data).share_memory_()
                if centered:
                    state['grad_avg'] = torch.zeros_like(p.data).share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # BUG: 'steps' ---> 'step' 
                self.state[p]['steps'] = self.state[p]['shared_step'].item()
                self.state[p]['shared_step'] += 1
        super().step(closure)
        
class A3C():
    def __init__(self, 
                 policy_model_fn, 
                 policy_model_max_grad_norm, 
                 policy_optimizer_fn, 
                 policy_optimizer_lr,
                 value_model_fn, 
                 value_model_max_grad_norm, 
                 value_optimizer_fn, 
                 value_optimizer_lr, 
                 entropy_loss_weight, 
                 max_n_steps, 
                 n_workers):
        self.policy_model_fn = policy_model_fn
        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        
        self.value_model_fn = value_model_fn
        self.value_model_max_grad_norm = value_model_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        
        self.entropy_loss_weight = entropy_loss_weight
        self.max_n_steps = max_n_steps
        self.n_workers = n_workers

    def optimize_model(self, logpas, entropies, rewards, values, 
                       local_policy_model, local_value_model):
        # (s_0, a_0, r_0, s_1)
        # (s_1, a_1, r_1, s_2)
        # \vdots
        # (s_{T-2}, a_{T-2}, r_{T-2}, s_{T-1})
        # rewards = [r_0, r_1, ..., r_{T-2}, v(s_{T-2})]
        T = len(rewards)
        
        # discounts = [1, gamma, gamma**2, gamma**3, ..., gamma**(T-1)]
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        # returns = [return of action 0, ..., return of action T-2, gamma**(T-1) * v(s_{T-2})]
        returns = np.array([np.sum(discounts[:T-t] * rewards[t:]) for t in range(T)])
        # discounts: [[1], [gamma], [gamma**2], [gamma**3], ..., [gamma**(T-2)]]
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1)
        # returns = [[return of action 0], ..., [return of action T-2]]
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1)
        
        # logpas = [ [\pi(a_0)],[\pi(a_1)], ...,[\pi(a_{T-2})] ]
        logpas = torch.cat(logpas)
        entropies = torch.cat(entropies)
        values = torch.cat(values)

        value_error = returns - values
        policy_loss = -(discounts * value_error.detach() * logpas).mean()
        entropy_loss = -entropies.mean()
        loss = policy_loss + self.entropy_loss_weight * entropy_loss
         # BUG: zero_grad() ----> zero_grad(set_to_none=True)
        self.shared_policy_optimizer.zero_grad()
        # self.shared_policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_policy_model.parameters(), 
                                       self.policy_model_max_grad_norm)
        for param, shared_param in zip(local_policy_model.parameters(), 
                                       self.shared_policy_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_policy_optimizer.step()
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())

        value_loss = value_error.pow(2).mul(0.5).mean()
        # BUG: zero_grad() ----> zero_grad(set_to_none=True)
        self.shared_value_optimizer.zero_grad()
        # self.shared_value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_value_model.parameters(), 
                                       self.value_model_max_grad_norm)
        for param, shared_param in zip(local_value_model.parameters(), 
                                       self.shared_value_model.parameters()):
                        
            # .grad is a function wrapped by "@property"
            if shared_param.grad is None:
                # After _grad is set, .grad is not None anymore
                shared_param._grad = param.grad
        self.shared_value_optimizer.step()
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

    # Static methods have a very clear use-case. When we need some functionality not w.r.t an Object
    # but w.r.t the complete class, we make a method static. Note that in a static method, we don't need 
    # the self to be passed as the first argument. 
    @staticmethod
    def interaction_step(state, env, local_policy_model, local_value_model,
                         logpas, entropies, rewards, values):
        action, is_exploratory, logpa, entropy = local_policy_model.full_pass(state)
        new_state, reward, is_terminal, info = env.step(action)
        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']

        logpas.append(logpa)
        entropies.append(entropy)
        rewards.append(reward)
        values.append(local_value_model(state))

        return new_state, reward, is_terminal, is_truncated, is_exploratory

    def work(self, rank):        
        last_debug_time = float('-inf')
        self.stats['n_active_workers'].add_(1)
        
        local_seed = self.seed + rank
        env = self.make_env_fn(**self.make_env_kargs, seed=local_seed)
        torch.manual_seed(local_seed) ; np.random.seed(local_seed) ; random.seed(local_seed)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        local_policy_model = self.policy_model_fn(nS, nA)
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())
        local_value_model = self.value_model_fn(nS)
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

        global_episode_idx = self.stats['episode'].add_(1).item() - 1
        
        while not self.get_out_signal:            
            episode_start = time.time()
            state, is_terminal = env.reset(), False
            
            # collect n_steps rollout
            n_steps_start, total_episode_rewards = 0, 0
            total_episode_steps, total_episode_exploration = 0, 0
            logpas, entropies, rewards, values = [], [], [], []

            for step in count(start=1):
                state, reward, is_terminal, is_truncated, is_exploratory = self.interaction_step(
                    state, env, local_policy_model, local_value_model, 
                    logpas, entropies, rewards, values)

                total_episode_steps += 1
                total_episode_rewards += reward
                total_episode_exploration += int(is_exploratory)
                
                if is_terminal or step - n_steps_start == self.max_n_steps:
                    is_failure = is_terminal and not is_truncated
                    next_value = 0 if is_failure else local_value_model(state).detach().item()
                    rewards.append(next_value)

                    self.optimize_model(logpas, entropies, rewards, values, 
                                        local_policy_model, local_value_model)
                    logpas, entropies, rewards, values = [], [], [], []
                    n_steps_start = step
                
                if is_terminal:
                    # garbage collection
                    gc.collect()
                    break

            # save global stats
            episode_elapsed = time.time() - episode_start
            evaluation_score, _ = self.evaluate(local_policy_model, env)
            self.save_checkpoint(global_episode_idx, local_policy_model)
            
            # These are shared data structure. 
            self.stats['episode_elapsed'][global_episode_idx].add_(episode_elapsed)
            self.stats['episode_timestep'][global_episode_idx].add_(total_episode_steps)
            self.stats['episode_reward'][global_episode_idx].add_(total_episode_rewards)
            self.stats['episode_exploration'][global_episode_idx].add_(total_episode_exploration/total_episode_steps)
            self.stats['evaluation_scores'][global_episode_idx].add_(evaluation_score)

            mean_10_reward = self.stats[
                'episode_reward'][:global_episode_idx+1][-10:].mean().item()
            mean_100_reward = self.stats[
                'episode_reward'][:global_episode_idx+1][-100:].mean().item()
            mean_100_eval_score = self.stats[
                'evaluation_scores'][:global_episode_idx+1][-100:].mean().item()
            mean_100_exp_rat = self.stats[
                'episode_exploration'][:global_episode_idx+1][-100:].mean().item()
            std_10_reward = self.stats[
                'episode_reward'][:global_episode_idx+1][-10:].std().item()
            std_100_reward = self.stats[
                'episode_reward'][:global_episode_idx+1][-100:].std().item()
            std_100_eval_score = self.stats[
                'evaluation_scores'][:global_episode_idx+1][-100:].std().item()
            std_100_exp_rat = self.stats[
                'episode_exploration'][:global_episode_idx+1][-100:].std().item()
            if std_10_reward != std_10_reward: std_10_reward = 0            
            if std_100_reward != std_100_reward: std_100_reward = 0
            if std_100_eval_score != std_100_eval_score: std_100_eval_score = 0
            if std_100_exp_rat != std_100_exp_rat: std_100_exp_rat = 0
            global_n_steps = self.stats[
                'episode_timestep'][:global_episode_idx+1].sum().item()
            global_training_elapsed = self.stats[
                'episode_elapsed'][:global_episode_idx+1].sum().item()
            wallclock_elapsed = time.time() - self.training_start
            
            # global_n_steps: 截止到 global_episode_idx, 所有的进程和环境交互的次数
            self.stats['result'][global_episode_idx][0].add_(global_n_steps)
            # mean_100_reward:  非贪心策略（带有探索性），最近 100 轮的平均收益（所有进程上） 
            self.stats['result'][global_episode_idx][1].add_(mean_100_reward)
            # mean_100_eval_score: 贪心策略（纯利用），最近 100 轮的平均收益 （所有进程上）
            self.stats['result'][global_episode_idx][2].add_(mean_100_eval_score)
            # 所有进程训练时间的总和
            self.stats['result'][global_episode_idx][3].add_(global_training_elapsed)
            # 所有进程耗费时间的总和（训练时间 + evaluate 的时间）
            self.stats['result'][global_episode_idx][4].add_(wallclock_elapsed)
            
            # time.gmtime: 将一个时间戳转换为 UTC 时区的 struct_time，其参数 sec 表示从 1970-1-1 以来的
            # 秒数。
            # time.strftime: 用于格式化时间
            # \u00B1 plus-minus sign
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, global_episode_idx, global_n_steps, mean_10_reward, std_10_reward, 
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            
            if rank == 0:
                # '\r' clears the line just printed
                print(debug_message, end='\r', flush=True)
                if time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS:
                    print(ERASE_LINE + debug_message, flush=True)
                    last_debug_time = time.time()
            
            # self.get_out_lock 是一个 token，取得 token 的进程进入此段代码。
            # 在此段代码中判断是否达到了终止程序的条件。
            # 在满足条件时，设置 get_out_signal，以通知其他进程
            with self.get_out_lock:
                potential_next_global_episode_idx = self.stats['episode'].item()
                self.reached_goal_mean_reward.add_(
                    mean_100_eval_score >= self.goal_mean_100_reward)
                self.reached_max_minutes.add_(
                    time.time() - self.training_start >= self.max_minutes * 60)
                self.reached_max_episodes.add_(
                    potential_next_global_episode_idx >= self.max_episodes)
                if self.reached_max_episodes or \
                   self.reached_max_minutes or \
                   self.reached_goal_mean_reward:
                    self.get_out_signal.add_(1)
                    break
                # else go work on another episode
                global_episode_idx = self.stats['episode'].add_(1).item() - 1

        # rank-0 finishes his job and waits for other processes.
        while rank == 0 and self.stats['n_active_workers'].item() > 1:
            pass
        
        # rank-0 print info
        if rank == 0:
            print(ERASE_LINE + debug_message)
            if self.reached_max_minutes: print(u'--> reached_max_minutes \u2715')
            if self.reached_max_episodes: print(u'--> reached_max_episodes \u2715')
            if self.reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')

        env.close() ; del env
        self.stats['n_active_workers'].sub_(1)

    def train(self, make_env_fn, make_env_kargs, seed, gamma, 
              max_minutes, max_episodes, goal_mean_100_reward):

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        self.max_minutes = max_minutes
        self.max_episodes = max_episodes
        self.goal_mean_100_reward = goal_mean_100_reward

        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        nS, nA = env.observation_space.shape[0], env.action_space.n
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)

        self.stats = {}
        self.stats['episode'] = torch.zeros(1, dtype=torch.int).share_memory_()
        self.stats['result'] = torch.zeros([max_episodes, 5]).share_memory_()
        self.stats['evaluation_scores'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_reward'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_timestep'] = torch.zeros([max_episodes], dtype=torch.int).share_memory_()
        self.stats['episode_exploration'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_elapsed'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['n_active_workers'] = torch.zeros(1, dtype=torch.int).share_memory_()
        
        # The shared memory is a memory pool, which can be used by multiple processes
        # to exchange information and data.
        self.shared_policy_model = self.policy_model_fn(nS, nA).share_memory()
        self.shared_policy_optimizer = self.policy_optimizer_fn(self.shared_policy_model, 
                                                                self.policy_optimizer_lr)
        self.shared_value_model = self.value_model_fn(nS).share_memory()
        self.shared_value_optimizer = self.value_optimizer_fn(self.shared_value_model, 
                                                              self.value_optimizer_lr)

        self.get_out_lock = mp.Lock()
        self.get_out_signal = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_max_minutes = torch.zeros(1, dtype=torch.int).share_memory_() 
        self.reached_max_episodes = torch.zeros(1, dtype=torch.int).share_memory_() 
        self.reached_goal_mean_reward  = torch.zeros(1, dtype=torch.int).share_memory_() 
        self.training_start = time.time()
        workers = [mp.Process(target=self.work, args=(rank,)) for rank in range(self.n_workers)]
        [w.start() for w in workers] ; [w.join() for w in workers]
    
        # 多进程并发运行所需要的时间
        wallclock_time = time.time() - self.training_start

        final_eval_score, score_std = self.evaluate(self.shared_policy_model, env, n_episodes=100)
        env.close() ; del env

        final_episode = self.stats['episode'].item()
        # 所有进程训练时间的总和
        training_time = self.stats['episode_elapsed'][:final_episode+1].sum().item()

        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))

        self.stats['result'] = self.stats['result'].numpy()
        self.stats['result'][final_episode:, ...] = np.nan
        self.get_cleaned_checkpoints()
        return self.stats['result'], final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1, greedy=True):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                if greedy:
                    a = eval_policy_model.select_greedy_action(s)
                else: 
                    a = eval_policy_model.select_action(s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=5):
        try: 
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}
        # 获得 self.checkpoint_dir 下所有 tar 文件
        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        paths_dic = {int(path.split('.')[-2]):path for path in paths}
        last_ep = max(paths_dic.keys())
        # checkpoint_idxs = np.geomspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=np.int)-1
        checkpoint_idxs = np.linspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=np.int)-1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                # deletes the file path. 
                os.unlink(path)

        return self.checkpoint_paths

    def demo_last(self, title='Fully-trained {} Agent', n_episodes=3, max_n_videos=3):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.shared_policy_model.load_state_dict(torch.load(checkpoint_paths[last_ep]))

        self.evaluate(self.shared_policy_model, env, n_episodes=n_episodes)
        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def demo_progression(self, title='{} Agent progression', max_n_videos=5):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.shared_policy_model.load_state_dict(torch.load(checkpoint_paths[i]))
            self.evaluate(self.shared_policy_model, env, n_episodes=1)

        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            subtitle_eps=sorted(checkpoint_paths.keys()),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(), 
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))

# 将采用 lambda 定义的函数修改为正常定义的函数，以避免 
# can't pickle <function <lambda> at......> 的错误
# policy_model_fn = lambda nS, nA: FCDAP(nS, nA, hidden_dims=(128,64))
def policy_model_fn(nS, nA):
    return FCDAP(nS, nA, hidden_dims=(128,64))

# policy_optimizer_fn = lambda net, lr: SharedAdam(net.parameters(), lr=lr)
def policy_optimizer_fn(net, lr):
    return SharedAdam(net.parameters(), lr=lr)

# value_model_fn = lambda nS: FCV(nS, hidden_dims=(256,128))    
def value_model_fn(nS):
    return FCV(nS, hidden_dims=(256,128))

# value_optimizer_fn = lambda net, lr: SharedRMSprop(net.parameters(), lr=lr)
def value_optimizer_fn(net, lr):
    return SharedRMSprop(net.parameters(), lr=lr)

if __name__ == '__main__':
    a3c_results = []
    best_agent, best_eval_score = None, float('-inf')
    for seed in SEEDS:
        environment_settings = {
            'env_name': 'CartPole-v1',
            'gamma': 1.00,
            'max_minutes': 10, 
            'max_episodes': 10000,
            'goal_mean_100_reward': 475
        }

        
    
        policy_model_max_grad_norm = 1
        policy_optimizer_lr = 0.0005
        value_model_max_grad_norm = float('inf')
        value_optimizer_lr = 0.0007

        entropy_loss_weight = 0.001

        max_n_steps = 50
        n_workers = 8

        env_name, gamma, max_minutes, \
        max_episodes, goal_mean_100_reward = environment_settings.values()
        agent = A3C(policy_model_fn,
                    policy_model_max_grad_norm, 
                    policy_optimizer_fn, 
                    policy_optimizer_lr,
                    value_model_fn,
                    value_model_max_grad_norm,
                    value_optimizer_fn, 
                    value_optimizer_lr,
                    entropy_loss_weight, 
                    max_n_steps,
                    n_workers)

        make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name)
        result, final_eval_score, training_time, wallclock_time = agent.train(
            make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
        a3c_results.append(result)
        if final_eval_score > best_eval_score:
            best_eval_score = final_eval_score
            best_agent = agent
    a3c_results = np.array(a3c_results)
    _ = BEEP()