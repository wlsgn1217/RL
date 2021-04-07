import os
import torch as T
import torch.nn.functional as F
import numpy as np
from ReplayBuffer import ReplayBuffer
from Updated_SAC_Network import ActorNetwork, CriticNetwork
import torch.optim as optim

class Agent():
    def __init__(self, lr, input_dims, tau, env, env_id, gamma, n_actions, max_size, layer1_size, layer2_size, batch_size, use_automatic_entropy_tuning=True):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(lr, input_dims, layer1_size, layer2_size, n_actions=n_actions, name = env_id+'_actor')
        self.critic_1 = CriticNetwork(lr, input_dims, layer1_size, layer2_size, n_actions=n_actions, name=env_id+'_critic_1')
        self.critic_2 = CriticNetwork(lr, input_dims, layer1_size, layer2_size, n_actions=n_actions, name=env_id+'_critic_2')
        self.target_critic_1 = CriticNetwork(lr, input_dims, layer1_size, layer2_size, n_actions=n_actions, name=env_id+'_target_critic_1')
        self.target_critic_2 = CriticNetwork(lr, input_dims, layer1_size, layer2_size, n_actions=n_actions, name=env_id+'_target_critic_2')
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning

        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(6).item()
            self.log_temp = T.zeros(1, requires_grad=True, device=self.actor.device)
            self.temp_optimizer = optim.Adam([self.log_temp], lr=lr)

       
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        action = self.actor(state).rsample()

        return action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()

        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + (1-tau)*target_critic_1_state_dict[name].clone()
        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + (1-tau)*target_critic_2_state_dict[name].clone()
        

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)

    def save_models(self):
        print('...saving models ...')
        self.actor.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        

    def load_models(self):
        print('...loading models ...')
        self.actor.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        

    def compute_loss(self, batch):
        state, action, reward, next_state, done = batch

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        #Compute Policy and Alpha Loss

        distribution= self.actor(state)
        new_state_actions, log_pi = distribution.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)

        if self.use_automatic_entropy_tuning:
            temp_loss = -(self.log_temp * (log_pi + self.target_entropy).detach()).mean()
            temp = self.log_temp.exp()
        else:
            temp_loss = 0
            temp = 1

        q_new_actions = T.min(self.critic_1(state, new_state_actions), self.critic_2(state, new_state_actions))
        policy_loss = (temp*log_pi - q_new_actions).mean()

        #Compute Critic Loss
        q1_pred = self.critic_1(state, action)
        q2_pred = self.critic_2(state, action)
        next_dist = self.actor(next_state)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        target_q_value = T.min(self.target_critic_1(next_state, new_next_actions), self.target_critic_2(next_state, new_next_actions)) - temp*new_log_pi

        q_target = reward.unsqueeze(-1) +(1.-done.unsqueeze(-1).float())*self.gamma*target_q_value
        critic_1_loss = F.mse_loss(q1_pred, q_target.detach())
        critic_2_loss = F.mse_loss(q2_pred, q_target.detach())

        return temp_loss, policy_loss, critic_1_loss, critic_2_loss

    def learn(self):
        if self.memory.mem_cntr <self.batch_size:
            return
        
        temp_loss, policy_loss, critic_1_loss, critic_2_loss = self.compute_loss(self.memory.sample_buffer(self.batch_size))

        if self.use_automatic_entropy_tuning:
            self.temp_optimizer.zero_grad()
            temp_loss.backward()
            self.temp_optimizer.step()

        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1.optimizer.step()

        self.critic_2.optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2.optimizer.step()

        self.update_network_parameters()