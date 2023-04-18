#Soft Actor-Critic Framework
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from sgdhess import SGDHess
from model import QNetwork
from policy import GaussianPolicy


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.optimizer_class = args.optimizer_class

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        dis_act = type(env.action_space) != gym.spaces.box.Box

        action_n = action_space.n if dis_act else action_space.shape[0]
        self.action_n = action_n

        self.critic = QNetwork(num_inputs, action_n, args.hidden_size).to(device=self.device)
        
        #Vary optimizier class based on args.optimizer_class arg
        if self.optimizer_class == "SGDHess":
          self.opt_c = SGDHess(self.critic.parameters(), momentum=0.9, clip=False, lr = args.lr)
        elif self.optimizer_class == "SGD":
          self.opt_c = SGD(self.critic.parameters(), momentum=0.9, lr=args.lr)
        elif self.optimizer_class == "Adam":
          self.opt_c = Adam(self.critic.parameters(), lr=args.lr)
        elif self.optimizer_class == "AdamW":
          self.opt_c = AdamW(self.critic.parameters(), lr=args.lr)
        
        self.critic_optim = self.opt_c

        self.critic_target = QNetwork(num_inputs, action_n, args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args["lr"])

        self.policy = GaussianPolicy(num_inputs, action_n, args.hidden_size, action_space, device=self.device).to(self.device)
        
        #Vary optimizier class based on args.optimizer_class arg
        if self.optimizer_class == "SGDHess":
          self.opt_p = SGDHess(self.policy.parameters(), momentum=0.9, clip=False, lr=args.lr])
        elif self.optimizer_class == "SGD":
          self.opt_p = SGD(self.policy.parameters(), momentum=0.9, lr=args.lr)
        elif self.optimizer_class == "Adam":
          self.opt_p = Adam(self.policy.parameters(), lr=args.lr)  
        elif self.optimizer_class == "AdamW":
          self.opt_p = AdamW(self.policy.parameters(), lr=args.lr)
        
        self.policy_optim = self.opt_p


    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if torch.isnan(state).any() == True or torch.isinf(state).any() == True:
          state = torch.zeros(state.size()[0], state.size()[1])
        
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # if action_batch.size != (batch_size, self.action_n):
        #   action_batch = action_batch.view(batch_size,1).expand(batch_size, self.action_n)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward(create_graph=True)
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward(create_graph=True)
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
