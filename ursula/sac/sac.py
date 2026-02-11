from typing import Optional, Type, Union
from gymnasium import Env
import torch 
import torch.nn as nn 
import torch.optim as optim # Optimization algorithms
import numpy as np 
from ursula.buffers import Buffer, ReplayBuffer
from ursula.networks import ACTOR_REGISTRY, CRITIC_REGISTRY
from ursula.networks import Critic, GaussianActor as Actor
from ursula.networks import BaseValueFunction, BasePolicy

#TODO: actor is Actor or BasePolicy? Define the difference between this classes.

class SAC:
    actor: BasePolicy
    critic1: BaseValueFunction
    critic2: BaseValueFunction
    critic_target1: BaseValueFunction
    critic_target2: BaseValueFunction
    replay_buffer: Buffer
    
    def __init__(self, 
                 env: Env,
                 buffer_size: int=int(1e6), 
                 min_buffer_size: int=1000,
                 batch_size: int=256, 
                 hidden_sizes: list = [256, 256], 
                 max_action: float=1.0,
                 actor: str | type[BasePolicy] = Actor, 
                 critic: str | type[BaseValueFunction] = Critic, 
                 buffer: type[Buffer] = ReplayBuffer, 
                 actor_lr: float=3e-4, 
                 critic_lr: float=3e-4,
                 max_grad_norm_actor: float | None = 1.0,
                 max_grad_norm_critic: float | None = 1.0,
                 max_grad_norm_alpha: float | None  = None,
                 log_grad_norm: bool = False,
                 gamma: float=0.99, 
                 tau: float=0.005, 
                 alpha: float=0.2, 
                 auto_entropy: bool=False,
                 use_amp: bool=False):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == "cuda"

        #! AMP methods are deprecated i guess. I will figure out later.
        #? self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        print(torch.__version__)
        self.env = env
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        state_dim = int(np.prod(observation_space.shape)) if observation_space.shape is not None else 1 # Flatten state dimension if it's multi-dimensional
        action_dim = int(np.prod(action_space.shape)) if action_space.shape is not None else 1 # Flatten action dimension if it's multi-dimensional
        
        # Initialize replay buffer
        if issubclass(buffer, Buffer):
            self.replay_buffer = buffer(max_size=buffer_size, 
                                        state_dim=state_dim, 
                                        action_dim=action_dim)
        else:
            raise ValueError("Provided buffer class must be a subclass of Buffer. Make sure your custom buffer implements the Buffer interface.")
        
        self.min_buffer_size = min_buffer_size # Minimum buffer size before training
        self.batch_size = batch_size
        
        self._create_actor(actor,observation_space, action_space, max_action, hidden_sizes)
        self._create_critics(critic, observation_space, action_space, hidden_sizes)
        self._init_optimizers(actor_lr,critic_lr)

        self.max_grad_norm_actor = max_grad_norm_actor
        self.max_grad_norm_critic = max_grad_norm_critic
        self.max_grad_norm_alpha = max_grad_norm_alpha
        self.log_grad_norm = log_grad_norm

        self.gamma = gamma # Discount factor
        self.tau = tau # Soft update factor

        self.auto_entropy = auto_entropy # Automatic entropy tuning
        self._setup_entropy(alpha=alpha, action_dim=action_dim, learning_rate=actor_lr)

    # a function to refactor network init    
    def _create_actor(self, 
                      actor: str | type[BasePolicy],
                      observation_space, 
                      action_space, 
                      max_action, 
                      hidden_sizes,
                      log_std_min = -20,
                      log_std_max = 2):
                
        # Actor Init
        if isinstance(actor, str):
            if actor not in ACTOR_REGISTRY:
                raise KeyError(f"Actor '{actor}' not found in registry. Available: {list(ACTOR_REGISTRY.keys())}")
            actor_class = ACTOR_REGISTRY[actor]
        elif isinstance(actor, type):
            # check if it's the right subclass for extra safety
            if not issubclass(actor, torch.nn.Module):
                raise TypeError(f"Custom actor {actor} must be a subclass of torch.nn.Module")
            actor_class = actor
        else:
            raise TypeError(f"Actor must be a string or a class type, received {type(actor)}")
        
        self.actor = actor_class(observation_space = observation_space, 
                                 action_space = action_space, 
                                 hidden_sizes = hidden_sizes, 
                                 max_action = max_action, 
                                 log_std_min = -20, 
                                 log_std_max = 2).to(self.device)
    
    def _create_critics(self,
                        critic,
                        observation_space,
                        action_space,
                        hidden_sizes):
    
        if isinstance(critic, str):
            if critic not in CRITIC_REGISTRY:
                raise KeyError(f"Critic '{critic}' not found in registry. Available: {list(CRITIC_REGISTRY.keys())}")
            critic_class = CRITIC_REGISTRY[critic]
        elif isinstance(critic, type):
            critic_class = critic
        else:
            raise TypeError(f"Critic must be a string or a class type, received {type(critic)}")

        self.critic1 = critic_class(observation_space = observation_space, 
                                        action_space = action_space, 
                                        hidden_sizes = hidden_sizes).to(self.device)
        self.critic2 = critic_class(observation_space = observation_space, 
                                        action_space = action_space, 
                                        hidden_sizes = hidden_sizes).to(self.device)
        self.critic_target1 = critic_class(observation_space = observation_space, 
                                              action_space = action_space, 
                                              hidden_sizes = hidden_sizes).to(self.device)
        self.critic_target2 = critic_class(observation_space = observation_space, 
                                              action_space = action_space, 
                                              hidden_sizes = hidden_sizes).to(self.device)

        # Copy weights from Critic to target Critic networks
        self.critic_target1.load_state_dict(self.critic1.state_dict()) # Copy weights to target
        self.critic_target2.load_state_dict(self.critic2.state_dict()) # Copy weights to target

    def _setup_entropy(self, alpha:float, action_dim, learning_rate: float):
        if self.auto_entropy:
            # 1. Set Target Entropy: The common heuristic is -dim(A).
            self.target_entropy = -float(action_dim) # Target entropy
            # 2. We optimize log_alpha instead of alpha directly.
            # Why? Because alpha must always be positive. exp(log_alpha) is always positive.
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device) # Log alpha parameter
            # 3. Create an optimizer specifically for alpha
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate) # Alpha optimizer

            self.alpha = self.log_alpha.exp() # Initial alpha value
        else:
            self.alpha = torch.tensor(alpha, device=self.device) # Entropy coefficient if not auto tuning

    def _init_optimizers(self, actor_lr: float, critic_lr: float):
        #? Should i use separate optimizers for each critic? I think yes but not sure.
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr) # Actor optimizer
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr) # Critic optimizer
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr) # Critic optimizer
        self.mse_loss = nn.MSELoss()

    # Gradien Clipping
    def _optimize(self,
                  loss: torch.Tensor,
                  optimizer: optim.Optimizer,
                  parameters,
                  max_grad_norm: Optional[float] = None,
                  log_grad_norm: bool = True) -> None:
        
        params = list(parameters)

        optimizer.zero_grad()

        #! AMP methods are deprecated i guess. I will figure out later.
        # if self.scaler is not None:
        #     self.scaler.scale(loss).backward()
        #     self.scaler.unscale_(optimizer)
        # else:
        #     loss.backward()

        loss.backward()

        #TODO: Log properly
        # Log grad norms before clipping
        pre_clip_grad_norm = None
        if log_grad_norm:
            pre_clip_grad_norm = self._grad_norm(params)

        # if len(params) == 0:
        #     raise RuntimeError("No parameters passed to optimizer.")

        # Clip grad norms
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, max_grad_norm)

        # Log grad norms after clipping
        post_clip_grad_norm = None
        if log_grad_norm:
            post_clip_grad_norm = self._grad_norm(params)

        #! AMP methods are deprecated i guess. I will figure out later.
        # if self.scaler is not None:
        #     self.scaler.step(optimizer)
        #     self.scaler.update()
        # else:
        #     optimizer.step()

        optimizer.step()

        """
        For combinad parameters (Multihead or Shared Networks)
        this method can be called like that:

        self._optimize(self,
                        loss=joint_loss,
                        optimizer=self.shared_optimizer,
                        parameters=itertools.chain(
                            self.actor.parameters(),
                            self.critic1.parameters(),
                        ),
                        max_grad_norm=1.0,
                    )
        """
    
    def _grad_norm(self, parameters) -> float:
        """
        Compute L2 norm of gradients for a set of parameters.
        """
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    # Soft update target networks
    def _soft_update(self, net: torch.nn.Module, target_net: torch.nn.Module) -> None:
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # Select action based on current policy
    @torch.no_grad
    def choose_action(self, state: np.ndarray, evaluate: bool=False) -> np.ndarray:
        #? with torch.cuda.amp.autocast(enabled=self.use_amp): 
            state = np.array(state).reshape(-1)  # Ensure state is a 1D array
            state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device) # Convert state to tensor and move to device
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
            
            if evaluate:
                # Use deterministic mean action for evaluation
                sampled_action, _, mean_action = self.actor.sample(state_tensor)
                action = mean_action
            else:
                # Use stochastic sampled action for exploration
                action, _, _ = self.actor.sample(state_tensor)

            return action.squeeze(0).cpu().detach().numpy() # Remove batch dimension and convert to numpy array
        
    # Update the SAC agent
    def train(self) -> None:
        
        if self.replay_buffer.__len__() < self.min_buffer_size:
            return  # Not enough data to train
        
        # Sample a batch of transitions from the replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample_buffer(self.batch_size)

        # Convert to PyTorch tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        # reward = torch.FloatTensor(reward).view(-1, 1).to(self.device) ! No need to reshape
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        # done = torch.FloatTensor(done).view(-1, 1).to(self.device) ! No need to reshape
        done = torch.FloatTensor(done).to(self.device)

        # Update Critic networks
        with torch.no_grad(): #?, torch.cuda.amp.autocast(enabled=self.use_amp):
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1 = self.critic_target1(next_state, next_action)
            target_q2 = self.critic_target2(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * (torch.min(target_q1, target_q2) - self.alpha * next_log_prob)

        #? with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Current Q estimates
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        #? Compute Critic losses: Why not target_q.detach()?
        #*  it is already detached from the gradient graph. Adding .detach() is redundant but safe.
        critic1_loss = self.mse_loss(current_q1, target_q)
        critic2_loss = self.mse_loss(current_q2, target_q)

        # Optimize critic using _optimize
        self._optimize(loss = critic1_loss, 
                       optimizer = self.critic1_optimizer, 
                       parameters = self.critic1.parameters(), 
                       max_grad_norm=self.max_grad_norm_critic,
                       log_grad_norm=self.log_grad_norm)

        self._optimize(loss = critic2_loss, 
                       optimizer = self.critic2_optimizer, 
                       parameters = self.critic2.parameters(), 
                       max_grad_norm=self.max_grad_norm_critic,
                       log_grad_norm=self.log_grad_norm)

        # Update Actor network
        #? with torch.cuda.amp.autocast(enabled=self.use_amp):

        # Get new actions and log probabilities for the current states 
        new_action, log_prob, _ = self.actor.sample(state)
            
        # Q values for the new actions 
        q1_new_action = self.critic1(state, new_action)
        q2_new_action = self.critic2(state, new_action)
            
        # Minimum Q value for the new actions
        q_new_action = torch.min(q1_new_action, q2_new_action)

        # Compute Actor loss
        actor_loss = (self.alpha * log_prob - q_new_action).mean()

        self._optimize(loss = actor_loss, 
                       optimizer = self.actor_optimizer, 
                       parameters = self.actor.parameters(), 
                       max_grad_norm=self.max_grad_norm_actor,
                       log_grad_norm=self.log_grad_norm)


        # Update entropy coefficient alpha if using automatic entropy tuning
        if self.auto_entropy:
            # Compute alpha loss
            # lp (log_prob) is negative. If -lp > target_entropy, alpha decreases.
            # We use .detach() on lp because we are optimizing alpha, not the actor here.
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self._optimize(
                loss=alpha_loss,
                optimizer=self.alpha_optimizer,
                parameters=[self.log_alpha],
                max_grad_norm=self.max_grad_norm_alpha,
                log_grad_norm=self.log_grad_norm
            )

            # Update alpha value
            self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        self._soft_update(self.critic1, self.critic_target1)
        self._soft_update(self.critic2, self.critic_target2)
    
        # Clear memory cache just in case
        # gc.collect()
        # torch.cuda.empty_cache()