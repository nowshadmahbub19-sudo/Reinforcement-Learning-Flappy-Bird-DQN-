import torch
import gymnasium as gym
import flappy_bird_gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch.nn as nn
import torch.optim as optim




if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class Agent:
    def __init__(self, param_set):                      # FIX: typo "peram_set" -> "param_set"

        self.param_set = param_set

        with open("parameters.yaml", "r") as f:          # FIX: "perameters.yamal" -> valid filename
            all_params_set = yaml.safe_load(f)
            params = all_params_set[param_set]           # FIX: dict lookup [] not call (), store in `params`

        self.alpha = params['alpha']                     # FIX: keys must be strings, not bare names
        self.epsilon_init = params['epsilon_init']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.replay_memory_size = params['replay_memory_size']
        self.mini_batch_size = params['mini_batch_size']
        self.network_sync_rate = params['network_sync_rate']
        self.gamma = params['gamma']
        self.reward_thresold = params['reward_thresold']

        self.loss_fn = nn.MSELoss()
        self.optimizer = None


    def run(self, is_training=True, render=False):        # FIX: missing `self` -> self.* below would crash

        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        state, _ = env.reset()

        num_states = env.observation_space.shape[0]  # input dimension
        num_actions = env.action_space.n             # output dim
        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon=self.epsilon_init
        for episode in itertools.count():
            state, _ = env.reset()
            state=torch.tensor(state,torch.dtype=float,device=device)

            episode_reward = 0
            done = False                                  # FIX: `terminated=0` shadowed the step() flag

            while not done:        # FIX: loop on done so truncation also ends the episode
                
                if is_training and random.random() <epsilon:
                    
                    action = env.action_space.sample()   #explore
                    action=torch.tensor(action,torch.dtype=float,device=device)

                else:
                    with torch.nograd():
                        action=policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()   #eploit
                    
                    
                next_state, reward, terminated, truncated, _,_ = env.step(action.item())
                
                #create tensor
                reward=torch.tensor(reward,torch.dtype=float,device=device)
                next_state=torch.tensor(next_state,torch.dtype=float,device=device)


                if is_training:
                    memory.append((state, action, reward, next_state, done))

                state = next_state
                episode_reward += reward

            print(f"for episode={episode+1} with total reward={episode_reward} & {epsilon}")
      
      
            #epsilon decay
            epsilon=max(epsilon * self.epsilon_decay,self.epsilon_min)
       # env.close() -we want to manuallly store
        #return memory if is_training else None

