from dqn import DQN
from env import Acrobot
import tqdm, sys, os, json, pickle
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime

def training(total_episodes, save_every = 50):
   try :
        from config import DQN_CONFIG
   except ImportError :
        print("config file is missing or DQN_CONFIG is missing from config file")
        sys.exit()
   except KeyError :
        print("DQN_CONFIG is incorrect")
        sys.exit()
   
   # using GMT time to create a unique folder for saving
   save_folder_name = strftime("save_%d-%b-%Y-%H-%M-%S", gmtime())
   os.mkdir(save_folder_name) # creating directory for saving checkpoints and recording
   os.mkdir(os.path.join(save_folder_name,'DQN_checkpoints')) # for storing the Q network weights
   # writing the Q-learning config used
   with open(os.path.join(save_folder_name,'dqn_config_used.json'),'w') as config:
      json.dump(DQN_CONFIG,config)

   episodes_rewards = [] # total reward for each episode
   # stats for every 5 episode
   aggr_episodes_rewards = {
      'ep' : [],
      'avg' : [],
      'min' : [],
      'max' : []
   } 
   # creating a DQN agent
   DQN_CONFIG['n_obs'] = 6 # size of obervation space
   DQN_CONFIG['n_actions'] = 3 # size of input actions
   agent = DQN(**DQN_CONFIG)

   # Creating a Environment instance
   env = Acrobot()

   progress_bar = tqdm.trange(total_episodes+1)
   for episode in progress_bar:

      save_eps = True if episode%(save_every) == 0 else False

      obs = env.reset(seed=42)
      eps_reward = 0
      while not env.done:

         action = agent.choose_action(obs)
         # step the environment
         next_obs, reward, done = env.step(action)
         eps_reward += reward

         # store the experience and train the agent
         agent.store_transition(obs,action,reward,next_obs,done)
         agent.learn()
         obs = next_obs
         
         if save_eps :
            # render the environment
            env.render(FPS_lock=15, debug=True)
            env.record(os.path.join(save_folder_name,f"E{episode}"),FPS=15)

         env.close_quit()
      progress_bar.set_description("Episodes")
      progress_bar.set_postfix(reward = eps_reward)
   
      # saving the DQN networks
      if save_eps :
         agent.save(os.path.join(save_folder_name,"DQN_checkpoints",f"E{episode}"))

      # updating the stats considering last 5 episodes
      if episode and episode%10 == 0 :
         aggr_episodes_rewards['ep'].append(episode)
         avg_reward = sum(episodes_rewards[-10:])/len(episodes_rewards[-10:])
         aggr_episodes_rewards['avg'].append(avg_reward)
         aggr_episodes_rewards['min'].append(min(episodes_rewards[-10:]))
         aggr_episodes_rewards['max'].append(max(episodes_rewards[-10:]))

      episodes_rewards.append(eps_reward)
      # decay epsilon for each episode
      agent.anneal_epsilon()

   fig_handle = plt.figure(figsize=(12,7))
   plt.title(f"Average episode rewards")
   plt.ylabel('Net episode reward')
   plt.plot(aggr_episodes_rewards['ep'],aggr_episodes_rewards['avg'],label='average reward')
   plt.plot(aggr_episodes_rewards['ep'],aggr_episodes_rewards['min'],label='min reward')
   plt.plot(aggr_episodes_rewards['ep'],aggr_episodes_rewards['max'],label='max reward')
   plt.xlabel("Episodes")
   plt.plot(np.arange(len(episodes_rewards)),np.array(episodes_rewards),alpha=0.2,color='#0066ff')
   plt.grid()
   plt.legend()

   plt.savefig(os.path.join(save_folder_name,'metrics.png'))
   # saving matplotlib figure for later use
   with open(os.path.join(save_folder_name,'metrics.pickle'), 'wb') as f:
      pickle.dump(fig_handle,f)
      print(f"matplotlib figure saved a {save_folder_name}/metrics.pickle file")

   plt.show()
   print("<< Done >>")


def inference(checkpoint_filepath:str):
   '''
   Params :
   checkpoint_filepath : relative path
   '''
   
   try :
        from config import DQN_CONFIG
   except ImportError :
        print("config file is missing or DQN_CONFIG is missing from config file")
        sys.exit()
   except KeyError :
        print("DQN_CONFIG is incorrect")
        sys.exit()
   # creating a DQN agent
   DQN_CONFIG['n_obs'] = 6 # size of obervation space
   DQN_CONFIG['n_actions'] = 3 # size of input actions
   agent = DQN(**DQN_CONFIG)

   # loading from saved model
   agent.load(checkpoint_filepath)

   # Creating a Environment instance
   env = Acrobot()

   obs = env.reset(seed=42)
   eps_reward = 0
   while not env.done:

      # choose action
      action = agent.choose_action(obs)

      # step the environment
      next_obs, reward, done = env.step(action)
      eps_reward += reward
      obs = next_obs
      
      env.render(FPS_lock=15, debug=True)
      env.close_quit()

   print(f"Net reward for the episode = {eps_reward}")

def main():
   training(total_episodes=6000, save_every=200)
   # inference('save_27-Nov-2022-12-51-18\DQN_checkpoints\E90_local.pth')

if __name__ == '__main__':
   main()
