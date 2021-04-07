#!/usr/bin/env python
import numpy as np
from Updated_SAC_Agent import Agent
import wandb
from indy_env_comm import communicate_indy
from ReplayBuffer import ReplayBuffer, HER_for_indy

if __name__ == '__main__':
    env = communicate_indy()
    agent = Agent(lr=3e-4, tau=0.005, gamma=0.99, env_id='indy',input_dims=(9,), env=env, max_size=1000000, batch_size=256, layer1_size=256, layer2_size=256, n_actions=6)
    wandb.init(project='Indy_Move_SAC')

    n_games = 10000
    best_score = -3000.0
    score_history = []
    steps = 0
    load_checkpoint = True #False
    achieved_goal = 0
    steps_per_episode = 0
    
    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        score = 0.0
        done = False
        observation = env.reset()

        episode_memory = HER_for_indy(30, (9,), 6)

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done= env.step(action/4)
            steps += 1
            steps_per_episode += 1
            score += reward
            #print("observation ", observation)
            #print("action ", action)
            #print("reward ", reward)
            #print("observation_ ", observation_)
            #print("done", done)
            agent.remember(observation, action, reward, observation_, done)

            episode_memory.instant_store(observation, action, reward, observation_, done)
            
            observation = observation_
        
        new_goal = observation[0][0:3]
        updated_done = False
        HER_steps = 0
        while HER_steps < steps_per_episode and not updated_done:
            updated_state, action, updated_reward, updated_state_ , updated_done = episode_memory.update_value(new_goal,HER_steps)
            agent.remember(updated_state, action, updated_reward, updated_state_ , updated_done)
            HER_steps +=1
            #print("HER value: ", updated_state, updated_reward, updated_state_, updated_done, HER_steps)

            if not load_checkpoint:
                agent.learn()



        wandb.log({"Steps": steps})
        wandb.log({"Return": score})
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        wandb.log({"Avg Return": avg_score})
            

        if steps_per_episode < 30:
            achieved_goal +=1
        wandb.log({"Achieved Goal": achieved_goal})

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode ', i, 'score %.1f' % score, 'trailing 100 games average %.1f' % avg_score, 'steps %d' % steps, 'indy_move')

