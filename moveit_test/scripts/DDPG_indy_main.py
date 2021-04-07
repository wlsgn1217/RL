#!/usr/bin/env python

# Run this file in ~/catkin_ws/src/moveit_test/scripts

import numpy as np
from indy_env_comm import communicate_indy
from DDPG_Agent import Agent
import matplotlib.pyplot as plt

def plot_learning_curve(x,scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = communicate_indy()
    indy = Agent(alpha=0.0001, beta=0.001, input_dims=(12, ) ,tau=0.001, batch_size=64, fc1_dims=400, fc2_dims=300, n_actions=6)

    n_games = 3000
    score_history = []
    filename = 'Indy_DDPG_' + str(indy.alpha) + '_beta_' + str(indy.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'
    best_score = 0

    for i in range(n_games):
        score = 0.0
        done = False
        obs = env.reset()
        indy.noise.reset()
        while not done:
            action = indy.choose_action(obs)
            obs_, reward, done = env.step(action/4)
            score += reward

            indy.remember(obs, action, reward, obs_, done)
            indy.learn()
            obs = obs_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            indy.save_models()

        print('episode ', i, 'score %.1f' % score, 'average score %.1f' % avg_score)

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x,score_history, figure_file)