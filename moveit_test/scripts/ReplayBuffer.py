import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr %self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class HER_for_indy():
    def __init__(self, episode_max_step, input_shape, n_actions):
        self.episode_memory = ReplayBuffer(episode_max_step, input_shape, n_actions)

    def instant_store(self, observation, action, reward, observation_, done):
        self.episode_memory.store_transition(observation, action, reward, observation_, done)

    def update_value(self, new_goal,index):

        state = self.episode_memory.state_memory[index]
        action = self.episode_memory.action_memory[index]
        state_ = self.episode_memory.new_state_memory[index]

        updated_state = self.update_observation(new_goal, state)
        updated_state_ = self.update_observation(new_goal, state_)
        updated_reward, updated_done = self.update_reward_and_done(new_goal, state, state_)


        return updated_state, action, updated_reward, updated_state_ , updated_done


    def update_reward_and_done(self, new_goal, prev_state, state):
        current_location = state[0:3]
        distance = np.linalg.norm(new_goal-current_location)
        done = False

        if distance < 0.05:
            reward = 1000.0
            done = True
        else:
            reward = -round(distance**2,3)
            if distance<0.5:
                reward = reward + 5/distance

        previous_location = prev_state[0:3]
        distance_from_prev = np.linalg.norm(current_location-previous_location)

        if distance_from_prev < 0.02:
            penalty = -20
        else:
            penalty = 0

        return round(reward + penalty,3), done

    def update_observation(self, new_goal, observation):
        updated_observation = observation
        updated_observation[6:9] = new_goal

        return updated_observation
