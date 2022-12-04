import tensorflow as tf
from keras.models import Sequential, clone_model
from keras.layers import Dense, Flatten, Conv2D, Input, GaussianNoise
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
from collections import deque

class ReplayMemory():
    def __init__(self,max_len):
        self.max_len = max_len
        self.frames = deque(maxlen = max_len)
        self.actions = deque(maxlen = max_len)
        self.rewards = deque(maxlen = max_len)
        self.done = deque(maxlen = max_len)

    def add_to_memory(self,next_frame, next_reward, next_action, is_done):
        self.frames.append(next_frame)
        self.actions.append(next_action)
        self.rewards.append(next_reward)
        self.done.append(is_done)


class Agent():
    def __init__(self,action_space,replay_memory_len,replay_memory_max_len,alpha, debug = False):
        self.replay_memory = ReplayMemory(replay_memory_max_len)
        self.starting_mem_len = replay_memory_len
        self.action_space = action_space
        self.gamma = 0.99
        self.alpha = alpha
        self.total_steps = 0
        self.num_trained = 0
        self.model = self.build_model()
        self.model_target = clone_model(self.model)


    def build_model(self):
        model = Sequential()
        model.add(Input((84,84,4)))
        model.add(Conv2D(filters = 32,kernel_size = (8,8),strides = 4,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(filters = 64,kernel_size = (4,4),strides = 2,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(filters = 64,kernel_size = (3,3),strides = 1,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Flatten())
        model.add(Dense(512,activation = 'relu', kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(GaussianNoise(0.01))
        model.add(Dense(len(self.action_space), activation = 'linear'))
        optimizer = Adam(self.alpha)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        # model.summary()
        return model

    def select_action(self,state):
        # Greedy action selection

        a_index = np.argmax(self.model.predict(state))
        return self.action_space[a_index]


    def is_valid(self,index):
        if self.replay_memory.done[index-3] or self.replay_memory.done[index-2] or self.replay_memory.done[index-1] or self.replay_memory.done[index]:
            return False
        else:
            return True

    def train(self,debug = False):
        states = []
        next_states = []
        actions_selected = []
        next_rewards = []
        next_done = []

        while True:

            if len(states) < 32:
                index = np.random.randint(4,len(self.replay_memory.frames) - 1)

                if self.is_valid(index):
                    
                    current_state = [self.replay_memory.frames[index-3], self.replay_memory.frames[index-2], self.replay_memory.frames[index-1], self.replay_memory.frames[index]]
                    current_state = np.moveaxis(current_state,0,2)/255
                    
                    next_state = [self.replay_memory.frames[index-2], self.replay_memory.frames[index-1], self.replay_memory.frames[index], self.replay_memory.frames[index+1]]
                    next_state = np.moveaxis(next_state,0,2)/255

                    states.append(current_state)
                    next_states.append(next_state)
                    actions_selected.append(self.replay_memory.actions[index])
                    next_rewards.append(self.replay_memory.rewards[index+1])
                    next_done.append(self.replay_memory.done[index+1])
            else:
                break

        #Get the ouputs from our model and the target model
        predicted_values = self.model.predict(np.array(states))
        target_predicted_values = self.model_target.predict(np.array(next_states))
        
        #Perform Bellman update for state and selected action
        for i in range(32):
            action = self.action_space.index(actions_selected[i])
            if next_done[i]:
                predicted_values[i][action] = next_rewards[i]
            else:
                predicted_values[i][action] = next_rewards[i] + (self.gamma * max(target_predicted_values[i]))


        #Train model using the updated state Q values
        self.model.fit(np.array(states),predicted_values,batch_size = 32, epochs = 1, verbose = 0)

        self.num_trained += 1
        
        #Update target model
        if self.num_trained % 10000 == 0:
            self.model_target.set_weights(self.model.get_weights())