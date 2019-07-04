from Model import Model
from collections import deque
import numpy as np

class Agent():

    def __init__(self, actions, state_size, max_mem, model_name, batch_size, gamma):
        self.inv = []
        self.action_space = actions
        self.memory = deque(maxlen=max_mem)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.brain = Model(self.action_space, state_size, batch_size, gamma)



    def run_exp(self, x_arr, y_arr, episodes, batch_size):
        for episode in range(episodes):
            x_batch = []
            y_batch = []

            for i in range(batch_size):
                rando = np.random.randint(0, len(y_arr))
                x_sample = x_arr[rando]
                y_sample = y_arr[rando]
                x_batch.append(x_sample)
                y_batch.append(y_sample)

            y_batch = np.array(y_batch)
            x_batch = np.array(x_batch)
            self.brain.simple_train(x_batch, y_batch)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            Q = 0
            action = np.random.randint(0, self.action_space)
        else:
            # predict Q values and choose maximum dependent on state
            Q = self.brain.model.predict(state)
        action = np.argmax(Q)
