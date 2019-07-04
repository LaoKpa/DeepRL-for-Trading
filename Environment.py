import numpy as np

class Environment():

    def __init__(self, data):

        if data == None:
            self.dataset = None
        else:
            self.dataset = data
        self.counter = 0
        self.rando = 0
        self.stock = 0

    def step(self):
        obsn = self.stock[self.counter]
        self.counter += 1
        return obsn

    def reset(self, window):
        self.counter = 0
        self.rando = np.random.randint(0, len(self.dataset))
        self.stock = self.dataset[self.rando]
        series_len = len(self.stock)

        state = deque(maxlen=window)
        for i in range(window):
            state.append(0)

        return state, series_len


    def run_exp(self, agent, x_arr, y_arr, episodes, batch_size):
        for episode in range(episodes):
            x_batch = []
            y_batch = []

            for i in range(batch_size):
                rando = np.random.randint(0, train_size)
                x_sample = x_arr[rando]
                y_sample = y_arr[rando]
                x_batch.append(x_sample)
                y_batch.append(y_sample)

            y_batch = np.array(y_batch)
            x_batch = np.array(x_batch)
            agent.train(x_batch, y_batch)

    #
    # def run_deep_exp(self, agent):
    # for episode in range(episodes):
    #     state, series_length = env.reset(stacks)
    #     old_obsn = 0
    #     totalReward = 0
    #     done = False
    #
    #     for i in range(series_length):
    #         #         print(i)
    #         state_in = np.array(state)
    #         state_in = np.expand_dims(state_in, axis=0)
    #         #         print(state, state_in.shape)
    #
    #         action, Q = agent.act(state_in)
    #         obsn = int(env.step())
    #
    #         change = obsn - old_obsn
    #         reward = (change * action)
    #         totalReward += reward
    #         prev_state = cp.deepcopy(state)
    #         # new state
    #         state.append(obsn)
    #
    #         save_state = np.array(state)
    #         save_state_in = np.expand_dims(save_state, axis=0)
    #         prev_save_state = np.array(prev_state)
    #         prev_save_state_in = np.expand_dims(prev_save_state, axis=0)
    #
    #         # store experience in memory
    #         agent.memory.append((prev_save_state_in, action, reward, save_state_in, done))
    #         print(prev_state, action, reward, state, done)
    #
    #         if (i % 10 == 0):
    #             print("T", i, "PNL:", totalReward, Q)
    #
    #         if i > batch_size:
    #             minibatch = random.sample(agent.memory, batch_size)
    #             losso = agent.train(minibatch)
    #             pltLoss.append(losso)
    #
    #         if (i % 50 == 0):
    #             agent.target_train()
    #
    #         old_obsn = obsn
    #
    #     print('End of Episode: ', episode, ' reward: ', totalReward, 'length: ', i)
    #
    #     pltLengths.append(i)
    #     pltRewards.append(totalReward)
    #
    # print(lifeRewards)
    # print(pltLoss)