import DataStream as ds
from Agent import Agent
from Environment import Environment
import TimeseriesGenerator as gen
import numpy as np

path = 'C:/Users/joehe/Documents/Summer Project/kibot_min/kibot_min'
files= 1
data = ds.readInFiles(path, files)

# size of window for each timeseries
win = 10
train_size = 10000
test_size = 1000
x_train, y_train = gen.make_timeseries(win, train_size)
x_test, y_test = gen.make_timeseries(win, test_size)

b_size = 100
eps = 1000
action_space = 2
observetime = 10000
max_mem = 1000000
batch_size = 10
episodes = 10
stacks = 4
state_size = 4
model_name = "JOE"
gamma = 0.95

# plotting records
lifeRewards = 0
pltLengths = []
pltRewards = []
pltLoss = []

env = Environment(data)
agentQ = Agent(action_space, state_size, max_mem, model_name, batch_size, gamma)
agentQ.run_exp(x_train, y_train, eps, b_size)
# state = np.expand_dims(x_test[0], axis=0)
# Q = agentQ.brain.predictAction(state)
# print("Ouput from network: ", int(Q),"Label: ", y_test[0])
agentQ.brain.model.evaluate(x_test, y_test)

