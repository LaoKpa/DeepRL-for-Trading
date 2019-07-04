import tensorflow as tf
import numpy as np

class Model():

    def __init__(self, action_space, state_size, batch_size, gamma):

        self.action_space = action_space
        self.state_size = state_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.model = self.make_model()
        self.target_model = self.make_model()


    def make_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])
        return model



    def simple_train(self, x_batch, y_batch):
        self.model.fit(x_batch, y_batch, verbose=0)

    def predictAction(self, x):
        self.model.predict(x)

    def train(self, minibatch):
        inputs = np.zeros((self.batch_size, self.state_size))
        targets = np.zeros((self.batch_size, self.action_space))

        # for each sample in minibatch
        for i in range(0, self.batch_size):
            state = minibatch[i][0]
            action = minibatch[i][1]
            reward = minibatch[i][2]
            state_new = minibatch[i][3]

            # Build Bellman equation for the Q function
            inputs[i] = state
            targets[i] = self.model.predict(state)
            Q_sa = self.target_model.predict(state_new)
            targets[i, action] = reward + self.gamma * np.max(Q_sa)

        # Train network to output the Q function
        loss = self.model.train_on_batch(inputs, targets)
        return loss

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())
