import gym
from gym.wrappers.record_video import RecordVideo
from gym.core import ObservationWrapper, Wrapper
from gym.spaces import Box
import matplotlib.pyplot as plt

from collections import deque
import tensorflow as tf
print(tf.__version__)
import random
import numpy as np
import cv2

class Memory:
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.append((state, action, reward, newState, isFinal))

    def getCurrentSize(self):
        return len(self.states)
    
    def getMiniBatch(self, size):
        batch = random.sample(self.memory, size)
        return batch

class PreprocessAtari(ObservationWrapper):
    def __init__(self, env):
        ObservationWrapper.__init__(self,env)

        self.img_size = (84, 84)
        self.observation_space = Box(0.0, 1.0, (self.img_size[0], self.img_size[1], 1))

    def observation(self, img):
        img = img[34:-16, :, :]
        img = cv2.resize(img, self.img_size)
        img = img.mean(-1, keepdims=True)
        img = img.astype('float32') / 255.
        return img
    
class FrameBuffer(Wrapper):
    def __init__(self, env, n_frames=4, dim_order="tenserflow"):
        super(FrameBuffer, self).__init__(env)
        self.dim_order = dim_order
        height, width, n_channels = env.observation_space.shape
        obs_shape = [height, width, n_channels * n_frames]
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.frame_buffer = np.zeros(obs_shape, 'float32')

    def reset(self):
        self.frame_buffer = np.zeros_like(self.frame_buffer)
        self.update_buffer(self.env.reset()[0])
        return self.frame_buffer

    def step(self, action):
        res = self.env.step(action)
        new_img = res[0]
        reward = res[1]
        done = res[2]
        info = res[3]
        self.update_buffer(new_img)
        return self.frame_buffer, reward, done, info
    
    def update_buffer(self, img):
        offset = env.observation_space.shape[-1] - 1
        axis = -1
        cropped_frame_buffer = self.frame_buffer[:,:,:offset]
        self.frame_buffer = np.concatenate([img, cropped_frame_buffer], axis = axis)

class DeepQN:
    def __init__(self, state_shape, n_actions, memory_size, learning_rate, discountFactor):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.memory = Memory(memory_size)
        self.learning_rate = learning_rate
        self.discountFactor = discountFactor

        self.network = tf.keras.models.Sequential()
        self.network.add(tf.keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=state_shape,
                                                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        self.network.add(tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu',
                                                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        self.network.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu',
                                                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        self.network.add(tf.keras.layers.Conv2D(1024, (7, 7), strides=1, activation='relu',
                                                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        self.network.add(tf.keras.layers.Flatten())
        self.network.add(tf.keras.layers.Dense(n_actions, activation='linear', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        self.network.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer)
        self.network.summary()        

    def getQValues(self, state):
        state = np.expand_dims(state, axis=0)
        qValues = self.network.predict(state, verbose=0)
        return qValues[0]
    
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.n_actions)
        else :
            action = np.argmax(qValues)
        return action

    def addToMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def calculateTarget(self, qValuesNewState, reward, final):
        if final:
            return reward
        else:
            return reward + self.discountFactor * np.max(qValuesNewState)

    def trainOnMiniBatch(self, minibatch_size):
        minibatch = self.memory.getMiniBatch(minibatch_size)
        X_batch = np.empty((0, self.state_shape[0], self.state_shape[1], self.state_shape[2]), dtype = np.float64)
        Y_batch = np.empty((0, self.n_actions), dtype = np.float64)
        for sample in minibatch:
            state = sample[0]
            action = sample[1]
            reward = sample[2]
            newState = sample[3]
            isFinal = sample[4]
            qValues = self.getQValues(state)
            qValuesNewState = self.getQValues(newState)
            targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

            X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
            Y_sample = qValues.copy()
            Y_sample[action] = targetValue
            Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
            if isFinal:
                X_batch = np.append(X_batch, np.array([newState.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[reward] * self.n_actions]), axis=0)
        self.network.fit(X_batch, Y_batch, batch_size = len(minibatch), verbose = 0)


env = gym.make("BreakoutDeterministic-v4", render_mode="rgb_array")
env = RecordVideo(env, video_folder="./atari_DQN",  episode_trigger=lambda t: t % 50 == 0)
env = PreprocessAtari(env)
env = FrameBuffer(env, n_frames=4, dim_order='tensorflow')

epochs = 100
steps = 10000
explorationRate = 0.5
learnStart = 64
memorySize = 500
learningRate = 0.1
discountFactor = 0.9
stepCounter = 0
minibatch_size = 10

n_actions = env.action_space.n
state_dim = env.observation_space.shape
model = DeepQN(state_dim, n_actions, memorySize, learningRate, discountFactor)

for epoch in range(epochs):
    observation = env.reset()
    observation = env.frame_buffer
    startCount = stepCounter
    for t in range(steps):
        qValues = model.getQValues(observation)
        action = model.selectAction(qValues, explorationRate)
        return_val = env.step(action)
        newObservation = return_val[0]
        reward = return_val[1]
        done = return_val[2]
        info = return_val[3]
        model.addToMemory(observation, action, reward, newObservation, done)
        if stepCounter >= learnStart and stepCounter % 10 == 0:
            model.trainOnMiniBatch(minibatch_size)
        stepCounter += 1
        if done:
            break
        observation = newObservation
    print("Episode ",epoch," finished after {} timesteps".format(stepCounter - startCount))
    explorationRate *= 0.995
    explorationRate = max(0.05, explorationRate)