import tensorflow as tf
import gym
import random
import numpy as np

from cart_pole import QAgent

agent = QAgent

MAX_EPISODES = 3000

for i_episode in range(MAX_EPISODES):
    Done = False
    obs = env.reset()
    score = 0

    while not Done:
        