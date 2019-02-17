import tensorflow as tf
import gym
import random
import numpy as np

MAX_SCORE_QUEUE_SIZE = 10
GAME = 'CartPole-v0'

MAX_EPISODE = 30000000 # max number of episodes iteration
ACTION_DIM = 2
OBSERVATION_DIM = 4
GAMMA = 0.9 # discount factor of Q learning
INIT_EPS = 1e-4 # 1.0 # initial probability for randomly sampling action
FINAL_EPS = 1e-4 # finial probability for randomly sampling action
EPS_DECAY = 0.95 # epsilon decay rate
EPS_ANNEAL_STEPS = 10 # steps interval to decay epsilon
LR = 1e-5 # learning rate
MAX_EXPERIENCE = 30000 # size of experience replay memory
BATCH_SIZE = 256 # mini batch size
H1_SIZE = 128
H2_SIZE = 128
H3_SIZE = 128

class QAgent:
    
    def __init__(self):
        self.default_initializer = tf.contrib.layers.xavier_initializer()
        self.W1 = self.var([OBSERVATION_DIM, H1_SIZE])
        self.b1 = self.var([H1_SIZE])
        self.W2 = self.var([H1_SIZE, H2_SIZE])
        self.b2 = self.var([H2_SIZE])
        self.W3 = self.var([H2_SIZE, H3_SIZE])
        self.b3 = self.var([H3_SIZE])
        self.W4 = self.var([H3_SIZE, ACTION_DIM])
        self.b4 = self.var([ACTION_DIM])

    def var(self, shape):
        return tf.Variable(self.default_initializer(shape))

    def add_value_net(self):
        observation = tf.placeholder(tf.float32, [None, OBSERVATION_DIM])
        h1 = tf.nn.relu(tf.matmul(observation, self.W1) + self.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.W2) + self.b2)
        h3 = tf.nn.relu(tf.matmul(h2, self.W3) + self.b3)
        Q = tf.squeeze(tf.matmul(h3, self.W4) + self.b4)

        return observation, Q

    def sample_action(self, Q, feed, eps): # epsilon greedy
        act_values = Q.eval(feed_dict=feed)

        if random.random() <= eps:
            action_index = random.randrange(ACTION_DIM)
        else:
            action_index = np.argmax(act_values)
        
        action = np.zeros(ACTION_DIM)
        action[action_index] = 1
        
        return action

def train(env):

    MAX_SCORE_EARNED = 2000

    agent = QAgent()
    sess = tf.InteractiveSession()

    obs, Q1 = agent.add_value_net()
    next_obs, Q2 = agent.add_value_net()
    act = tf.placeholder(tf.float32, [None, ACTION_DIM])
    rwd = tf.placeholder(tf.float32, [None, ])

    values1 = tf.reduce_sum(tf.multiply(Q1, act), axis=1)
    values2 = rwd + GAMMA * tf.reduce_max(Q2, axis=1)

    loss = tf.reduce_mean(tf.square(values1 - values2))
    train_step = tf.train.AdamOptimizer(LR).minimize(loss)
    
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()    
    checkpoint = tf.train.get_checkpoint_state("checkpoints-cartpole")

    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    
    else:
        print("Could not find old network weights")

    feed = {}
    eps = INIT_EPS
    global_step = 0
    exp_pointer = 0
    learning_finished = True

    # The replay memory
    obs_queue = np.empty([MAX_EXPERIENCE, OBSERVATION_DIM])
    act_queue = np.empty([MAX_EXPERIENCE, ACTION_DIM])
    rwd_queue = np.empty([MAX_EXPERIENCE])

    next_obs_queue = np.empty([MAX_EXPERIENCE, OBSERVATION_DIM])

    score_queue = [] # Score cache

    for i_episode in range(MAX_EPISODE):

        observation = env.reset()
        done = False
        score = 0
        sum_loss_value = 0

        # mov_queue = np.empty([])

        # step loop
        while not done:
            global_step += 1

            # exploration eps decay
            if global_step % EPS_ANNEAL_STEPS == 0 and eps > FINAL_EPS:
                eps = eps * EPS_DECAY
            
            env.render()

            obs_queue[exp_pointer] = observation
            action = agent.sample_action(Q1, {obs : [observation]}, eps)
            act_queue[exp_pointer] = action
            observation, reward, done, _ = env.step(np.argmax(action)) # index => action

            score += reward 
            reward += abs(observation[2]) * 0.1

            if done and score < 500:
                reward -= 4

            rwd_queue[exp_pointer] = reward
            next_obs_queue[exp_pointer] = observation

            exp_pointer += 1
            if exp_pointer == MAX_EXPERIENCE:
                exp_pointer = 0 # Refill the memory if it is full
            
            if global_step >= MAX_EXPERIENCE:
                rand_indexs = np.random.choice(MAX_EXPERIENCE, BATCH_SIZE) # random index for mini batch
                feed.update({obs : obs_queue[rand_indexs]})
                feed.update({act : act_queue[rand_indexs]})
                feed.update({rwd : rwd_queue[rand_indexs]})
                feed.update({next_obs : next_obs_queue[rand_indexs]})

                if not learning_finished : # If not solved, keep training
                    step_loss_value, _ = sess.run([loss, train_step], feed_dict = feed)
                
                else: # If solved, just get step loss
                    step_loss_value = sess.run(loss, feed_dict = feed)

                sum_loss_value += step_loss_value # Use sum to calculate average loss of current episode

        print("====== Episode %d ended with score = %f, avg_loss = %f ======" %(i_episode+1, score, sum_loss_value / score))
        
        score_queue.append(score)
        if len(score_queue) > MAX_SCORE_QUEUE_SIZE:
            score_queue.pop(0)
            if np.mean(score_queue) >= 3000: # The threshold of being solved
                learning_finished = True

        if learning_finished:
            print("Testing !!!")

        if learning_finished and MAX_SCORE_EARNED < np.mean(score_queue):
            saver.save(sess, 'checkpoints-cartpole/' + GAME + '-dqn', global_step = global_step)
            MAX_SCORE_EARNED = score
            env._max_episode_steps = score + 400

if __name__ == "__main__":
    env = gym.make(GAME)
    env._max_episode_steps = 3000
    train(env)