import numpy as np
import random
import copy
import collections
import time
import pickle
import tensorflow as tf

from treys import Card
from treys import Evaluator
from treys import Deck
from treys import CardSet

from memory import Memory
from policy import Policy
from qnetwork import QNetwork
from gamestate import GameState
from tree_node import TreeNode

class Pluribus:
    def __init__(self, init_state=None, num_players=6, num_layers=1, num_nodes=64,
                 self_play=False, dqn=False, num_episodes=1000, batch_size=32, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=10000):
        self.init_state = init_state
        self.num_players = num_players
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.self_play = self_play
        self.dqn = dqn
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.eval = Evaluator()
        self.deck = Deck()

        self.action_history = []
        self.reward_history = []

        self.policy_net = Policy(num_players, num_layers, num_nodes)
        self.target_net = Policy(num_players, num_layers, num_nodes)

        self.q_net = QNetwork(num_players, num_layers, num_nodes)

        self.memory = Memory()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.epsilon = self.epsilon_start
        
    def state_to_input(self, state):
        input_data = np.zeros((self.num_players, 3, 52))
        for i in range(self.num_players):
            input_data[i][0] = np.array([Card.get_rank_int(c) for c in state.hands[i]])
            input_data[i][1] = np.array([Card.get_suit_int(c) for c in state.hands[i]])
            input_data[i][2] = np.array([1 if c in state.board else 0 for c in state.hands[i]])
        return input_data.reshape((1, -1))

    def compute_policy_gradients(self, state):
        action_probs = self.sess.run(self.policy_net.action_probs,
                                     feed_dict={self.policy_net.inputs: self.state_to_input(state)})
        legal_actions = state.legal_actions()
        probs = np.zeros(len(action_probs))
        for i in range(len(action_probs)):
            if i in legal_actions:
                probs[i] = action_probs[i]
        probs /= np.sum(probs)
        action = np.random.choice(len(probs), p=probs)
        loss = -tf.reduce_mean(tf.log(probs[action]) *
                               self.policy_net.log_probs[0][action])
        grads = tf.gradients(loss, self.policy_net.trainable_variables)
        return grads, action, action_probs

    def update_policy_network(self, grads):
        self.sess.run(self.policy_net.optimizer, feed_dict=dict(zip(self.policy_net.grads, grads)))    
        
    def self_play_game(self):
        state = copy.deepcopy(self.init_state)
        while not state.is_terminal():
            current_player = state.current_player
            action_probs = self.sess.run(self.policy_net.action_probs,
                                         feed_dict={self.policy_net.inputs: self.state_to_input(state)})
            legal_actions = state.legal_actions()
            probs = np.zeros(len(action_probs))
            for i in range(len(action_probs)):
                if i in legal_actions:
                    probs[i] = action_probs[i]
            probs /= np.sum(probs)
            action = np.random.choice(len(probs), p=probs)
            action_card = Card.int_to_str(action)
            action_str = 'CHECK' if action_card == '2c' else 'BET'
            action_size = 0 if action_str == 'CHECK' else 100
            state.perform_action(current_player, action_str, action_size)
            next_state = copy.deepcopy(state)
            reward = np.zeros(self.num_players)
            if next_state.is_terminal():
                rewards = next_state.rewards()
                for i in range(self.num_players):
                    reward[i] = rewards[i]
            else:
                for i in range(self.num_players):
                    if i != current_player:
                        action_probs = self.sess.run(self.policy_net.action_probs,
                                                     feed_dict={self.policy_net.inputs: self.state_to_input(next_state)})
                        legal_actions = next_state.legal_actions()
                        probs = np.zeros(len(action_probs))
                        for j in range(len(action_probs)):
                            if j in legal_actions:
                                probs[j] = action_probs[j]
                        probs /= np.sum(probs)
                        action = np.random.choice(len(probs), p=probs)
                        action_card = Card.int_to_str(action)
                        action_str = 'CHECK' if action_card == '2c' else 'BET'
                        action_size = 0 if action_str == 'CHECK' else 100
                        next_state.perform_action(i, action_str, action_size)
                next_state = GameState(next_state.hands, next_state.board, next_state.current_player)
            self.memory.add(state, action, reward, next_state)
            state = copy.deepcopy(next_state)
            

def train_dqn(env, q_net, target_net, optimizer, memory, batch_size, gamma):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = select_action(state, q_net)
        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Train the network using a random batch of experiences from memory
        if len(memory) > batch_size:
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*experiences)

            # Convert the experience tuples to PyTorch tensors
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            # Compute the Q values for the current state and the next state using the Q network and the target network
            q_values = q_net(states).gather(1, actions)
            next_q_values = target_net(next_states).detach().max(1)[0].unsqueeze(1)
            expected_q_values = rewards + gamma * next_q_values * (1 - dones)

            # Compute the loss and update the Q network
            loss = F.mse_loss(q_values, expected_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the target network
            update_target(target_net, q_net)

    return total_reward
