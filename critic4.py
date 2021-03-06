"""
Data structure for implementing critic network for DDPG algorithm
Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf
Original author: Patrick Emami
Author: Bart Keulen
"""

import tensorflow as tf
import tflearn


class CriticNetwork4(object):

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Critic network
        self.inputs, self.action, self.outputs = self.create_critic_network()
        self.net_params = tf.trainable_variables()[num_actor_vars:]

        # Target network
        self.target_inputs, self.target_action, self.target_outputs = self.create_critic_network()
        self.target_net_params = tf.trainable_variables()[len(self.net_params) + num_actor_vars:]

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = \
            [self.target_net_params[i].assign(tf.multiply(self.net_params[i], self.tau) +
                                              tf.multiply(self.target_net_params[i], 1. - self.tau))
             for i in range(len(self.target_net_params))]

        # Network target (y_i)
        # Obtained from the target networks
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.outputs)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the critic w.r.t. the action
        self.action_grads = tf.gradients(self.outputs, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.state_dim])
        action = tflearn.input_data(shape=[None, self.action_dim])
        net1 = tflearn.fully_connected(inputs, 300, activation='relu')
        # net2 = tflearn.fully_connected(net1, 300, activation='relu')
        # net3 = tflearn.fully_connected(net2, 600, activation='relu')
        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net1, 400)
        t2 = tflearn.fully_connected(action, 400, activation='relu')
        net = tflearn.activation(tf.matmul(net1, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        # Linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        weight_init = tflearn.initializations.xavier(uniform=True, seed=None, dtype=tf.float32)
        outputs = tflearn.fully_connected(net, 1, weights_init=weight_init)
        return inputs, action, outputs

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.outputs, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.outputs, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_outputs, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, action):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def update_target_network(self):
        self.sess.run(self.update_target_net_params)