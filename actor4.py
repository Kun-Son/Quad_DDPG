"""
Data structure for implementing actor network for DDPG algorithm
Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf
Original author: Patrick Emami
Author: Bart Keulen
"""

import tensorflow as tf
import tflearn


class ActorNetwork4(object):

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor network
        self.inputs, self.outputs, self.scaled_outputs = self.create_actor_network()
        self.net_params = tf.trainable_variables()

        # Target network
        self.target_inputs, self.target_outputs, self.target_scaled_outputs = self.create_actor_network()
        self.target_net_params = tf.trainable_variables()[len(self.net_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = \
            [self.target_net_params[i].assign(tf.multiply(self.net_params[i], self.tau) +
                                              tf.multiply(self.target_net_params[i], 1. - self.tau))
             for i in range(len(self.target_net_params))]

        # Temporary placeholder action gradient
        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_dim])

        # Combine dnetScaledOut/dnetParams with criticToActionGradient to get actorGradient
        self.actor_gradients = tf.gradients(self.scaled_outputs, self.net_params, -self.action_gradients)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.net_params))

        self.num_trainable_vars = len(self.net_params) + len(self.target_net_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.state_dim])
        self.net1 = tflearn.fully_connected(inputs, 300, activation='relu')
        # self.net2 = tflearn.fully_connected(self.net1, 300, activation='relu')
        # self.net3 = tflearn.fully_connected(self.net2, 600, activation='relu')
        self.net4 = tflearn.fully_connected(self.net1, 400, activation='relu')
        # Final layer weight are initialized to Uniform[-3e-3, 3e-3]
        # weight_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        weight_init = tflearn.initializations.xavier(uniform=True, seed=None, dtype=tf.float32)
        outputs = tflearn.fully_connected(self.net4, self.action_dim, activation='tanh', weights_init=weight_init)
        scaled_outputs = tf.multiply(outputs, self.action_bound) # Scale output to [-action_bound, action_bound]

        return inputs, outputs, scaled_outputs

    def train(self, inputs, action_gradients):
        return self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradients: action_gradients
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_outputs, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_outputs, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_net_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

