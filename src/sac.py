import tflearn
import numpy as np
import tensorflow as tf

FEATURE_NUM = 128
ACTION_EPS = 1e-4
GAMMA = 0.99
MAX_POOL_NUM = 500000
TAU = 5e-3
Q_NETWORK_COUNT = 2
Target_entropy_ratio = 0.98

class Network():
    def CreateTarget(self, inputs, name):
        with tf.variable_scope(name):
            split_0 = tflearn.fully_connected(
                inputs[:, 0:1, -1], FEATURE_NUM, activation='relu')
            split_1 = tflearn.fully_connected(
                inputs[:, 1:2, -1], FEATURE_NUM, activation='relu')
            split_2 = tflearn.conv_1d(
                inputs[:, 2:3, :], FEATURE_NUM, 4, activation='relu')
            split_3 = tflearn.conv_1d(
                inputs[:, 3:4, :], FEATURE_NUM, 4, activation='relu')
            split_4 = tflearn.conv_1d(
                inputs[:, 4:5, :self.a_dim], FEATURE_NUM, 4, activation='relu')
            split_5 = tflearn.fully_connected(
                inputs[:, 5:6, -1], FEATURE_NUM, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            merge_net = tflearn.merge(
                [split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')

            net = tflearn.fully_connected(
                merge_net, FEATURE_NUM, activation='relu')

            value = tflearn.fully_connected(
                net, self.a_dim, activation='linear')

            return value

    def CreatePolicy(self, inputs, name):
        with tf.variable_scope(name):
            split_0 = tflearn.fully_connected(
                inputs[:, 0:1, -1], FEATURE_NUM, activation='relu')
            split_1 = tflearn.fully_connected(
                inputs[:, 1:2, -1], FEATURE_NUM, activation='relu')
            split_2 = tflearn.conv_1d(
                inputs[:, 2:3, :], FEATURE_NUM, 4, activation='relu')
            split_3 = tflearn.conv_1d(
                inputs[:, 3:4, :], FEATURE_NUM, 4, activation='relu')
            split_4 = tflearn.conv_1d(
                inputs[:, 4:5, :self.a_dim], FEATURE_NUM, 4, activation='relu')
            split_5 = tflearn.fully_connected(
                inputs[:, 5:6, -1], FEATURE_NUM, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            merge_net = tflearn.merge(
                [split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')

            net = tflearn.fully_connected(
                merge_net, FEATURE_NUM, activation='relu')

            pi = tflearn.fully_connected(net, self.a_dim, activation='softmax')

            return pi

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.entropy_ = 2.
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.sess = sess

        self.target_entropy = tf.placeholder(tf.float32)
        # self.log_alpha = tf.get_variable('sac_log_alpha', dtype=tf.float32, initializer=2.)
        self.alpha =  tf.placeholder(tf.float32) #tf.get_variable('sac_alpha', dtype=tf.float32, initializer=3.)

        self.r = tf.placeholder(tf.float32, [None, 1])
        self.done = tf.placeholder(tf.float32, [None, 1])

        self.inputs = tf.placeholder(
            tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        self.next_state = tf.placeholder(
            tf.float32, [None, self.s_dim[0], self.s_dim[1]])

        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])

        self.pi_out = self.CreatePolicy(inputs=self.inputs, name='pi_eval')
        self.pi = tf.clip_by_value(self.pi_out, 1e-4, 1. - 1e-4)

        self.pi_ns_out = self.CreatePolicy(
            inputs=self.next_state, name='pi_target')
        self.pi_ns = tf.clip_by_value(self.pi_ns_out, 1e-4, 1. - 1e-4)
        self.pi_params = \
            tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi_eval')
        self.pi_target_params = \
            tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi_target')
        self.pi_soft_update = [tf.assign(ta, ea)
                               for ta, ea in zip(self.pi_target_params, self.pi_params)]
        # mean
        self.entropy = - tf.reduce_sum(tf.multiply(self.pi, tf.log(self.pi)), reduction_indices=1, keepdims=True)

        self.q_eval_s, self.q_target_s, self.q_soft_update_s = [], [], []

        for i in range(Q_NETWORK_COUNT):
            q_eval = self.CreateTarget(
                inputs=self.inputs, name='q_eval_' + str(i))
            q_target = self.CreateTarget(
                inputs=self.next_state, name='q_target_' + str(i))
            q_eval_params = \
                tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_eval_' + str(i))
            q_target_params = \
                tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_target_' + str(i))
            q_soft_update = [tf.assign(ta, (1 - TAU) * ta + TAU * ea)
                             for ta, ea in zip(q_target_params, q_eval_params)]
            self.q_eval_s.append(q_eval)
            self.q_target_s.append(q_target)
            self.q_soft_update_s.append(q_soft_update)

        self.min_q_target = tf.reduce_min(self.q_target_s, axis=0, keepdims=True)[0]
        self.min_q_eval = tf.reduce_min(self.q_eval_s, axis=0, keepdims=True)[0]

        self.value = tf.reduce_sum(tf.multiply(self.pi_ns, self.min_q_target), reduction_indices=1, keepdims=True)
        self.q_target = self.r + GAMMA * (1 - self.done) * (self.value + self.alpha * self.entropy)

        self.pool = []

        # Get all network parameters
        self.network_params = \
            tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi_eval')
        self.network_params += \
            tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi_target')
        for i in range(Q_NETWORK_COUNT):
            self.network_params += \
                tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_eval_' + str(i))
            self.network_params += \
                tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_target_' + str(i))

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))

        self.pi_loss = - tf.reduce_mean(self.alpha * self.entropy + tf.reduce_sum(tf.multiply(self.pi, tf.stop_gradient(self.min_q_eval)), reduction_indices=1, keepdims=True))
        self.sac_loss = self.pi_loss
        if len(self.q_eval_s) > 0:
            for t in range(len(self.q_eval_s)):
                self.sac_loss += 0.5 * tflearn.mean_square(
                    tf.reduce_sum(tf.multiply(self.q_eval_s[t], self.acts), reduction_indices=1, keepdims=True),
                    tf.stop_gradient(self.q_target))

        self.sac_opt = tf.train.AdamOptimizer(
            self.lr_rate).minimize(self.sac_loss)

    def get_entropy(self, step):
        return np.clip(self.entropy_, 1e-10, 5.)
    
    def entropy_decay(self, decay=0.98):
        self.entropy_ *= decay

    def predict(self, input):
        action = self.sess.run(self.pi, feed_dict={
            self.inputs: input
        })
        return action[0]

    def train(self, s_batch, a_batch, ns_batch, r_batch, d_batch, epoch):
        # ns: next state
        for (s, a, r, ns, d) in zip(s_batch, a_batch, r_batch, ns_batch, d_batch):
            self.pool.append([s, a, r, ns, d])
            if len(self.pool) > MAX_POOL_NUM:
                pop_item = np.random.randint(len(self.pool))
                self.pool.pop(pop_item)

        if len(self.pool) > 10000:
            s_batch, a_batch, r_batch = [], [], []
            ns_batch, d_batch = [], []

            for p in range(1024):
                pop_item = np.random.randint(len(self.pool))
                s_, a_, r_, n_, d_ = self.pool[pop_item]
                s_batch.append(s_)
                a_batch.append(a_)
                r_batch.append(r_)
                ns_batch.append(n_)
                d_batch.append(d_)

            self.sess.run(self.sac_opt, feed_dict={
                self.inputs: s_batch,
                self.acts: a_batch,
                self.r: r_batch,
                self.done: d_batch,
                self.next_state: ns_batch,
                self.alpha: self.get_entropy(epoch)
            })

            self.sess.run(self.pi_soft_update)
            self.sess.run(self.q_soft_update_s)
