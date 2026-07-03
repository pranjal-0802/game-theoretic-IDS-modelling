class DDPGbase(object):
    def __init__(self, a_dim, s_dim):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.reparam_noise = 1e-6
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.is_training = tf.placeholder(tf.bool)
        #self.saver = tf.train.Saver()

        self.a = self._build_a(self.S,)
        q1 = self._build_c1(self.S, self.a, )
        q2 = self._build_c2(self.S, self.a, )
        self.Qvalue1 = q1
        self.Qvalue2 = q2
        self.v = self._build_v(self.S,)

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c1_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic1')
        c2_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic2')
        v_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Value')
        
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        # target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        
        target_update = [ema.apply(v_params)]
        # a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        # q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)
        self.v_ = self._build_v(self.S_, reuse=True, custom_getter=ema_getter)

        a_loss = tf.reduce_sum(0.5*tf.math.log(self.a), axis = 1) - tf.squeeze(tf.math.minimum(q1, q2))  # maximize the q
        a_loss = tf.reduce_mean(a_loss)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        q_hat = self.R + GAMMA * self.v_
        critic_1_loss = 0.5*tf.losses.mean_squared_error(labels=q_hat, predictions=q1)
        critic_2_loss = 0.5*tf.losses.mean_squared_error(labels=q_hat, predictions=q2)
        self.ctrain1 = tf.train.AdamOptimizer(LR_C).minimize(critic_1_loss, var_list=c1_params)
        self.ctrain2 = tf.train.AdamOptimizer(LR_C).minimize(critic_2_loss, var_list=c2_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            critic_value = tf.squeeze(tf.math.minimum(q1, q2))
            log_probs = tf.reduce_sum(tf.math.log(self.a), axis = 1)
            v_target = critic_value-log_probs
            td_error = 0.5*tf.losses.mean_squared_error(labels=v_target, predictions=self.v)
            self.vtrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=v_params)

        # with tf.control_dependencies(target_update):    # soft replacement happened at here
        #     q_target = self.R + GAMMA * q_
        #     td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        #     self.ctrain1 = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c1_params)

        # with tf.control_dependencies(target_update):    # soft replacement happened at here
        #     q_target = self.R + GAMMA * q_
        #     td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        #     self.ctrain2 = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c2_params)
        
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def Q_value(self, s, a):
        return np.minimum(self.sess.run(self.Qvalue1, {self.S: s[np.newaxis, :], self.a: a[np.newaxis, :]})[0], \
            self.sess.run(self.Qvalue2, {self.S: s[np.newaxis, :], self.a: a[np.newaxis, :]})[0])

    def learn(self):
        indices = np.random.choice(min(MEMORY_CAPACITY, self.pointer), size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # value = tf.squeeze(self.sess.run(self.v, {self.S: bs}))
        # value_ = tf.squeeze(self.sess.run(self.v_, {self.S: bs_}))
        
        # actions = self.sess.run(self.a, {self.S: bs})
        # q1_new_policy = self.sess.run(self.Qvalue1, {self.S: bs, self.a: actions})
        # q2_new_policy = self.sess.run(self.Qvalue2, {self.S: bs, self.a: actions})
        # critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy))

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.vtrain, {self.S: bs})
        self.sess.run(self.ctrain1, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        self.sess.run(self.ctrain2, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        
    def sample_normal(self, mu, sigma):
        return tfd.Normal(loc=mu, scale=sigma)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        raise NotImplementedError

    def _build_v(self, s, reuse=None, custom_getter=None):
        raise NotImplementedError

    def _build_c1(self, s, a, reuse=None, custom_getter=None):
        raise NotImplementedError

    def _build_c2(self, s, a, reuse=None, custom_getter=None):
        raise NotImplementedError

class DDPGdefend(DDPGbase):
    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            h1 = tf.layers.dense(s, H_DEF, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer(),name='h1', trainable=trainable)
            a = tf.clip_by_value(tf.layers.dense(h1, self.a_dim, activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='a', trainable=trainable), 1e-6, 1)
            return a

    def _build_c1(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic1', reuse=reuse, custom_getter=custom_getter):
            n_l1 = H_DEF
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=tf.keras.initializers.he_normal(), trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=tf.keras.initializers.he_normal(), trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            h1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            Q = tf.layers.dense(h1, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)  # Q(s,a)
            return Q

    def _build_c2(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic2', reuse=reuse, custom_getter=custom_getter):
            n_l1 = H_DEF
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=tf.keras.initializers.he_normal(), trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=tf.keras.initializers.he_normal(), trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            h1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            Q = tf.layers.dense(h1, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)  # Q(s,a)
            return Q

    def _build_v(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Value', reuse=reuse, custom_getter=custom_getter):
            h1 = tf.layers.dense(s, H_DEF, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer(),name='h1', trainable=trainable)
            a = tf.layers.dense(h1, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='a', trainable=trainable)
            return a


class DDPGattack(DDPGbase):
    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            h1 = tf.layers.dense(s, H_ADV, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer(),name='h1', trainable=trainable)
            a = tf.clip_by_value(tf.layers.dense(h1, self.a_dim, activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='a', trainable=trainable), 1e-6, 1)
            return a

    def _build_c1(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic1', reuse=reuse, custom_getter=custom_getter):
            n_l1 = H_ADV
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=tf.keras.initializers.he_normal(), trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=tf.keras.initializers.he_normal(), trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            h1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            Q = tf.layers.dense(h1, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)  # Q(s,a)
            return Q

    def _build_c2(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic2', reuse=reuse, custom_getter=custom_getter):
            n_l1 = H_ADV
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=tf.keras.initializers.he_normal(), trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=tf.keras.initializers.he_normal(), trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            h1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            Q = tf.layers.dense(h1, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)  # Q(s,a)
            return Q

    def _build_v(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Value', reuse=reuse, custom_getter=custom_getter):
            h1 = tf.layers.dense(s, H_ADV, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer(),name='h1', trainable=trainable)
            a = tf.layers.dense(h1, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='a', trainable=trainable)
            return a