import tensorflow as tf

class GRU:

    def __init__(self, hidden_size, inputs, inputs_s, futures, loss_mask,
                 reverse=True, global_step=0, input_keep_prob=1.0, output_keep_prob=1.0):

        self.batch_num = inputs[0].get_shape().as_list()[0]
        self.input_dimensions = inputs[0].get_shape().as_list()[1]
        # self.zeta_squared = tf.constant(0.587213, dtype=tf.float32, shape=[1], name="zeta_squared")

        self.alpha_sigmoid = tf.constant(1.171573, dtype=tf.float32, shape=[1], name="alpha_sigmoid")
        self.alpha_squared_sigmoid = tf.constant(1.372583, dtype=tf.float32, shape=[1], name="alpha_sigmoid_squared")
        self.beta_sigmoid = tf.constant(-0.881374, dtype=tf.float32, shape=[1], name="beta_sigmoid")
        # self.zeta_squared_alpha_squared_sigmoid = tf.constant(0.805999, dtype=tf.float32, shape=[1], name="alpha_sigmoid_squared")

        self.alpha_tanh = tf.constant(2.343146, dtype=tf.float32, shape=[1], name="alpha_tanh")
        self.alpha_squared_tanh = tf.constant(5.490332, dtype=tf.float32, shape=[1], name="alpha_squared_tanh")
        self.beta_tanh = tf.constant(-0.440687, dtype=tf.float32, shape=[1], name="beta_tanh")
        # self.zeta_squared_alpha_squared_tanh = tf.constant(3.223994, dtype=tf.float32, shape=[1], name="zeta_squared_alpha_squared_tanh")

        # Their constants
        self.zeta_squared = tf.constant(0.392699, dtype=tf.float32, shape=[1], name="zeta_squared")
        self.zeta_squared_alpha_squared_sigmoid = tf.constant(0.805999, dtype=tf.float32, shape=[1], name="alpha_sigmoid_squared")
        self.zeta_squared_alpha_squared_tanh = tf.constant(3.223994, dtype=tf.float32, shape=[1], name="zeta_squared_alpha_squared_tanh")

        # self.zeta_squared = tf.constant(0.01, dtype=tf.float32, shape=[1], name="zeta_squared")
        # self.zeta_squared_alpha_squared_sigmoid = tf.constant(0.1, dtype=tf.float32, shape=[1], name="alpha_sigmoid_squared")
        # self.zeta_squared_alpha_squared_tanh = tf.constant(1.0, dtype=tf.float32, shape=[1], name="zeta_squared_alpha_squared_tanh")

        # self.input_dimensions = input_dimensions
        # self.hidden_size = hidden_size

        # This is the actual input used for forward propagation. Input dropout is applied here.
        inputs = tf.nn.dropout(inputs, keep_prob=input_keep_prob)
        inputs_s = tf.square(inputs_s)

        # ENCODER
        with tf.variable_scope('ENCODER'):

            # initialize encoder weights
            self.initialize_encoder(hidden_size, mean_m=0.0, mean_s=0.0, mean_b_m=0.0, mean_b_s=0.0, w_std=0.001)

            x_t = tf.transpose(inputs, [0, 1, 2], name='x_t')
            # x_t_s = tf.transpose(inputs_s, [0, 1, 2], name='x_t_s')
            x_t_s = tf.zeros(tf.shape(x_t), dtype=tf.float32, name='x_t_s')

            # hidden_inputs: (80, 2048) [batch_size, hidden_size]
            h_0_enc = tf.zeros(dtype=tf.float32, shape=(self.batch_num, hidden_size), name='h_0_enc')
            h_0_s_enc = tf.zeros(dtype=tf.float32, shape=(self.batch_num, hidden_size), name='h_0_s_enc')

            x_t_ms = tf.stack([x_t, x_t_s], axis=1, name='x_t_stack')
            h_0_ms_enc = tf.stack([h_0_enc, h_0_s_enc], axis=0, name='h_0_stack')
            print "x_t_ms.get_shape():", x_t_ms.get_shape()
            print "h_0_ms_enc.get_shape():", h_0_ms_enc.get_shape()

            h_t_transposed_enc = tf.scan(self.forward_pass_enc_np, x_t_ms, initializer=h_0_ms_enc, name='h_t_transposed_enc')

            print "h_t_transposed_enc.get_shape():", h_t_transposed_enc.get_shape()

            h_t_enc, h_t_s_enc = tf.unstack(h_t_transposed_enc, axis=1)
            h_t_enc = tf.transpose(h_t_enc, [1, 0, 2], name='h_t_enc')
            h_t_s_enc = tf.transpose(h_t_s_enc, [1, 0, 2], name='h_t_s_enc')

            print "-- ENCODER CELL:"
            print "x_t.get_shape():", x_t.get_shape()
            print "h_0_enc.get_shape():", h_0_enc.get_shape()
            print "h_t_transposed_enc.get_shape():", h_t_transposed_enc.get_shape()
            print "h_t_enc.get_shape():", h_t_enc.get_shape()


        # DECODER
        with tf.variable_scope('DECODER') as vs:

            # initialize decoder weights
            self.initialize_decoder(hidden_size, mean_m=0.0, mean_s=0.0, mean_b_m=0.0, mean_b_s=0.0, w_std=0.001)

            # x_t: (10, 80, 2048)
            x_t_dec = tf.zeros(tf.shape(inputs), dtype=tf.float32, name='x_t_dec')
            x_t_s_dec = tf.zeros(tf.shape(inputs), dtype=tf.float32, name='x_t_s')
            # x_t_dec = tf.transpose(inputs, [0, 1, 2], name='x_t_dec')
            # x_t_s_dec = tf.zeros(tf.shape(x_t_dec), dtype=tf.float32, name='x_t_s_dec')

            # hidden_inputs: (80, 2048) [batch_size, hidden_size]
            h_0_dec = h_t_enc[:, -1, :]
            h_0_s_dec = h_t_s_enc[:, -1, :]
            h_0_ms_dec = tf.stack([h_0_dec, h_0_s_dec], axis=0, name='h_0_dec_stack')
            x_t_ms_dec = tf.stack([x_t_dec, x_t_s_dec], axis=1, name='x_t_dec_stack')
            print "x_t_ms.get_shape():", x_t_ms.get_shape()
            print "h_0_ms_enc.get_shape():", h_0_ms_enc.get_shape()

            h_t_transposed_dec = tf.scan(self.forward_pass_dec_np, x_t_ms_dec, initializer=h_0_ms_dec, name='h_t_transposed_dec')

            h_t_dec, h_t_s_dec = tf.unstack(h_t_transposed_dec, axis=1)

            # reverse THEN transpose ONLY in decoder
            # if reverse: h_t_dec = h_t_dec[::-1]
            # if reverse: h_t_s_dec = h_t_s_dec[::-1]

            # h_t_dec: (80, 10, 2048) [batch_size, seq_length, hidden_size]
            h_t_dec = tf.transpose(h_t_dec, [1, 0, 2], name='h_t_dec')
            h_t_s_dec = tf.transpose(h_t_s_dec, [1, 0, 2], name='h_t_s_dec')

            W_m = tf.Variable(tf.truncated_normal([hidden_size, self.input_dimensions], dtype=tf.float32), name="dec_weight")
            W_m = tf.tile(tf.expand_dims(W_m, 0), [self.batch_num, 1, 1])
            b_m = tf.Variable(tf.constant(0.1, shape=[self.input_dimensions], dtype=tf.float32), name="dec_bias")

            W_s = tf.Variable(tf.truncated_normal([hidden_size, self.input_dimensions], dtype=tf.float32), name="dec_s_weight")
            W_s = tf.tile(tf.expand_dims(W_s, 0), [self.batch_num, 1, 1])
            b_s = tf.Variable(tf.constant(0.1, shape=[self.input_dimensions], dtype=tf.float32), name="dec_s_bias")

            print "-- DECODER CELL:"
            print "x_t_dec.get_shape():", x_t_dec.get_shape()
            print "h_t_enc.get_shape():", h_t_enc.get_shape()
            print "h_0_dec.get_shape():", h_0_dec.get_shape()
            print "h_t_transposed_dec.get_shape():", h_t_transposed_dec.get_shape()
            print "h_t_dec.get_shape():", h_t_dec.get_shape()
            print "h_t_s_dec.get_shape():", h_t_s_dec.get_shape()
            print "-- OUTPUT LAYER:"
            # print "dec_weight_.get_shape():", dec_weight_.get_shape()
            # print "dec_bias_.get_shape():", dec_bias_.get_shape()
            # print "dec_s_weight_.get_shape():", dec_s_weight_.get_shape()
            # print "dec_s_bias_.get_shape():", dec_s_bias_.get_shape()

            o_m_t = tf.matmul(h_t_dec, W_m) + b_m

            print "o_m_t.get_shape():", o_m_t.get_shape()
            # self.o_s_t = tf.matmul(h_t_s_dec, self.W_s) + self.b_s
            o_s_t = tf.matmul(h_t_s_dec, W_s) + b_s + tf.matmul(h_t_s_dec, tf.square(W_m)) + tf.matmul(tf.square(h_t_dec), W_s)

            # self.o_s_t = self.o_s_pos(self.o_s_t)
            # o_s_t = tf.nn.relu(o_s_t)
            o_s_t = tf.square(o_s_t)
            # o_s_t = tf.abs(o_s_t)

            print "o_s_t.get_shape():", o_s_t.get_shape()
            self.dec_output_ = self.sigmoid_gaussian_mean(o_m_t, o_s_t)
            # self.dec_output_ = tf.sigmoid(o_m_t)
            # self.dec_s_output_ = self.sigmoid_gaussian_variance(o_m_t, o_s_t, self.dec_output_)
            self.dec_s_output_ = o_s_t
            # self.dec_s_output_ = tf.sigmoid(o_s_t)
            # self.dec_s_output_ = tf.sqrt(self.dec_s_output_)


            # self.dec_output_ = tf.sigmoid(tf.matmul(h_t_dec, self.W_m) + self.b_m)
            # self.dec_s_output_ = tf.sigmoid(tf.matmul(h_t_s_dec, self.W_s) + self.b_s)


            o_m_t_enc = tf.matmul(h_t_enc, W_m) + b_m
            o_s_t_enc = tf.matmul(h_t_s_enc, W_s) + b_s + tf.matmul(h_t_s_enc, tf.square(W_m)) + tf.matmul(tf.square(h_t_enc), W_s)
            o_s_t_enc = tf.square(o_s_t_enc)
            self.enc_output_ = self.sigmoid_gaussian_mean(o_m_t_enc, o_s_t_enc)
            self.enc_s_output_ = self.sigmoid_gaussian_variance(o_m_t_enc, o_s_t_enc, self.enc_output_)

        self.input_ = tf.transpose(tf.stack(inputs), [1, 0, 2])
        self.input_s_ = tf.transpose(tf.stack(inputs_s), [1, 0, 2])
        self.future_ = tf.transpose(tf.stack(futures), [1, 0, 2])
        self.future_s_ = tf.transpose(tf.stack(futures), [1, 0, 2])

        print "dec_output_.get_shape():", self.dec_output_.get_shape()
        print "dec_s_output_.get_shape():", self.dec_s_output_.get_shape()

        print "future_.get_shape():", self.future_.get_shape()
        print "future_s_.get_shape():", self.future_s_.get_shape()

        # self.input_ = tf.nn.dropout(self.input_, keep_prob=output_keep_prob)

        # self.loss = tf.reduce_mean(tf.square(self.future_ - self.dec_output_), name="loss_m")
        self.loss_s = tf.reduce_mean(tf.square(self.future_s_ - self.dec_s_output_), name="loss_s")
        # self.loss_s = tf.reduce_mean(tf.square(self.input_ - self.dec_s_output_), name="loss_s")

        # self.loss_m1 = tf.reduce_mean(tf.square(self.future_ - self.dec_output_), axis=[0], name="loss_m1")
        # print "self.loss_m1.get_shape():", self.loss_m1.get_shape()
        # loss_mask_2 = tf.transpose(loss_mask, [0], name='loss_mask_2')
        # loss_mask_2 = tf.reshape(tf.tile(tf.expand_dims(loss_mask, 0), [self.batch_num, self.input_dimensions]), [self.batch_num, 10, self.input_dimensions])

        # loss_mask_2 = tf.reshape(tf.tile(tf.expand_dims(loss_mask, 1), [self.batch_num, self.input_dimensions]), [self.batch_num, 10, self.input_dimensions])
        # print "loss_mask_2:", loss_mask_2.get_shape()
        # self.loss = tf.reduce_mean(tf.multiply(loss_mask_2, tf.square(self.future_ - self.dec_output_)), name="loss_m")

        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.future_, logits=self.dec_output_))



        # print "self.loss_m1.get_shape():", self.loss_m1.get_shape()
        # print "loss_mask:", loss_mask.get_shape()
        # # loss_mask_2 = tf.zeros(tf.shape(self.loss_m1), dtype=tf.float32, name='loss_mask_2')
        # loss_mask_2 = tf.transpose(loss_mask, [0], name='loss_mask_2')
        # # loss2 = tf.multiply(loss_mask_2, self.loss_m1)
        # self.loss = tf.matmul(tf.transpose(loss_mask_2), tf.transpose(self.loss_m1), name="loss_m")
        # # print "loss2:", loss2.get_shape()
        #
        # # self.loss = tf.reduce_mean(loss2, axis=[0], name="loss_m")
        # # self.loss = tf.(tf.multiply(loss_mask_2, self.loss_m1), name="loss_m")
        # print "self.loss.get_shape():", self.loss.get_shape()


        # self.loss_ms = tf.reduce_mean(self.loss, self.loss_s, name="loss_ms")

        # self.loss_ms = tf.reduce_sum(tf.divide(1.1, self.dec_s_output_ + 1.0)) + \
        #                tf.reduce_sum(tf.multiply(self.dec_s_output_, tf.subtract(self.dec_output_, self.input_))) + \
        #                tf.reduce_sum(tf.log(self.dec_s_output_ + 1.0))

        starter_learning_rate = 0.001
        # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 2000, 0.9, staircase=False)
        # self.train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.99,
        #                                     epsilon=0.0000001, ).minimize(self.loss, global_step=global_step)
        self.train = tf.train.AdamOptimizer(learning_rate=starter_learning_rate, beta1=0.9, beta2=0.99,
                                            epsilon=0.0000001, ).minimize(self.loss)

    # SIGMOID GAUSSIAN MEAN
    def sigmoid_gaussian_mean(self, o_m, o_s):
        return tf.sigmoid(tf.div(o_m, tf.sqrt(tf.add(1.0, tf.multiply(self.zeta_squared, o_s) ) ) ) )
        # return tf.sigmoid(o_m)

    # SIGMOID GAUSSIAN VARIANCE
    def sigmoid_gaussian_variance(self, o_m, o_s, a_m):
        return tf.subtract( tf.sigmoid( tf.div(tf.multiply(self.alpha_sigmoid, tf.add(o_m, self.beta_sigmoid) ),
                            tf.sqrt( tf.add( 1.0 , tf.multiply( self.zeta_squared_alpha_squared_sigmoid, o_s ) ) ) ) ),
                            tf.square(a_m) )
        # return tf.sigmoid(o_s)

    # TANH GAUSSIAN MEAN
    def tanh_gaussian_mean(self, o_m, o_s):
        return tf.multiply(2.0, tf.sigmoid(tf.div(o_m, tf.sqrt( tf.add(0.25, tf.multiply(self.zeta_squared, o_s) ) ) ) ) ) - 1.0
        # return tf.tanh(o_m)

    # TANH GAUSSIAN VARIANCE
    def tanh_gaussian_variance(self, o_m, o_s, a_m):
        remainder = tf.add( tf.square(a_m), tf.multiply(2.0, a_m) ) + 1.0
        return tf.subtract( tf.multiply(4.0, tf.sigmoid( tf.div(tf.multiply(self.alpha_tanh, tf.add(o_m, self.beta_tanh) ),
                                     tf.sqrt( tf.add( 1.0 , tf.multiply( self.zeta_squared_alpha_squared_tanh, o_s ) ) )) ) ),
                                     remainder )
        # return tf.tanh(o_s)

    def o_s_pos(self, o_s):
        return tf.log(1.0+tf.exp(o_s))

    # VARIANCE APPROXIMATION
    def approx_variance(self, x_m, x_s, s_m, s_s, Um, Us, Wm, Ws, b_m, b_s):
        print "x_m.get_shape():", x_m.get_shape()
        print "x_s.get_shape():", x_s.get_shape()
        print "s_m.get_shape():", s_m.get_shape()
        print "s_s.get_shape():", s_s.get_shape()
        print "Um.get_shape():", Um.get_shape()
        print "Us.get_shape():", Us.get_shape()
        print "Wm.get_shape():", Wm.get_shape()
        print "Ws.get_shape():", Ws.get_shape()
        print "b_s.get_shape():", b_s.get_shape()

        o_s = tf.matmul(x_s, Us) + tf.matmul(s_s, Ws) + b_s + \
               tf.matmul(x_s, tf.square(Um)) + tf.matmul(tf.square(x_m), Us) + \
               tf.matmul(s_s, tf.square(Wm)) + tf.matmul(tf.square(s_m), Ws)

        # o_s = tf.matmul(x_s, Us) + tf.matmul(s_s, Ws) + b_s + \
        #        tf.matmul(x_s, tf.square(Um)) + \
        #        tf.matmul(s_s, tf.square(Wm)) + tf.matmul(tf.square(s_m), Ws)

        # o_s = tf.matmul(x_s, Us) + tf.matmul(s_s, Ws) + b_s + \
        #        tf.matmul(s_s, tf.square(Wm)) + tf.matmul(tf.square(s_m), Ws)

        # o_s = tf.matmul(s_s, Ws) + b_s + tf.matmul(s_s, tf.square(Wm)) + tf.matmul(tf.square(s_m), Ws)

        # o_s = tf.nn.relu(o_s)
        o_s = tf.square(o_s)
        # o_s = self.o_s_pos(o_s)
        return o_s

    # FORWARD PROP - ENCODER
    def forward_pass_enc_np(self, h_0_ms_enc, x_t_ms):

        x_m_t, x_s_t = tf.unstack(x_t_ms, axis=0)
        s_m_t, s_s_t = tf.unstack(h_0_ms_enc, axis=0)

        o_r_m_t = tf.matmul(x_m_t, self.Ur_enc_m) + tf.matmul(s_m_t, self.Wr_enc_m) + self.br_enc_m
        o_r_s_t = self.approx_variance(x_m_t, x_s_t, s_m_t, s_s_t, self.Ur_enc_m, self.Ur_enc_s, self.Wr_enc_m, self.Wr_enc_s, self.br_enc_m, self.br_enc_s)

        r_m_t = self.sigmoid_gaussian_mean(o_r_m_t, o_r_s_t)
        r_s_t = self.sigmoid_gaussian_variance(o_r_m_t, o_r_s_t, r_m_t)

        o_z_m_t = tf.matmul(x_m_t, self.Uz_enc_m) + tf.matmul(s_m_t, self.Wz_enc_m) + self.bz_enc_m
        o_z_s_t = self.approx_variance(x_m_t, x_s_t, s_m_t, s_s_t, self.Uz_enc_m, self.Uz_enc_s, self.Wz_enc_m, self.Wz_enc_s, self.bz_enc_m, self.bz_enc_s)

        z_m_t = self.sigmoid_gaussian_mean(o_z_m_t, o_z_s_t)
        z_s_t = self.sigmoid_gaussian_variance(o_z_m_t, o_z_s_t, z_m_t)

        o_h_m_t = tf.matmul(x_m_t, self.Uh_enc_m) + tf.matmul(tf.multiply(r_m_t, s_m_t), self.Wh_enc_m) + self.bh_enc_m
        o_h_s_t = self.approx_variance(x_m_t, x_s_t, tf.multiply(r_m_t, s_m_t), tf.multiply(r_s_t, s_s_t),
                                            self.Uh_enc_m, self.Uh_enc_s, self.Wh_enc_m, self.Wh_enc_s, self.bh_enc_m, self.bh_enc_s)

        h_m_proposal = self.tanh_gaussian_mean(o_h_m_t, o_h_s_t)
        h_s_proposal = self.tanh_gaussian_variance(o_h_m_t, o_h_s_t, h_m_proposal)

        h_m_t = tf.multiply(1 - z_m_t, h_m_proposal) + tf.multiply(z_m_t, s_m_t)
        h_s_t = tf.multiply(1 - tf.multiply(z_s_t,z_s_t), h_s_proposal) + tf.multiply(tf.multiply(z_s_t,z_s_t), s_s_t)

        return tf.stack([h_m_t, h_s_t], axis=0)

    # FORWARD PROP - ENCODER
    def forward_pass_dec_np(self, h_0_ms_dec, x_t_ms):

        x_m_t, x_s_t = tf.unstack(x_t_ms, axis=0)
        s_m_t, s_s_t = tf.unstack(h_0_ms_dec, axis=0)

        o_r_m_t = tf.matmul(x_m_t, self.Ur_dec_m) + tf.matmul(s_m_t, self.Wr_dec_m) + self.br_dec_m
        o_r_s_t = self.approx_variance(x_m_t, x_s_t, s_m_t, s_s_t, self.Ur_dec_m, self.Ur_dec_s, self.Wr_dec_m, self.Wr_dec_s, self.br_dec_m, self.br_dec_s)

        r_m_t = self.sigmoid_gaussian_mean(o_r_m_t, o_r_s_t)
        r_s_t = self.sigmoid_gaussian_variance(o_r_m_t, o_r_s_t, r_m_t)

        o_z_m_t = tf.matmul(x_m_t, self.Uz_dec_m) + tf.matmul(s_m_t, self.Wz_dec_m) + self.bz_dec_m
        o_z_s_t = self.approx_variance(x_m_t, x_s_t, s_m_t, s_s_t, self.Uz_dec_m, self.Uz_dec_s, self.Wz_dec_m,
                                       self.Wz_dec_s, self.bz_dec_m, self.bz_dec_s)

        z_m_t = self.sigmoid_gaussian_mean(o_z_m_t, o_z_s_t)
        z_s_t = self.sigmoid_gaussian_variance(o_z_m_t, o_z_s_t, z_m_t)

        o_h_m_t = tf.matmul(x_m_t, self.Uh_dec_m) + tf.matmul(tf.multiply(r_m_t, s_m_t), self.Wh_dec_m) + self.bh_dec_m

        o_h_s_t = self.approx_variance(x_m_t, x_s_t, tf.multiply(r_m_t, s_m_t), tf.multiply(r_s_t, s_s_t),
                                           self.Uh_dec_m, self.Uh_dec_s, self.Wh_dec_m, self.Wh_dec_s,
                                           self.bh_dec_m, self.bh_dec_s)

        h_m_proposal = self.sigmoid_gaussian_mean(o_h_m_t, o_h_s_t)
        h_s_proposal = self.sigmoid_gaussian_variance(o_h_m_t, o_h_s_t, h_m_proposal)

        h_m_t = tf.multiply(1 - z_m_t, h_m_proposal) + tf.multiply(z_m_t, s_m_t)
        h_s_t = tf.multiply(1 - tf.multiply(z_s_t,z_s_t), h_s_proposal) + tf.multiply(tf.multiply(z_s_t,z_s_t), s_s_t)

        return tf.stack([h_m_t, h_s_t], axis=0)

    def initialize_encoder(self, hidden_size, mean_m, mean_s, mean_b_m, mean_b_s, w_std, dtype=tf.float32):

        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Ur_enc_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, hidden_size), mean=mean_m, stddev=w_std), name='Ur_enc_m')
        self.Uz_enc_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, hidden_size), mean=mean_m, stddev=w_std), name='Uz_enc_m')
        self.Uh_enc_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, hidden_size), mean=mean_m, stddev=w_std), name='Uh_enc_m')

        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Wr_enc_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size, hidden_size), mean=mean_m, stddev=w_std), name='Wr_enc_m')
        self.Wz_enc_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size, hidden_size), mean=mean_m, stddev=w_std), name='Wz_enc_m')
        self.Wh_enc_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size, hidden_size), mean=mean_m, stddev=w_std), name='Wh_enc_m')

        # Biases for hidden vectors of shape (hidden_size,)
        self.br_enc_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size,), mean=mean_b_m, stddev=w_std), name='br_enc_m')
        self.bz_enc_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size,), mean=mean_b_m, stddev=w_std), name='bz_enc_m')
        self.bh_enc_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size,), mean=mean_b_m, stddev=w_std), name='bh_enc_m')

        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Ur_enc_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, hidden_size), mean=mean_s, stddev=w_std), name='Ur_enc_s')
        self.Uz_enc_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, hidden_size), mean=mean_s, stddev=w_std), name='Uz_enc_s')
        self.Uh_enc_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, hidden_size), mean=mean_s, stddev=w_std), name='Uh_enc_s')

        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Wr_enc_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size, hidden_size), mean=mean_s, stddev=w_std), name='Wr_enc_s')
        self.Wz_enc_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size, hidden_size), mean=mean_s, stddev=w_std), name='Wz_enc_s')
        self.Wh_enc_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size, hidden_size), mean=mean_s, stddev=w_std), name='Wh_enc_s')

        # Biases for hidden vectors of shape (hidden_size,)
        self.br_enc_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size,), mean=mean_b_s, stddev=w_std), name='br_enc_s')
        self.bz_enc_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size,), mean=mean_b_s, stddev=w_std), name='bz_enc_s')
        self.bh_enc_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size,), mean=mean_b_s, stddev=w_std), name='bh_enc_s')


    def initialize_decoder(self, hidden_size, mean_m, mean_s, mean_b_m, mean_b_s, w_std, dtype=tf.float32):

        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Ur_dec_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, hidden_size), mean=mean_m, stddev=w_std), name='Ur_dec_m')
        self.Uz_dec_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, hidden_size), mean=mean_m, stddev=w_std), name='Uz_dec_m')
        self.Uh_dec_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, hidden_size), mean=mean_m, stddev=w_std), name='Uh_dec_m')

        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Wr_dec_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size, hidden_size), mean=mean_m, stddev=w_std), name='Wr_dec_m')
        self.Wz_dec_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size, hidden_size), mean=mean_m, stddev=w_std), name='Wz_dec_m')
        self.Wh_dec_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size, hidden_size), mean=mean_m, stddev=w_std), name='Wh_dec_m')

        # Biases for hidden vectors of shape (hidden_size,)
        self.br_dec_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size,), mean=mean_b_m, stddev=w_std), name='br_dec_m')
        self.bz_dec_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size,), mean=mean_b_m, stddev=w_std), name='bz_dec_m')
        self.bh_dec_m = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size,), mean=mean_b_m, stddev=w_std), name='bh_dec_m')

        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Ur_dec_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, hidden_size), mean=mean_s, stddev=w_std), name='Ur_dec_s')
        self.Uz_dec_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, hidden_size), mean=mean_s, stddev=w_std), name='Uz_dec_s')
        self.Uh_dec_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, hidden_size), mean=mean_s, stddev=w_std), name='Uh_dec_s')

        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Wr_dec_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size, hidden_size), mean=mean_s, stddev=w_std), name='Wr_dec_s')
        self.Wz_dec_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size, hidden_size), mean=mean_s, stddev=w_std), name='Wz_dec_s')
        self.Wh_dec_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size, hidden_size), mean=mean_s, stddev=w_std), name='Wh_dec_s')

        # Biases for hidden vectors of shape (hidden_size,)
        self.br_dec_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size,), mean=mean_b_s, stddev=w_std), name='br_dec_s')
        self.bz_dec_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size,), mean=mean_b_s, stddev=w_std), name='bz_dec_s')
        self.bh_dec_s = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(hidden_size,), mean=mean_b_s, stddev=w_std), name='bh_dec_s')

