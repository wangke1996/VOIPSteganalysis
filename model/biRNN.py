import tensorflow as tf
from tensorflow.contrib import cudnn_rnn


class biLSTM(object):
    def __init__(self, max_input_length, num_class, input_dim=3, hidden_layer_num=2, num_hidden=50, fc_num_hidden=64,
                 bi_direction=True, hidden_layer_num_bi=2, num_hidden_bi=50, use_attention=True, attention_size=20,
                 dropout=0):
        self.bi_direction = bi_direction
        if self.bi_direction:
            self.num_hidden = num_hidden_bi
            self.hidden_layer_num = hidden_layer_num_bi
            self.direction = 'bidirectional'
        else:
            self.num_hidden = num_hidden
            self.hidden_layer_num = hidden_layer_num
            self.direction = 'unidirectional'
        self.fc_num_hidden = fc_num_hidden
        self.x = tf.placeholder(tf.float32, [None, max_input_length, input_dim])
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.shape(self.x)[0]

        with tf.variable_scope("rnn"):
            def add_cudnn_lstm_layer(input):
                """Adds CUDNN LSTM layers."""
                # Convolutions output [B, L, Ch], while CudnnLSTM is time-major.
                trans_perm = [1, 0] + [x + 2 for x in range(len(input.shape) - 2)]
                time_major_input = tf.transpose(input, trans_perm)
                lstm = cudnn_rnn.CudnnLSTM(num_layers=self.hidden_layer_num, num_units=self.num_hidden,
                                           direction=self.direction, dropout=dropout,
                                           kernel_initializer=tf.random_uniform_initializer(maxval=0.1))
                outputs, _ = lstm(time_major_input)

                # Convert back from time-major outputs to batch-major outputs.
                outputs = tf.transpose(outputs, trans_perm)
                return outputs

            rnn_outputs = add_cudnn_lstm_layer(self.x)
            if self.bi_direction:
                rnn_outputs = tf.concat(rnn_outputs, axis=2)

        with tf.variable_scope("attention"):
            def add_attention_layer(input):
                hidden_size = input.shape[2].value
                # Trainable parameters
                w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
                b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
                u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
                # w_omega = tf.Variable(tf.random_uniform([hidden_size, attention_size], maxval=0.1))
                # b_omega = tf.Variable(tf.random_uniform([attention_size], maxval=0.1))
                # u_omega = tf.Variable(tf.random_uniform([attention_size], maxval=0.1))

                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
                v = tf.tanh(tf.tensordot(input, w_omega, axes=1) + b_omega)

                # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
                vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
                alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

                #  Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
                output = tf.reduce_sum(input * tf.expand_dims(alphas, -1), 1)
                return output, alphas

            if use_attention:
                attention_output, _ = add_attention_layer(rnn_outputs)
                dropout = tf.nn.dropout(attention_output, self.keep_prob)
            else:
                dropout = tf.nn.dropout(tf.reduce_sum(rnn_outputs, 1) / rnn_outputs.shape[1].value, self.keep_prob)

        # with tf.name_scope("fc"):
        #     fc_output = tf.layers.dense(rnn_output_flat, self.fc_num_hidden, activation=tf.nn.relu)
        #     dropout = tf.nn.dropout(fc_output, self.keep_prob)
        #     self.fc_output = fc_output
        with tf.name_scope("output"):
            self.logits = tf.layers.dense(dropout, num_class, activation=tf.sigmoid)
            # self.predictions = tf.to_int32(tf.nn.softmax(self.logits))
            # self.logits = tf.layers.dense(rnn_output_flat, num_class, activation=tf.nn.relu)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
