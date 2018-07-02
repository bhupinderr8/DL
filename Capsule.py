import tensorflow as tf
import numpy as np

X_data = np.load("X_data.npy").astype(np.float32)
Y_data = np.load("Y_data_not_ont_hot.npy")

X = tf.placeholder(shape=[None, 96, 96, 3], dtype=tf.float32, name="X")
Y = tf.placeholder(shape=[None], dtype=tf.int64, name="Y")


caps1_n_maps = 32
caps1_n_caps = caps1_n_maps*40*40
caps1_n_dims = 8

conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}

conv2_params = {
    "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}

conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)

conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)

caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],
                       name="caps1_raw")



def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector

caps1_output = squash(caps1_raw)

caps2_n_caps = 2
caps2_n_dims = 16



init_sigma = 0.1

W_init = tf.random_normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")



batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                             name="caps1_output_tiled")

caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")

raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1], dtype=np.float32, name="raw_weights")

routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                   name="weighted_predictions")

weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                             name="weighted_sum")

caps2_output_round_1 = squash(weighted_sum, axis=2, name="caps2_output_round_1")



caps2_output_round_1_tiled = tf.tile(caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
                                     name="caps2_output_round_1_tiled")

agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")

raw_weights_round_2 = tf.add(raw_weights, agreement,
                             name="raw_weights_round_2")


routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                        axis=2,
                                        name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                           caps2_predicted,
                                           name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                     axis=1, keep_dims=True,
                                     name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2,
                              axis=-2,
                              name="caps2_output_round_2")


caps2_output = caps2_output_round_2







def condition(input, counter):
    return tf.less(counter, 100)

def loop_body(input, counter):
    out = tf.add(input, tf.square(counter))
    return out, tf.add(counter, 1)

with tf.name_scope("Compute_sum_of_squares"):
    counter = tf.constant(1)
    sum_of_squares = tf.constant(0)

    result = tf.while_loop(condition, loop_body, [sum_of_squares, counter])




def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norms = tf.reduce_sum(tf.square(s), axis=axis,
                                      keep_dims=keep_dims)
        return tf.sqrt(squared_norms + epsilon)

y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")
y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")

y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")

print (y_pred)

# Margin Loss

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

T = tf.one_hot(Y, depth=caps2_n_caps, name="T")

caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_poutput_norm")



present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, 2),
                           name="present_error")



absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, 2),
                          name="absent_error")


L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")

margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

# Accuracy

correct = tf.equal(Y, y_pred, name="correct")

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")


# Training ops

optimizer = tf.train.AdadeltaOptimizer()
training_op = optimizer.minimize(margin_loss, name="training_op")

# init and saver

init = tf.global_variables_initializer()

saver = tf.train.Saver()



# Training

n_epochs = 10
batch_size = 50
restore_checkpoint = True

n_iterations_per_epoch = 6000
n_iterations_validation = 7048 - n_iterations_per_epoch
best_loss_val = np.infty
checkpoint_path = "./Capsule_save"

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    for epoch in range(n_epochs):
        BATCH_SIZE = 10
        for i in range(1, n_iterations_per_epoch + 1):
            X_current = X_data[(i-1)*BATCH_SIZE:i*BATCH_SIZE]
            Y_current = Y_data[(i-1) * BATCH_SIZE:i*BATCH_SIZE]

            _, loss_train = sess.run(
                [training_op, margin_loss],
                feed_dict={
                    X: X_current.reshape([-1, 96, 96, 3]),
                    Y: Y_current})

            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                i, n_iterations_per_epoch,
                i * 100 / n_iterations_per_epoch,
                loss_train),
                end="")