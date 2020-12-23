import tensorflow as tf
import wandb

default_config = {
    'learning_rate' : 0.01
}

wandb.init(config = default_config)
learning_rate = wandb.config.learning_rate

msg = tf.constant("hello Tensor Flow!")

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())
v1 = tf.constant([1, 3, 3, 4])
v2 = tf.constant([4, 3, 2, 1])
ret = tf.add(v1, v2)

print(sess.run(ret))

wandb.log({'c':ret})
#wandb.tensorflow.log(tf.summary.merge_all())
