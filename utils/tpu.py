import tensorflow as tf

# The documentation on thsi takes some digigng to find, so here it is.
# See:https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/TPUStrategy
# Example: https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/keras_mnist_tpu.ipynb
# Example2: https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/fashion_mnist.ipynb


def get_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        coordinator_name='host'
    )

    tf.config.experimental_connect_to_cluster(resolver)

    tf.tpu.experimental.initialize_tpu_system(resolver)

    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    return strategy
