import tensorflow.keras.backend as K
import tensorflow as tf


def get_cosine_decay_with_linear_warmup(
        total_epochs: int,
        current_epoch=0,
        learning_rate_start=0.0,
        learning_rate_max=.08,
        warmup_percent=0.05,
        alpha=0.001,
        verbose=False):
    """
    Returns a tf.keras.callbacks.LearningRateScheduler in which the learning rate
    grows linearly from learning_rate_start to learning_rate_max for warmup_percent
    of total_steps, then decreases down to minumum_learning_rate by cosine decay
    rule.

    The returned schedule can be applied as a callback in model.fit().

    Args:
    total_steps: The number of total steps
    current_step: The number of the current step
    learning_rate_start: The learning rate to start at
    total_steps: Total number of training steps
    learning_rate_max: The max learning rate after which lr starts decreasing
    warmup_percent: Percent of total steps to increase linearly
    alpha: Minimum learning rate as a fraction of learning_rate_max
    verbose: Controls logging.
    """
    if learning_rate_start > learning_rate_max:
        raise ValueError("learning_rate_start must be < learning_rate_max")

    learning_rate_start = learning_rate_start
    verbose = verbose

    # After this step, switch from increasing linearly to cosine decay
    switch_epoch = warmup_percent * total_epochs

    # Increase the lr linearly by this much
    linear_slope = (learning_rate_max - learning_rate_start) / switch_epoch

    cosine_decay = tf.keras.experimental.CosineDecay(
        initial_learning_rate=learning_rate_max,
        decay_steps=total_epochs - switch_epoch,
        alpha=alpha,
        name="cosine_decay_lr")

    def scheduler(epoch_index, lr):
        # The epoch number starts at 0
        epoch_number = epoch_index + 1

        linearly_increased = learning_rate_start + \
            (linear_slope * epoch_number)

        cosine_decayed = cosine_decay(epoch_number - switch_epoch)

        if epoch_number > switch_epoch:
            lr = K.get_value(cosine_decayed)
        else:
            lr = linearly_increased

        if verbose:
            print(" Learning rate set to", lr, "for epoch", epoch_number)

        return lr

    return tf.keras.callbacks.LearningRateScheduler(scheduler)
