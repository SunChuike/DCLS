import tensorflow as tf
import math

class SearchAdagradDecay():
    def __init__(self, conf):
        self.conf = conf

    def get_optimizer(self, global_step):
        learning_rate = self.get_learning_rate(global_step)
        tf.summary.scalar(name="Optimize/learning_rate", tensor=learning_rate)

        return tf.train.AdagradDecayOptimizerV2(
            learning_rate,
            global_step,
            accumulator_decay_step=self.conf["decay_step"],
            accumulator_decay_rate=self.conf["decay_rate"],
            initial_accumulator_value=self.conf.get("initial_accumulator_value", 0.1)
        )

    def get_learning_rate(self, global_step):
        global_step_float = tf.cast(global_step, dtype=tf.float32)

        if 'lr_func' in self.conf and self.conf['lr_func'] == 'cold_start':

            peak_lr = self.conf["learning_rate"]
            warmup_lr = self.conf['lrcs_init_lr']
            warmup_steps = self.conf['lrcs_init_step']

            decay_steps = self.conf["decay_step"]
            decay_rate = self.conf["decay_rate"]

            warmup_steps_float = tf.cast(warmup_steps, dtype=tf.float32)

            warmup_rate = global_step_float / warmup_steps_float
            learning_rate_warmup = warmup_lr + (peak_lr - warmup_lr) * warmup_rate

            steps_after_warmup = global_step_float - warmup_steps_float
            learning_rate_decay = tf.train.exponential_decay(
                learning_rate=peak_lr,
                global_step=steps_after_warmup,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=self.conf.get("staircase", True)
            )

            is_warmup_phase = global_step_float < warmup_steps_float
            final_learning_rate = tf.where(
                is_warmup_phase,
                learning_rate_warmup,
                learning_rate_decay
            )

            return final_learning_rate

        else:

            initial_lr = self.conf['learning_rate']
            decay_steps = self.conf['decay_step']
            decay_rate = self.conf['decay_rate']

            final_learning_rate = tf.train.exponential_decay(
                learning_rate=initial_lr,
                global_step=global_step,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=self.conf.get("staircase", True)
            )

            return final_learning_rate
