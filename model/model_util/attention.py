#!/usr/bin/env python
#-*- coding:utf8 -*-

import tensorflow as tf
from tensorflow.contrib import layers


def multihead_attention(queries,
                        keys,
                        values,
                        num_units=None,
                        num_output_units=None,
                        activation_fn=None,
                        num_heads=8,
                        scope="multihead_attention",
                        atten_mode="base",
                        reuse=None,
                        query_masks=None,
                        key_masks=None,
                        variables_collections=None,
                        outputs_collections=None,
                        residual_connection=False,
                        attention_normalize=False,
                        use_atten_linear_project=True
                        ):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      queries_length: A 1d tensor with shape of [N].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      keys_length:  A 1d tensor with shape of [N].
      num_units: A scalar. Attention size.
      num_output_units: A scalar. Output Value size.
      keep_prob: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
      query_masks: A mask to mask queries with the shape of [N, T_k], if query_masks is None, use queries_length to mask queries
      key_masks: A mask to mask keys with the shape of [N, T_Q],  if key_masks is None, use keys_length to mask keys

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        if num_output_units is None or residual_connection:
            num_output_units = queries.get_shape().as_list()[-1]

        if atten_mode == 'cos' or atten_mode == 'ln':
            activation_fn = None

        # Linear projections, C = # dim or column, T_x = # vectors or actions
        if use_atten_linear_project:
            Q = layers.fully_connected(queries,
                                       num_units,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="Q")  # (N, T_q, C)
            K = layers.fully_connected(keys,
                                       num_units,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="K")  # (N, T_k, C)
            if values is None:
                V = layers.fully_connected(keys,
                                           num_output_units,
                                           variables_collections=variables_collections,
                                           outputs_collections=outputs_collections, scope="V")  # (N, T_k, C)
            else:
                V = layers.fully_connected(values,
                                           num_output_units,
                                           variables_collections=variables_collections,
                                           outputs_collections=outputs_collections, scope="V")  # (N, T_k, C)
        else:
            Q = queries
            K = keys
            if values is None:
                V = keys
            else:
                V = values


        def split_last_dimension_then_transpose(tensor, num_heads):
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, t_shape[-1] // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, t_shape[-1]]

        Q_ = split_last_dimension_then_transpose(Q, num_heads)  # (h*N, T_q, C/h)
        K_ = split_last_dimension_then_transpose(K, num_heads)  # (h*N, T_k, C/h)
        V_ = split_last_dimension_then_transpose(V, num_heads)  # (h*N, T_k, C'/h)

        if atten_mode == 'cos':
            Q_cos = tf.nn.l2_normalize(Q_, dim=-1)
            K_cos = tf.nn.l2_normalize(K_, dim=-1)

            # Multiplication
            # query-key score matrix
            # each big score matrix is then split into h score matrix with same size
            # w.r.t. different part of the feature
            outputs = tf.matmul(Q_cos, K_cos, transpose_b=True)  # (h*N, T_q, T_k)
            # [batch_size, num_heads, query_len, key_len]

            # Scale
            outputs = outputs * 20
        elif atten_mode == 'ln':
            # Multiplication
            # query-key score matrix
            # each big score matrix is then split into h score matrix with same size
            # w.r.t. different part of the feature
            Q_ = layers.layer_norm(Q_, begin_norm_axis=-1, begin_params_axis=-1)
            K_ = layers.layer_norm(K_, begin_norm_axis=-1, begin_params_axis=-1)
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # [batch_size, num_heads, query_len, key_len]

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        else:
            # Multiplication
            # query-key score matrix
            # each big score matrix is then split into h score matrix with same size
            # w.r.t. different part of the feature
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # [batch_size, num_heads, query_len, key_len]

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        query_len = queries.get_shape().as_list()[1]
        key_len = keys.get_shape().as_list()[1]

        key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, 1, key_len]),
                            [1, num_heads, query_len, 1])
        paddings = tf.fill(tf.shape(outputs), tf.constant(-1e9, dtype=tf.float32))
        outputs = tf.where(key_masks, outputs, paddings)

        # Causality = Future blinding: No use, removed

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
        if query_masks is not None:
            query_masks = tf.tile(tf.reshape(query_masks, [-1, 1, query_len, 1]),
                                  [1, num_heads, 1, key_len])
            paddings = tf.fill(tf.shape(outputs), tf.constant(0, dtype=tf.float32))
            outputs = tf.where(query_masks, outputs, paddings)

        # Attention vector
        att_vec = outputs

        # Dropouts
        # outputs = layers.dropout(outputs, keep_prob=keep_prob, is_training=is_training)

        # Weighted sum (h*N, T_q, T_k) * (h*N, T_k, C/h)
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        def transpose_then_concat_last_two_dimenstion(tensor):
            tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

        outputs = transpose_then_concat_last_two_dimenstion(outputs)  # (N, T_q, C)

        # Residual connection
        if residual_connection:
            tf.logging.info('[Attention] using residual connection.')
            outputs += queries

        # Normalize
        if attention_normalize:
            tf.logging.info('[Attention] using layer normalize.')
            outputs = layers.layer_norm(outputs, begin_norm_axis=-1, begin_params_axis=-1)  # (N, T_q, C)

    return outputs, att_vec


def feedforward(inputs,
                num_units=[2048, 512],
                activation_fn=tf.nn.relu,
                scope="feedforward",
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                residual_connection=False):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if residual_connection:
            num_units = [inputs.get_shape().as_list()[-1] * 4, inputs.get_shape().as_list()[-1]]
        outputs = layers.fully_connected(inputs,
                                         num_units[0],
                                         activation_fn=activation_fn,
                                         variables_collections=variables_collections,
                                         outputs_collections=outputs_collections)
        outputs = layers.fully_connected(outputs,
                                         num_units[1],
                                         activation_fn=None,
                                         variables_collections=variables_collections,
                                         outputs_collections=outputs_collections)
        outputs += inputs
        outputs = layers.layer_norm(outputs, begin_norm_axis=-1, begin_params_axis=-1)

    return outputs


def attention(queries,
              keys,
              values,
              num_units,
              num_output_units,
              reuse,
              scope,
              atten_mode,
              variables_collections,
              outputs_collections,
              activation_fn=None,
              query_masks=None,
              key_masks=None,
              num_heads=8,
              residual_connection=False,
              attention_normalize=False,
              use_atten_linear_project=True):

    s_item_vec, stt_vec = multihead_attention(queries=queries,
                                              keys=keys,
                                              values=values,
                                              num_units=num_units,
                                              num_output_units=num_output_units,
                                              activation_fn=activation_fn,
                                              scope=scope,
                                              atten_mode=atten_mode,
                                              reuse=reuse,
                                              query_masks=query_masks,
                                              key_masks=key_masks,
                                              variables_collections=variables_collections,
                                              outputs_collections=outputs_collections,
                                              num_heads=num_heads,
                                              residual_connection=residual_connection,
                                              attention_normalize=attention_normalize,
                                              use_atten_linear_project=use_atten_linear_project)

    item_vec = feedforward(s_item_vec,
                           num_units=[num_output_units * 4, num_output_units],
                           scope=scope + "_feed_forward",
                           reuse=reuse,
                           variables_collections=variables_collections,
                           outputs_collections=outputs_collections,
                           residual_connection=residual_connection)


    return item_vec, stt_vec

def l2_norm(vector, alpha=1):
    return tf.nn.l2_normalize(vector, -1)

def identity(vector, alpha=1):
    return vector

def squash(vector, alpha=1, epsilon=1e-9):
    '''Squashing function corresponding to Eq. 1
    Args:
      vector: A tensor with shape [N, T, num_caps, dim_caps].
    Returns:
      A tensor with the same shape as vector but squashed in last dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -1, True)
    scalar_factor = (vec_squared_norm / (alpha + vec_squared_norm)
                     / tf.sqrt(vec_squared_norm + epsilon))
    vec_squashed = scalar_factor * vector  # element-wise
    return vec_squashed

routing_activations = {
    "squash": squash,
    "l2_norm": l2_norm,
    "identity": identity
}

def get_shape(inputs):
    '''Get shapes, return static shapes as more as possible.
    '''
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape

def dynamic_routing(inputs,
                    interest_num,
                    interest_dim,
                    inputs_length,
                    stddev_b=1,
                    routing_iter=3,
                    linear_transform=True,
                    share_interert_weight=True,
                    random_init=True,
                    l2_normalize=False,
                    inner_activation='l2_norm',
                    last_activation='squash',
                    alpha=1,
                    enlarge_factor=1,
                    init_b_enlarge=False,
                    scope='dynamic_routing',
                    variables_collections=None,
                    outputs_collections=None,
                    reuse=None
                    ):
    with tf.variable_scope(scope, reuse=reuse):
        batch, seq_len, dim = get_shape(inputs)
        if linear_transform:
            if share_interert_weight:
                # [batch, seq_len, dim]
                inputs = layers.fully_connected(inputs,
                                                interest_dim,
                                                activation_fn=None,
                                                normalizer_fn=None,
                                                normalizer_params=None,
                                                variables_collections=variables_collections,
                                                outputs_collections=outputs_collections,
                                                biases_initializer=None)
                # [batch, seq_len, 1, dim]
                inputs = tf.expand_dims(inputs, 2)
                # [batch, seq_len, interest_num, dim]
                inputs = tf.tile(inputs, [1, 1, interest_num, 1])
                inputs_stopped = tf.stop_gradient(inputs, name='inputs_stopped')
            else:
                # [batch, seq_len, dim*interest_num]
                inputs = layers.fully_connected(inputs,
                                                interest_dim*interest_num,
                                                activation_fn=None,
                                                normalizer_fn=None,
                                                normalizer_params=None,
                                                variables_collections=variables_collections,
                                                outputs_collections=outputs_collections,
                                                biases_initializer=None)
                # [batch, seq_len, interest_num, dim]
                inputs = tf.reshape(inputs, [-1, seq_len, interest_num, interest_dim])
                inputs_stopped = tf.stop_gradient(inputs, name='inputs_stopped')
        else:
            inputs = tf.expand_dims(inputs, 2)
            inputs = tf.tile(inputs, [1, 1, interest_num, 1])
            inputs_stopped = tf.stop_gradient(inputs, name='inputs_stopped')

        inputs_mask = tf.sequence_mask(inputs_length, seq_len, tf.float32)
        inputs_mask = tf.reshape(inputs_mask, [batch, seq_len, 1, 1])
        if random_init or share_interert_weight:
            b_ij = tf.truncated_normal([batch, seq_len, interest_num, 1], stddev=stddev_b)
        else:
            mask = tf.reshape(inputs_mask, [-1])
            inputs_stopped_2d = tf.reshape(inputs_stopped, [-1, interest_num*interest_dim])
            inputs_stopped_4d = tf.reshape(tf.where(tf.greater(mask, 1), inputs_stopped_2d, tf.zeros_like(inputs_stopped_2d)),
                                           [-1, seq_len, interest_num, interest_dim])
            s_j = tf.reduce_sum(inputs_stopped_4d, 1, True) / interest_num
            v_j = l2_norm(s_j)
            b_ij = tf.reduce_sum(tf.multiply(inputs_stopped, v_j), -1, True)
        if init_b_enlarge:
            b_ij = b_ij * enlarge_factor
        b_ij = tf.stop_gradient(b_ij)

        for i in range(routing_iter):
            with tf.variable_scope('iter_' + str(i)):
                c_ij = tf.nn.softmax(b_ij, 2)
                c_ij = c_ij * inputs_mask

                if i < routing_iter - 1:
                    s_j = tf.multiply(c_ij, inputs_stopped)
                    s_j = tf.reduce_sum(s_j, 1, True)
                    v_j = routing_activations[inner_activation](s_j, alpha)

                    if l2_normalize:
                        b_ij = tf.reduce_sum(tf.multiply(l2_norm(inputs_stopped), v_j), -1, True)
                    else:
                        b_ij = tf.reduce_sum(tf.multiply(inputs_stopped, v_j), -1, True)
                    b_ij = b_ij * enlarge_factor
                else:
                    s_j = tf.multiply(c_ij, inputs)
                    s_j = tf.reduce_sum(s_j, 1, True)
                    v_j = routing_activations[last_activation](s_j, alpha)

        interest_vec = tf.squeeze(v_j, axis=1)
        raw_interest_vec = tf.squeeze(s_j, axis=1)
        attention = tf.transpose(tf.squeeze(c_ij, axis=-1), [0, 2, 1])
        raw_attention = tf.transpose(tf.squeeze(b_ij, axis=-1), [0, 2, 1])
        return interest_vec, raw_interest_vec, attention, raw_attention, inputs



