#!/usr/bin/env python
# -*- coding:utf8 -*-
import sys
import os

cur_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.abspath(os.path.join(cur_path, '..')))
sys.path.append(os.path.abspath(os.path.join(cur_path, '../..')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util
from tensorflow.contrib.layers.python.layers.feature_column_ops import _input_from_feature_columns
from tensorflow.contrib.layers.python.layers.feature_column import _EmbeddingColumn, _RealValuedColumn

import global_var as gl
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from prada_model_ops.metrics import auc
from tensorflow.python.ops import metrics
from prada_interface.algorithm import Algorithm
from model_util.fg import FgParser
from model_util.util import *
from model_util.attention import attention as atten_func
from model_util.attention import squash
from model_util.attention import dynamic_routing
from model_util.my_graph_learn import MyGraphLearn
from optimizer.adagrad_decay import SearchAdagradDecay
from optimizer import optimizer_ops as myopt
from tensorflow.python.framework.errors_impl import OutOfRangeError, ResourceExhaustedError
from requests.exceptions import ConnectionError
from tensorflow.contrib import layers


import numpy as np
from model_util import odps_io as myodps

optimizer_dict = {
    "AdagradDecay": lambda opt_conf, global_step: SearchAdagradDecay(opt_conf).get_optimizer(global_step),
}

class ActiveNet(Algorithm):

    def variable_scope(self, *args, **kwargs):
        kwargs['partitioner'] = partitioned_variables.min_max_variable_partitioner(
            max_partitions=self.config.get_job_config("ps_num"),
            min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
        kwargs['reuse'] = tf.AUTO_REUSE
        return tf.variable_scope(*args, **kwargs)

    def init(self, context):
        self.context = context
        self.logger = self.context.get_logger()
        self.config = self.context.get_config()

        gl._init()
        gl.set_value('logger', self.logger)

        for (k, v) in self.config.get_all_algo_config().items():
            self.model_name = k
            self.algo_config = v
            self.opts_conf = v['optimizer']
            self.model_conf = v['modelx']
            self.metric_conf = v['metrics']

        self.parse_param()
        self._my_graph_learn = MyGraphLearn(self.context, self.config)
        if not self._my_graph_learn.init_graph_learn_server():
            logger.error("Init graph learn server failed")
            return False
        logger.info("Run graph learn init suc")

        self.graph = self._my_graph_learn.get_graph()

        self.neg_sampler = self.graph.negative_sampler(
            'i',
            expand_factor=self.neg_num,
            strategy='node_weight',
            conditional=False,
            unique=False,
            batch_share=False
        )

        if self.model_name is None:
            self.model_name = "ActiveNet"

        self.user_column_blocks = []

        if self.algo_config.get('user_columns') is not None:
            arr_blocks = self.algo_config.get('user_columns').split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.user_column_blocks.append(block)

        self.seq_column_blocks = []
        self.seq_column_len = {}

        if self.algo_config.get('seq_column_blocks') is not None:
            arr_blocks = self.algo_config.get('seq_column_blocks').split(';', -1)
            for block in arr_blocks:
                if block == "":
                    continue
                arr = block.split(':', -1)
                if len(arr[0]) > 0:
                    self.seq_column_blocks.append(arr[0])
                if len(arr[1]) > 0:
                    self.seq_column_len[arr[0]] = arr[1]

        self.sequence_to_process = self.seq_column_blocks

        self.logger.info('sequence column blocks: {}'.format(self.seq_column_blocks))
        self.logger.info('sequence_to_process: {}'.format(self.sequence_to_process))

        # Define model variables collection
        self.self_atten_collections_dnn_hidden_layer = "{}_self_atten_dnn_hidden_layer".format(self.model_name)
        self.self_atten_collections_dnn_hidden_output = "{}_self_atten_dnn_hidden_output".format(self.model_name)
        self.mlp_collections_dnn_hidden_layer = "{}_mlp_dnn_hidden_layer".format(self.model_name)
        self.mlp_collections_dnn_hidden_output = "{}_mlp_dnn_hidden_output".format(self.model_name)
        self.potential_emb_collections_dnn_hidden_layer = "{}_potential_emb_dnn_hidden_layer".format(self.model_name)
        self.potential_emb_collections_dnn_hidden_output = "{}_potential_emb_dnn_hidden_output".format(self.model_name)
        self.prediction_mlp_collections_dnn_hidden_layer = "{}_prediction_mlp_dnn_hidden_layer".format(self.model_name)
        self.prediction_mlp_collections_dnn_hidden_output = "{}_prediction_mlp_dnn_hidden_output".format(self.model_name)
        self.prediction_head_collections_dnn_hidden_layer = "{}_prediction_head_dnn_hidden_layer".format(self.model_name)
        self.prediction_head_collections_dnn_hidden_output = "{}_prediction_head_dnn_hidden_output".format(self.model_name)
        self.user_net_collections_dnn_hidden_layer = "{}_user_net_dnn_hidden_layer".format(self.model_name)
        self.user_net_collections_dnn_hidden_output = "{}_user_net_dnn_hidden_output".format(self.model_name)
        self.item_net_collections_dnn_hidden_layer = "{}_item_net_dnn_hidden_layer".format(self.model_name)
        self.item_net_collections_dnn_hidden_output = "{}_item_net_dnn_hidden_output".format(self.model_name)

        self.layer_dict = {}
        self.sequence_layer_dict = {}

        self.metrics = {}
        self.sink = context.get_sink()
        self.fg = FgParser(self.config.get_fg_config())
        self.debug_tensor_collector = {}

        try:
            self.is_training = tf.get_default_graph().get_tensor_by_name("training:0")
        except KeyError:
            self.is_training = tf.placeholder(tf.bool, name="training")

    def parse_param(self):
        self.neg_num = self.model_conf['model_hyperparameter'].get('neg_num', 20)
        self.vocab_size = self.model_conf['model_hyperparameter'].get('vocab_size', 64)
        self.interest_nums = self.model_conf['model_hyperparameter'].get('interest_nums', 8)
        self.interest_dim = self.model_conf['model_hyperparameter'].get('interest_dim', 128)
        self.dynamic_routing_alpha = self.model_conf['model_hyperparameter'].get('dynamic_routing_alpha', 1)
        self.dynamic_routing_enlarge_factor = self.model_conf['model_hyperparameter'].get('dynamic_routing_enlarge_factor', 10)
        self.dynamic_routing_l2_normalize = self.model_conf['model_hyperparameter'].get('dynamic_routing_l2_normalize',False)
        self.long_seq_nums = self.model_conf['model_hyperparameter'].get('long_seq_nums', 8)
        self.long_seq_max_len = self.model_conf['model_hyperparameter'].get('long_seq_max_len', 100)
        self.dnn_potential_emb_space_output_units = self.model_conf['model_hyperparameter'].get('dnn_potential_emb_space_output_units', [128])
        self.temperature = self.model_conf['model_hyperparameter'].get('temperature', 0.02)
        self.dnn_l2_reg = self.model_conf['model_hyperparameter'].get('dnn_l2_reg', 0.0)
        self.activation = self.model_conf['model_hyperparameter'].get('activation', 'lrelu')
        self.use_user_context = self.model_conf['model_hyperparameter'].get('use_user_context', True)

    def build_graph(self, context, features, feature_columns, labels):

        self.features = features[self.model_name]
        self.feature_columns = feature_columns[self.model_name]
        self.logger.info("[LogAllFeature] %s" % self.features)
        self.logger.info("[LogAllFeatureColumn] %s" % self.feature_columns)

        self.set_global_step()
        self.inference(self.features, self.feature_columns)
        self.loss()
        self.optimizer()
        self.mark_output()
        self.summary()
        self.set_reset_op()

    def set_reset_op(self):
        self.reset_ops, self.localvar = self.reset_variables(collection_key=tf.GraphKeys.LOCAL_VARIABLES,
                                                             matchname='Metrics/local')

    def reset_variables(self, collection_key=tf.GraphKeys.LOCAL_VARIABLES, matchname='auc/', not_match=None):
        localv = tf.get_collection(collection_key)
        self.logger.info("##### local variables: {}".format(localv))
        localv = [x for x in localv if matchname in x.name]
        if not_match is not None:
            localv = [x for x in localv if not_match not in x.name]
        self.logger.info("##### match local variables: {}".format(localv))
        retvops = [tf.assign(x, array_ops.zeros(shape=x.get_shape(), dtype=x.dtype)) for x in localv]
        if len(retvops) == 0:
            return None, None
        retvops = tf.tuple(retvops)
        return retvops, localv

    def set_global_step(self):
        """Sets up the global step Tensor."""
        self.global_step = training_util.get_or_create_global_step()
        self.global_step_reset = tf.assign(self.global_step, 0)
        self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
        tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)

    def inference(self, features, feature_columns):
        self.embedding_layer(features, feature_columns)
        self.sequence_layer()
        self.user_net()
        self.item_net()
        self.logits_layer()

    def add_sample_trace_dict(self, key, value):
        try:
            self.sample_trace_dict[key] = tf.sparse_tensor_to_dense(value, default_value="")
        except:
            self.sample_trace_dict[key] = value

    def get_neg_attrs(self, ids):
        def _get_neg_attrs_func(ids):
            neg_nodes = self.neg_sampler.get(ids=ids)
            # [batch, neg_num, count(string)]
            return neg_nodes.string_attrs

        return tf.py_func(func=_get_neg_attrs_func,
                          inp=[ids], Tout=[tf.string])

    def build_sequence(self, seq_column_blocks, seq_column_len, scope):
        features = self.features
        feature_columns = self.feature_columns
        sequence_layer_dict = {}
        if seq_column_blocks is None or len(seq_column_blocks) == 0:
            return

        if len(seq_column_blocks) > 0:
            for block_name in seq_column_blocks:
                logger.info("Debug seq_column_bolcks: {}".format(block_name))
                if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                    raise ValueError('block_name:(%s) not in feature_columns for seq' % block_name)
                seq_len = self.fg.get_seq_len_by_sequence_name(block_name)
                self.logger.info("seq_len: {}".format(seq_len))
                sequence_stack = _input_from_feature_columns(features,
                                                             feature_columns[block_name],
                                                             weight_collections=None,
                                                             trainable=True,
                                                             scope=scope,
                                                             output_rank=3,
                                                             default_name='sequence_input_from_feature_columns')
                sequence_stack = tf.reshape(sequence_stack, [-1, seq_len, sequence_stack.get_shape()[(-1)].value])
                sequence_2d = tf.reshape(sequence_stack, [-1, tf.shape(sequence_stack)[2]])

                if block_name in seq_column_len and seq_column_len[block_name] in self.layer_dict:
                    sequence_length = self.layer_dict[seq_column_len[block_name]]
                    sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), seq_len)
                    self.logger.info("sequence_mask.type: {}".format(sequence_mask.dtype))

                    sequence_stack = tf.reshape(
                        tf.where(tf.reshape(sequence_mask, [-1]), sequence_2d, tf.zeros_like(sequence_2d)),
                        tf.shape(sequence_stack))
                else:
                    sequence_stack = tf.reshape(sequence_2d, tf.shape(sequence_stack))
                sequence_layer_dict[block_name] = sequence_stack

        return sequence_layer_dict

    def embedding_layer(self, features, feature_columns):
        with self.variable_scope("Embedding_Layer") as scope:

            for block_name in (self.seq_column_len.values()):
                if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                    raise ValueError("block_name:(%s) not in feature_columns for embed" % block_name)
                self.logger.info("block_name:%s, len(feature_columns[block_name])=%d" %
                                 (block_name, len(feature_columns[block_name])))

                self.layer_dict[block_name] = layers.input_from_feature_columns(features,
                                                                                feature_columns=feature_columns[
                                                                                    block_name],
                                                                                scope=scope)

            self.tmsid_label = layers.input_from_feature_columns(features, feature_columns['tmsid_64_1'], scope=scope)
            self.long_clk_seq_tmsid = layers.input_from_feature_columns(features, feature_columns['long_clk_seq_list_long_clk_seq_tmsid_64_1'], scope=scope)
            self.long_sids = tf.cast(self.long_clk_seq_tmsid, tf.int32)  # [B, L]

            self.logger.info("##### tmsid_label: {}".format(self.tmsid_label)) # (?, 1)
            self.logger.info("##### long_clk_seq_tmsid: {}".format(self.long_clk_seq_tmsid)) # (?, 4000)

            self.sequence_layer_dict = self.build_sequence(self.seq_column_blocks, self.seq_column_len, scope)

            self.long_embs = self.sequence_layer_dict['long_clk_seq_list']  # [B, L, d]
            self.logger.info("##### long_embs: {}".format(self.long_embs))  # (?, 4000, 120)

            self.pos_emb = layers.input_from_feature_columns(features, feature_columns['target_columns'], scope=scope)
            self.pos_item_ids_str = tf.squeeze(tf.sparse_tensor_to_dense(features['emb_item_id'], default_value='0'))
            self.pos_item_ids = tf.string_to_number(self.pos_item_ids_str, out_type=tf.int64)

            self.logger.info("##### pos_emb: {}".format(self.pos_emb))

            # neg
            attr_names = self.config.get_job_config('attr_names')
            self.neg_features = {}
            # [batch, neg_num, count(string)]
            self.neg_attrs = self.get_neg_attrs(self.pos_item_ids)
            # [batch*neg_num, count(string)]
            self.neg_attrs = tf.reshape(self.neg_attrs, [-1, len(attr_names)])
            for i in range(len(attr_names)):
                self.neg_features[attr_names[i]] = self.neg_attrs[:, i]
            self.logger.info("##### neg_attrs: {}".format(self.neg_attrs))
            self.logger.info("##### neg_features: {}".format(self.neg_features))

            # [batch*neg_num, dim]
            self.neg_emb = layers.input_from_feature_columns(self.neg_features, feature_columns['target_columns'],scope=scope)
            # [batch*neg_num]
            self.neg_item_ids_str = self.neg_features['emb_item_id']
            self.neg_item_ids = tf.string_to_number(self.neg_item_ids_str, out_type=tf.int64)
            self.logger.info("##### neg_emb: {}".format(self.neg_emb))
            ##### neg_emb: Tensor("Embedding_Layer/Embedding_Layer_1/concat:0", shape=(?, 96), dtype=float32, device=/job:worker/task:1)
            self.logger.info("##### neg_item_ids: {}".format(self.neg_item_ids))
            ##### neg_item_ids: Tensor("Embedding_Layer/StringToNumber_1:0", shape=(?,), dtype=int64, device=/job:worker/task:1)

    def dnn_potential_emb_space(self, dnn_input, dnn_hidden_units, name):
        if isinstance(dnn_hidden_units, int):
            dnn_hidden_units = [dnn_hidden_units]

        with arg_scope(model_arg_scope(weight_decay=self.dnn_l2_reg)):
            for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                with self.variable_scope(name_or_scope="{}_potential_hidden_layer_{}".format(name, layer_id)) as dnn_hidden_layer_scope:
                    dnn_input = layers.fully_connected(
                        dnn_input,
                        num_hidden_units,
                        getActivationFunctionOp(self.activation),
                        scope=dnn_hidden_layer_scope,
                        normalizer_fn=layers.layer_norm,
                        normalizer_params={
                            "begin_norm_axis": -1,
                            "begin_params_axis": -1
                        },
                        variables_collections=[self.potential_emb_collections_dnn_hidden_layer],
                        outputs_collections=[self.potential_emb_collections_dnn_hidden_output]
                    )
        return dnn_input

    def mlp(self, dnn_input, dnn_hidden_units, name):

        if isinstance(dnn_hidden_units, int):
            dnn_hidden_units = [dnn_hidden_units]

        with arg_scope(model_arg_scope(weight_decay=self.dnn_l2_reg)):
            for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                with self.variable_scope(
                        name_or_scope="{}_mlp_hidden_layer_{}".format(name, layer_id)) as dnn_hidden_layer_scope:
                    dnn_input = layers.fully_connected(
                        dnn_input,
                        num_hidden_units,
                        getActivationFunctionOp(self.activation),
                        scope=dnn_hidden_layer_scope,
                        normalizer_fn=layers.layer_norm,
                        normalizer_params={
                            "begin_norm_axis": -1,
                            "begin_params_axis": -1
                        },
                        variables_collections=[self.mlp_collections_dnn_hidden_layer],
                        outputs_collections=[self.mlp_collections_dnn_hidden_output]
                    )
        return dnn_input

    def prediction_head(self, dnn_input, dnn_hidden_units, name):

        if isinstance(dnn_hidden_units, int):
            dnn_hidden_units = [dnn_hidden_units]

        with arg_scope(model_arg_scope(weight_decay=self.dnn_l2_reg)):
            for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                with self.variable_scope(name_or_scope="{}_prediction_head_hidden_layer_{}".format(name, layer_id)) as dnn_hidden_layer_scope:
                    dnn_input = layers.fully_connected(
                        dnn_input,
                        num_hidden_units,
                        None,
                        scope=dnn_hidden_layer_scope,
                        normalizer_fn=None,
                        normalizer_params=None,
                        variables_collections=[self.prediction_head_collections_dnn_hidden_layer],
                        outputs_collections=[self.prediction_head_collections_dnn_hidden_output]
                    )
        return dnn_input

    def build_mean_pooling(self, sequence_layer_dict, seq_column_len, block_name, name):
        logger.info("Debug!! Start Build Mean Pooling: {}".format(name))
        with tf.name_scope("build_mean_pooling_{}".format(name)):
                max_len = self.fg.get_seq_len_by_sequence_name(block_name)
                sequence = sequence_layer_dict[block_name]
                if block_name not in seq_column_len or seq_column_len[block_name] not in self.layer_dict:
                    sequence_mask = tf.sequence_mask(tf.ones_like(sequence[:, 0, 0], dtype=tf.int32), 1)
                    sequence_mask = tf.tile(sequence_mask, [1, max_len])
                else:
                    sequence_length = self.layer_dict[seq_column_len[block_name]]
                    sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), max_len)
                logger.info("Debug!! max len of {} is {}".format(block_name, str(max_len)))

                sequence = tf.stop_gradient(sequence)
                logger.info("stop_gradient_{}_{}".format(name, block_name))
                mask_expanded = tf.expand_dims(tf.cast(sequence_mask, tf.float32), axis=-1)  # [B, max_len, 1]
                masked_item_vec = sequence * mask_expanded
                sum_vec = tf.reduce_sum(masked_item_vec, axis=1)  # [B, d]
                valid_counts = tf.reduce_sum(tf.cast(sequence_mask, tf.float32), axis=1,keepdims=True) + 2
                dec = sum_vec / valid_counts

        logger.info("Debug!! Finsh Build Mean Pooling: {}".format(name))
        return dec

    def build_query_attention(self, attention, sequence_layer_dict, seq_column_len, block_name, name):
        logger.info("Debug!! Start Build Self Attention: {}".format(name))
        layer_dict ={}
        if sequence_layer_dict is None:
            return layer_dict
        with arg_scope(model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['atten_param']['attention_l2_reg'])):
            with self.variable_scope("build_query_attention_{}".format(name)) as scope:
                max_len = self.fg.get_seq_len_by_sequence_name(block_name)
                sequence = sequence_layer_dict[block_name]
                if block_name not in seq_column_len or seq_column_len[block_name] not in self.layer_dict:
                    sequence_mask = tf.sequence_mask(tf.ones_like(sequence[:, 0, 0], dtype=tf.int32), 1)
                    sequence_mask = tf.tile(sequence_mask, [1, max_len])
                else:
                    sequence_length = self.layer_dict[seq_column_len[block_name]]
                    sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), max_len)
                logger.info("Debug!! max len of {} is {}".format(block_name, str(max_len)))

                seq_potential_input = sequence
                dnn_output_units = sequence.get_shape().as_list()[-1]
                seq_potential_output = self.dnn_potential_emb_space(seq_potential_input, dnn_output_units,'long_seq_potential')
                sequence = tf.concat([sequence, seq_potential_output], -1)
                self.logger.info("##### query_attention_sequence_after_potential: {}".format(sequence))

                item_vec, stt_vec = atten_func(
                    query_masks=None,
                    key_masks=sequence_mask,
                    queries=attention,
                    keys=sequence,
                    values=None,
                    num_units=self.model_conf['model_hyperparameter']['atten_param']['dill_sa_num_units'],
                    num_output_units=self.model_conf['model_hyperparameter']['atten_param']['dill_sa_num_output_units'],
                    scope=name + "_query_attention",
                    atten_mode=self.model_conf['model_hyperparameter']['atten_param']['atten_mode'],
                    reuse=tf.AUTO_REUSE,
                    variables_collections=[self.self_atten_collections_dnn_hidden_layer],
                    outputs_collections=[self.self_atten_collections_dnn_hidden_output],
                    num_heads=self.model_conf['model_hyperparameter']['atten_param']['num_heads'],
                    residual_connection=self.model_conf['model_hyperparameter']['atten_param'].get('residual_connection',True),
                    attention_normalize=self.model_conf['model_hyperparameter']['atten_param'].get('attention_normalize',True),
                    use_atten_linear_project=self.model_conf['model_hyperparameter']['atten_param'].get('use_atten_linear_project', True))

                logger.info("Debug!!block_name is {}, item_vec is {}, stt_vec is {}".format(block_name, item_vec, stt_vec))
        logger.info("Debug!! Finsh Build Self Attention: {}".format(name))
        return item_vec

    def build_self_attention(self, sequence_layer_dict, seq_column_len, seq_names, max_len, name):
        logger.info("Debug!! Start Build Self Attention: {}".format(name))
        layer_dict ={}
        if sequence_layer_dict is None:
            return layer_dict

        with arg_scope(model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['atten_param']['attention_l2_reg'])):
            with self.variable_scope("build_self_attention_{}".format(name)) as scope:
                for block_name in seq_names:
                        sequence = sequence_layer_dict[block_name]
                        sequence_length = seq_column_len[block_name]
                        sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), max_len)
                        logger.info("Debug!! max len of {} is {}".format(block_name, str(max_len)))

                        seq_potential_input = sequence
                        dnn_output_units = sequence.get_shape().as_list()[-1]
                        seq_potential_output = self.dnn_potential_emb_space(seq_potential_input, dnn_output_units,'long_seq_potential')
                        sequence = tf.concat([sequence, seq_potential_output], -1)
                        self.logger.info("##### self_sequence_after_potential: {}".format(sequence))

                        item_vec, stt_vec = atten_func(
                            query_masks=None,
                            key_masks=sequence_mask,
                            queries=sequence,
                            keys=sequence,
                            values=None,
                            num_units=self.model_conf['model_hyperparameter']['atten_param']['dill_sa_num_units'],
                            num_output_units=self.model_conf['model_hyperparameter']['atten_param']['dill_sa_num_output_units'],
                            scope="MHA_{}".format(name),
                            atten_mode=self.model_conf['model_hyperparameter']['atten_param']['atten_mode'],
                            reuse=tf.AUTO_REUSE,
                            variables_collections=[self.self_atten_collections_dnn_hidden_layer],
                            outputs_collections=[self.self_atten_collections_dnn_hidden_output],
                            num_heads=self.model_conf['model_hyperparameter']['atten_param']['num_heads'],
                            residual_connection=self.model_conf['model_hyperparameter']['atten_param'].get('residual_connection', True),
                            attention_normalize=self.model_conf['model_hyperparameter']['atten_param'].get('attention_normalize', True),
                            use_atten_linear_project=self.model_conf['model_hyperparameter']['atten_param'].get('use_atten_linear_project', True)
                        )
                        self.logger.info("Debug! output of self attention is {}".format(item_vec))
                        mask_expanded = tf.expand_dims(tf.cast(sequence_mask, tf.float32), axis=-1)  # [B, max_len, 1]
                        masked_item_vec = item_vec * mask_expanded
                        sum_vec = tf.reduce_sum(masked_item_vec, axis=1)  # [B, d]
                        valid_counts = tf.reduce_sum(tf.cast(sequence_mask, tf.float32), axis=1,keepdims=True) + 2
                        dec = sum_vec / valid_counts
                        layer_dict[block_name] = dec
                        logger.info("Debug!!block_name is {}, item_vec is {}, stt_vec is {}, dec is {}".format(block_name, item_vec, stt_vec, dec))
        logger.info("Debug!! Finsh Build Self Attention: {}".format(name))
        return layer_dict

    def topk_interest_lookup(self, predicted_sid, k, max_len):
        with tf.name_scope("topk_interest_lookup"):

            topk_scores, topk_indices = tf.nn.top_k(
                predicted_sid,
                k=k,
                sorted=True,
                name="TopK_Scores_Indices"
            )  # [B, k], [B, k]

            topk_indices = topk_indices + 1

            long_seq_dict = {}
            long_seq_len_dict = {}

            for i in range(k):
                self.current_sid = tf.expand_dims(topk_indices[:, i], -1)
                match_mask = tf.equal(self.long_sids, self.current_sid)
                self.match_score = tf.cast(match_mask, tf.float32)
                _, self.selected_indices = tf.nn.top_k(self.match_score, k=max_len, sorted=True) # [B, max_len]
                sub_emb = tf.batch_gather(self.long_embs, self.selected_indices)
                gathered_sids = tf.batch_gather(self.long_sids, self.selected_indices)
                valid_mask = tf.cast(tf.equal(gathered_sids, self.current_sid), tf.float32)  # [B, max_len]
                valid_mask_expanded = tf.expand_dims(valid_mask, -1)  # [B, max_len, 1]
                sub_emb = sub_emb * valid_mask_expanded
                real_len = tf.reduce_sum(valid_mask, axis=1)
                real_len = tf.cast(real_len, tf.int32)
                long_seq_key = "long_seqs_{}".format(i + 1)
                long_seq_dict[long_seq_key] = sub_emb # [B, max_len, d]
                long_seq_len_dict[long_seq_key] = real_len # [B]

            return long_seq_dict, long_seq_len_dict

    def sequence_layer(self):
        with self.variable_scope("Sequence_Layer") as scope:
             with arg_scope(model_arg_scope(weight_decay=self.dnn_l2_reg)):
                self.mp_outputs = self.build_mean_pooling(self.sequence_layer_dict, self.seq_column_len,'all_clk_seq_list', "short_seq_mp")  # (B, D)
                self.predicted_sid = self.prediction_head(self.mp_outputs, [self.vocab_size], "predicted_sid")
                self.long_seq_dict, self.long_seq_len_dict = self.topk_interest_lookup(self.predicted_sid,self.long_seq_nums,self.long_seq_max_len)
                long_seq_names = ["long_seqs_{}".format(i + 1) for i in range(self.long_seq_nums)]
                self.sa_long_seq_outputs = self.build_self_attention(self.long_seq_dict, self.long_seq_len_dict,long_seq_names, self.long_seq_max_len, "long_seq_self")  # (B, D)

                self.seq_raw_input = self.sequence_layer_dict['all_clk_seq_list']
                seq_potential_input = self.seq_raw_input
                dnn_output_units = self.seq_raw_input.get_shape().as_list()[-1]
                self.seq_potential_input = self.dnn_potential_emb_space(seq_potential_input, dnn_output_units, 'short_seq_mind_potential')
                dynamic_routing_input = tf.concat([self.seq_raw_input, self.seq_potential_input], -1)
                self.sequence_length_1d = tf.reshape(self.layer_dict[self.seq_column_len['all_clk_seq_list']], [-1])
                self.interests, self.raw_interests, self.interest_weights, self.raw_interest_weights, self.routing_inputs = \
                    dynamic_routing(inputs=dynamic_routing_input,
                                    interest_num=self.interest_nums,
                                    interest_dim=self.interest_dim,
                                    inputs_length=self.sequence_length_1d,
                                    stddev_b=1,
                                    routing_iter=3,
                                    linear_transform=True,
                                    share_interert_weight=True,
                                    random_init=True,
                                    l2_normalize=self.dynamic_routing_l2_normalize,
                                    inner_activation='l2_norm',
                                    last_activation='squash',
                                    alpha=self.dynamic_routing_alpha,
                                    enlarge_factor=self.dynamic_routing_enlarge_factor,
                                    init_b_enlarge=False,
                                    scope='dynamic_routing'
                                    )
                self.long_interest_enh = self.build_query_attention(self.interests, self.sequence_layer_dict,self.seq_column_len, 'long_clk_seq_list', "long_seq")
                self.long_interest_enh_squash = squash(self.long_interest_enh, alpha=self.dynamic_routing_alpha)
                self.interests_concat1 = tf.concat([self.interests, self.long_interest_enh_squash], -1)
                self.interests_concat1_mlp = self.mlp(self.interests_concat1, 128, 'long_enh_seq_mlp')

                self.long_interest_supp = tf.stack([self.sa_long_seq_outputs[f'long_seqs_{i + 1}'] for i in range(self.long_seq_nums)],axis=1)
                self.long_interest_supp_squash = squash(self.long_interest_supp, alpha=self.dynamic_routing_alpha)
                self.long_interest_supp_mlp = self.mlp(self.long_interest_supp_squash, 128, 'long_supp_seq_mlp')
                self.interests_concat = tf.concat([self.interests_concat1_mlp, self.long_interest_supp_mlp], axis=1)

    def user_net(self):
        with self.variable_scope(name_or_scope="user_net") as scope:
            dnn_hidden_units = self.model_conf['model_hyperparameter']['user_net_dnn_hidden_units']
            self.user_vec = self.interests_concat

            with arg_scope(model_arg_scope(weight_decay=self.dnn_l2_reg)):
                for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                    with self.variable_scope(name_or_scope="user_hidden_layer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                        self.user_vec = layers.fully_connected(
                            self.user_vec,
                            num_hidden_units,
                            getActivationFunctionOp(self.activation),
                            scope=dnn_hidden_layer_scope,
                            variables_collections=[self.user_net_collections_dnn_hidden_layer],
                            outputs_collections=[self.user_net_collections_dnn_hidden_output],
                            normalizer_fn=layers.layer_norm,
                            normalizer_params={
                                "begin_norm_axis": -1,
                                "begin_params_axis": -1
                            }
                        )

                if self.use_user_context:
                    static_list = [self.layer_dict[name] for name in self.user_column_blocks]
                    self.static_vec = tf.concat(static_list, axis=-1)

                    for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                        with self.variable_scope(name_or_scope="hidden_layer_user_profile_{}".format(layer_id)) as dnn_hidden_layer_scope:
                            self.static_vec = layers.fully_connected(
                                self.static_vec,
                                num_hidden_units,
                                getActivationFunctionOp(self.activation),
                                scope=dnn_hidden_layer_scope,
                                variables_collections=[self.user_net_collections_dnn_hidden_layer],
                                outputs_collections=[self.user_net_collections_dnn_hidden_output],
                                normalizer_fn=layers.layer_norm,
                                normalizer_params={
                                    "begin_norm_axis": -1,
                                    "begin_params_axis": -1
                                }
                            )

                    static_tiled = tf.tile(tf.expand_dims(self.static_vec, 1),[1, self.interest_nums + self.long_seq_nums, 1])
                    self.user_vec = tf.concat([self.user_vec, static_tiled], axis=-1)

                    for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                        with self.variable_scope(name_or_scope="hidden_layer_union_{}".format(layer_id)) as dnn_hidden_layer_scope:
                            self.user_vec = layers.fully_connected(
                                self.user_vec,
                                num_hidden_units,
                                getActivationFunctionOp(self.model_conf['model_hyperparameter']['activation']),
                                scope=dnn_hidden_layer_scope,
                                variables_collections=[self.user_net_collections_dnn_hidden_layer],
                                outputs_collections=[self.user_net_collections_dnn_hidden_output],
                                normalizer_fn=layers.layer_norm,
                                normalizer_params={
                                    "begin_norm_axis": -1,
                                    "begin_params_axis": -1
                                }
                            )

            self.user_vec = tf.nn.l2_normalize(self.user_vec, dim=-1)
            self.user_vec = tf.identity(self.user_vec, name='user_vec')
            self.logger.info("##### user_vec shape: {}".format(self.user_vec)) # (?, 8, 128)

    def item_net(self):
        with self.variable_scope(name_or_scope="item_net") as scope:

            dnn_hidden_units = self.model_conf['model_hyperparameter']['item_net_dnn_hidden_units']

            dnn_output_units = self.pos_emb.get_shape().as_list()[-1]
            self.pos_item_vec = self.dnn_potential_emb_space(self.pos_emb, dnn_output_units, 'item')
            self.pos_item_vec = tf.concat([self.pos_emb, self.pos_item_vec], -1)
            self.neg_item_vec = self.dnn_potential_emb_space(self.neg_emb, dnn_output_units, 'item')
            self.neg_item_vec = tf.concat([self.neg_emb, self.neg_item_vec], -1)

            with arg_scope(model_arg_scope(weight_decay=0.0)):
                for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                    with self.variable_scope(
                            name_or_scope="item_hidden_layer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                        self.pos_item_vec = layers.fully_connected(
                            self.pos_item_vec,
                            num_hidden_units,
                            getActivationFunctionOp(self.activation),
                            scope=dnn_hidden_layer_scope,
                            variables_collections=[self.item_net_collections_dnn_hidden_layer],
                            outputs_collections=[self.item_net_collections_dnn_hidden_output],
                            normalizer_fn=None,
                            normalizer_params=None)

                        self.neg_item_vec = layers.fully_connected(
                            self.neg_item_vec,
                            num_hidden_units,
                            getActivationFunctionOp(self.activation),
                            scope=dnn_hidden_layer_scope,
                            variables_collections=[self.item_net_collections_dnn_hidden_layer],
                            outputs_collections=[self.item_net_collections_dnn_hidden_output],
                            normalizer_fn=None,
                            normalizer_params=None)

            self.pos_item_vec = tf.nn.l2_normalize(self.pos_item_vec, dim=-1)
            self.neg_item_vec = tf.nn.l2_normalize(self.neg_item_vec, dim=-1)
            self.pos_item_vec = tf.identity(self.pos_item_vec, name='pos_item_vec')
            self.neg_item_vec = tf.identity(self.neg_item_vec, name='neg_item_vec')
            self.logger.info("##### pos_item_vec shape: {}".format(self.pos_item_vec)) # (?, 128)
            self.logger.info("##### neg_item_vec shape: {}".format(self.neg_item_vec)) # (?, 128)

    def logits_layer(self):
        with self.variable_scope("logits_layer"):
            B = tf.shape(self.user_vec)[0]
            K = tf.shape(self.user_vec)[1]
            D = tf.shape(self.user_vec)[2]

            user_expanded = tf.expand_dims(self.user_vec, axis=2)  # [B, K, 1, D]
            pos_expanded = tf.expand_dims(self.pos_item_vec, axis=0)  # [1, B, D] → [B, B, D]
            pos_expanded = tf.expand_dims(pos_expanded, axis=1)  # [B, 1, B, D]
            sim = tf.reduce_sum(user_expanded * pos_expanded, axis=-1)  # [B, K, B]
            self.inbatch_pos_logits = tf.reduce_max(sim, axis=1) / self.temperature  # [B, B]

            self.neg_logits = tf.matmul(
                tf.reshape(self.user_vec, [B * K, D]),
                self.neg_item_vec,
                transpose_b=True
            )  # [B*K, B*N]
            self.neg_logits = tf.reshape(self.neg_logits, [B, K, -1])  # [B, K, B*N]
            self.neg_logits = tf.reduce_max(self.neg_logits, axis=1) / self.temperature  # [B, B*N]

            pos_sim = tf.reduce_sum(self.user_vec * tf.expand_dims(self.pos_item_vec, 1), axis=-1)  # [B, K, D] [B, D] -> [B, K, D] -> [B, K]
            self.pos_logits = tf.reduce_max(pos_sim, axis=-1, keepdims=True) / self.temperature  # [B, 1]

    def reg_loss_f(self):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_loss = tf.reduce_sum(reg_losses)

    def Category_loss(self):
        with tf.name_scope("Category_loss"):
            self.label_ids = tf.cast(self.tmsid_label, tf.int64)   # [B] or [B,1]
            if len(self.label_ids.shape) == 2:
                self.label_ids = tf.squeeze(self.label_ids, axis=-1)  # -> [B]
            self.label_ids = self.label_ids - 1

            self.per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.label_ids,  # [B]
                    logits=self.predicted_sid  # [B, N]
                )
            self.category_loss = tf.reduce_mean(self.per_example_loss)

            def calculate_hr_at_k(targets, predictions, k):
                in_top_k = tf.nn.in_top_k(predictions=predictions, targets=targets, k=k)
                return tf.reduce_mean(tf.cast(in_top_k, tf.float32))

            self.stag1_hit_rate_at_1 = calculate_hr_at_k(self.label_ids, self.predicted_sid, 1)
            self.stag1_hit_rate_at_5 = calculate_hr_at_k(self.label_ids, self.predicted_sid, 5)
            self.stag1_hit_rate_at_10 = calculate_hr_at_k(self.label_ids, self.predicted_sid, 10)
            self.stag1_hit_rate_at_15 = calculate_hr_at_k(self.label_ids, self.predicted_sid, 15)
            self.stag1_hit_rate_at_20 = calculate_hr_at_k(self.label_ids, self.predicted_sid, 20)
            self.stag1_hit_rate_at_25 = calculate_hr_at_k(self.label_ids, self.predicted_sid, 25)
            self.stag1_hit_rate_at_30 = calculate_hr_at_k(self.label_ids, self.predicted_sid, 30)

    def loss(self):
        with tf.name_scope("Loss_Op"):
            self.reg_loss_f()
            self.Category_loss()

            B = tf.shape(self.inbatch_pos_logits)[0]
            full_logits = tf.concat([self.inbatch_pos_logits, self.neg_logits], axis=-1)
            pos_labels = tf.eye(B)  # [B, B]
            neg_labels = tf.zeros_like(self.neg_logits)  # [B, B*N]
            full_labels = tf.concat([pos_labels, neg_labels], axis=-1)

            self.sampled_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=full_labels,
                logits=full_logits
            ))
            self.loss_op = self.sampled_loss + self.reg_loss + self.category_loss

    def get_optimizer(self, opt_name, opt_conf, global_step):
        optimizer = None
        for name in optimizer_dict:
            if opt_name == name:
                optimizer = optimizer_dict[name](opt_conf, global_step)
                break

        return optimizer

    def optimizer(self):
        with self.variable_scope("Optimize"):

            global_opt_name = None
            global_optimizer = None
            global_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=None)

            if len(global_opt_vars) == 0:
                raise ValueError("no trainable variables")

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            train_ops = []
            for opt_name, opt_conf in self.opts_conf.items():
                optimizer = self.get_optimizer(opt_name, opt_conf, self.global_step)
                global_opt_name = opt_name
                global_optimizer = optimizer

            if global_opt_name is not None:
                train_op, self.out_gradient_norm, self.out_var_norm = myopt.optimize_loss(
                    loss=self.loss_op,
                    global_step=self.global_step,
                    learning_rate=self.opts_conf[global_opt_name].get("learning_rate", 0.01),
                    optimizer=global_optimizer,
                    # update_ops=update_ops,
                    clip_gradients=self.opts_conf[global_opt_name].get('clip_gradients', 5.0),
                    variables=global_opt_vars,
                    increment_global_step=False,
                    summaries=myopt.OPTIMIZER_SUMMARIES,
                )
                train_ops.append(train_op)

            with tf.control_dependencies(update_ops):
                train_op_vec = control_flow_ops.group(*train_ops)
                with ops.control_dependencies([train_op_vec]):
                    with ops.colocate_with(self.global_step):
                        self.train_ops = state_ops.assign_add(self.global_step, 1).op

    def predictions(self, logits):
        pass

    def mark_output(self):
        with tf.name_scope("Mark_Output"):
            rank_predict = tf.identity(self.pos_logits, name="rank_predict")
            pos_item_vec = tf.identity(self.pos_item_vec, name="pos_item_vec")
            user_vec = tf.identity(self.user_vec, name="user_vec")

            self.logger.info("##### rank_predict: {}".format(rank_predict))

    def summary(self):
        with tf.name_scope("{}_Metrics".format(self.model_name)):
            self.metrics['scalar/loss_mean'] = self.loss_op
            self.metrics['scalar/reg_loss'] = self.reg_loss
            self.metrics['scalar/sampled_loss'] = self.sampled_loss
            self.metrics['scalar/category_loss'] = self.category_loss

            self.metrics['scalar/hit_rate_stag1'] = self.stag1_hit_rate_at_1
            self.metrics['scalar/hit_rate_at_5_stag1'] = self.stag1_hit_rate_at_5
            self.metrics['scalar/hit_rate_at_10_stag1'] = self.stag1_hit_rate_at_10
            self.metrics['scalar/hit_rate_at_15_stag1'] = self.stag1_hit_rate_at_15
            self.metrics['scalar/hit_rate_at_20_stag1'] = self.stag1_hit_rate_at_20
            self.metrics['scalar/hit_rate_at_25_stag1'] = self.stag1_hit_rate_at_25
            self.metrics['scalar/hit_rate_at_30_stag1'] = self.stag1_hit_rate_at_30

            for i in range(self.long_seq_nums):
                key = "long_seqs_{}".format(i + 1)
                real_len_tensor = self.long_seq_len_dict[key]
                mean_real_len = tf.reduce_mean(tf.cast(real_len_tensor, tf.float32))
                self.metrics['scalar/long_seq_{}_real_length'.format(i)] = mean_real_len

            # eval
            tf_conf = json.loads(os.environ['TF_CONFIG'])
            self.logger.info(('##### tf_conf:{}'.format(tf_conf)))
            task_index = tf_conf["task"]["index"]

            if task_index <= 1:

                # recall
                top_50_indices_list = []
                all_interest_logits = []

                # [batch+batch*neg_num, dim]
                item_vec = tf.concat([self.pos_item_vec, self.neg_item_vec], 0)
                item_ids = tf.concat([self.pos_item_ids, self.neg_item_ids], 0)
                # [batch]
                labels = tf.cast(tf.range(tf.shape(self.pos_item_vec)[0]), tf.int64)

                B = tf.shape(labels)[0]  # batch size
                M = tf.shape(item_vec)[0]  # total items = B + B*N

                recall_ks = [1, 3, 5, 10, 20, 30, 40, 50]
                top_k_indices_dict = {k: [] for k in recall_ks}

                for i in range(self.interest_nums + self.long_seq_nums):
                    # [batch, batch+batch*neg_num]
                    interest_logits = tf.matmul(self.user_vec[:, i, :], item_vec, transpose_b=True)
                    all_interest_logits.append(interest_logits)

                    _, top_50_indices = tf.nn.top_k(interest_logits, k=50)
                    top_50_indices_list.append(top_50_indices)

                    for k in recall_ks:
                        top_k_indices_dict[k].append(top_50_indices[:, :k])

                    current_recall_50, update_recall_50 = metrics.recall_at_k(
                        labels=labels,
                        predictions=interest_logits,
                        k=50,
                        name='local_interest_{}_recall_50'.format(i)
                    )
                    with tf.control_dependencies([update_recall_50]):
                        current_recall_50 = tf.identity(current_recall_50)
                    self.metrics["scalar/interest_{}_recall_50".format(i)] = current_recall_50

                # [batch, interest_nums, 50]
                self.top_50_indices = tf.stack(top_50_indices_list, axis=1)
                self.top_50_item_ids = tf.gather(item_ids, self.top_50_indices)
                self.logger.info("##### top_50_indices: {}".format(self.top_50_indices))
                self.logger.info("##### top_50_item_ids: {}".format(self.top_50_item_ids))

                labels_all = tf.cast(tf.expand_dims(labels, 1), tf.int32)  # [batch, 1]

                for k in recall_ks:
                    top_k_indices_all = tf.concat(top_k_indices_dict[k], axis=1)
                    is_match_tensor = tf.cast(tf.equal(top_k_indices_all, labels_all), tf.float32)
                    is_match_any = tf.reduce_max(is_match_tensor, axis=-1)
                    recall_k = tf.reduce_mean(is_match_any)
                    self.metrics["scalar/a_all_interest_recall_{}".format(k)] = recall_k

                aggregated_logits = tf.reduce_max(tf.stack(all_interest_logits, axis=1), axis=1)  # [B, M]
                pos_scores = tf.linalg.diag_part(aggregated_logits[:, :B])  # [B]
                pos_scores = tf.expand_dims(pos_scores, axis=1)  # [B, 1]
                greater = aggregated_logits > pos_scores  # [B, M]
                per_sample_count = tf.reduce_sum(tf.cast(greater, tf.float32), axis=1)  # [B]
                per_sample_total = tf.cast(M - 1, tf.float32)
                per_sample_ratio = per_sample_count / per_sample_total

                self.metrics["scalar/avg_num_neg_greater_per_sample"] = tf.reduce_mean(per_sample_count)
                self.metrics["scalar/avg_ratio_neg_greater_per_sample"] = tf.reduce_mean(per_sample_ratio)

        with tf.name_scope("Metrics_Scalar"):
            for key, metric in self.metrics.items():
                tf.summary.scalar(name=key, tensor=metric)

        with tf.name_scope('{}_Embedding_Summary'.format(self.model_name)):
            set_name = set()
            add_embed_layer_norm(self.pos_emb, self.feature_columns['target_columns'], omit=set_name)
            add_embed_layer_norm(self.neg_emb, self.feature_columns['target_columns'], omit=set_name)
            for block_name, layer in self.layer_dict.items():
                add_embed_layer_norm(layer, self.feature_columns[block_name], omit=set_name)

        # variable
        with tf.name_scope("{}_Self_Atten_Summary".format(self.model_name)):
            add_norm2_summary(self.self_atten_collections_dnn_hidden_layer)
            add_dense_output_summary(self.self_atten_collections_dnn_hidden_output)
            add_weight_summary(self.self_atten_collections_dnn_hidden_layer)

        with tf.name_scope("{}_Mlp_Summary".format(self.model_name)):
            add_norm2_summary(self.mlp_collections_dnn_hidden_layer)
            add_dense_output_summary(self.mlp_collections_dnn_hidden_output)
            add_weight_summary(self.mlp_collections_dnn_hidden_layer)

        with tf.name_scope("{}_potential_emb_Summary".format(self.model_name)):
            add_norm2_summary(self.potential_emb_collections_dnn_hidden_layer)
            add_dense_output_summary(self.potential_emb_collections_dnn_hidden_output)
            add_weight_summary(self.potential_emb_collections_dnn_hidden_layer)

        with tf.name_scope("{}_Prediction_Mlp_Summary".format(self.model_name)):
            add_norm2_summary(self.prediction_mlp_collections_dnn_hidden_layer)
            add_dense_output_summary(self.prediction_mlp_collections_dnn_hidden_output)
            add_weight_summary(self.prediction_mlp_collections_dnn_hidden_layer)

        with tf.name_scope("{}_Prediction_Head_Summary".format(self.model_name)):
            add_norm2_summary(self.prediction_head_collections_dnn_hidden_layer)
            add_dense_output_summary(self.prediction_head_collections_dnn_hidden_output)
            add_weight_summary(self.prediction_head_collections_dnn_hidden_layer)

        with tf.name_scope("{}_User_Net_Summary".format(self.model_name)):
            add_norm2_summary(self.user_net_collections_dnn_hidden_layer)
            add_dense_output_summary(self.user_net_collections_dnn_hidden_output)
            add_weight_summary(self.user_net_collections_dnn_hidden_layer)

        with tf.name_scope("{}_Item_Net_Summary".format(self.model_name)):
            add_norm2_summary(self.item_net_collections_dnn_hidden_layer)
            add_dense_output_summary(self.item_net_collections_dnn_hidden_output)
            add_weight_summary(self.item_net_collections_dnn_hidden_layer)

        return self.metrics

    def run_predict(self, context, mon_session, task_index, thread_index):
        if int(thread_index) != 0:  # predict with one thread
            self.logger.info("Skip thread_ind==%s" % str(thread_index))
            return

        self.odps = myodps.resetOdpsTable(self.algo_config.get('table_name'), task_id=task_index,
                                          local_mode=False, odps_user=self.algo_config.get('odps_user'))
        self.tablewriter = myodps.getTableWriter(self.odps,
                                                 self.algo_config.get('table_name'),
                                                 task_id=task_index,
                                                 ds_output='eval_part',
                                                 local_mode=False)

        predict_step = self.algo_config.get('predict_max_step', 100)
        localcnt = 0
        id_feature_tensor = self.features["id"]
        try:
            id_feature_tensor = tf.sparse_tensor_to_dense(id_feature_tensor, default_value="")
            self.logger.info("#qlLog# self.id_feature_tensor")
        except:
            pass

        run_ops = [self.prediction, self.label, id_feature_tensor]
        debug_tensor_names = []
        for tensor_name, tensor in self.debug_tensor_collector.items():
            debug_tensor_names.append(tensor_name)
            run_ops.append(tensor)

        print('global_variables')
        for variable_name in tf.global_variables():
            print(variable_name)

        while True:
            localcnt += 1
            feed_dict = {'training:0': False}

            try:
                run_res = mon_session.run(run_ops, feed_dict=feed_dict)
                prob, y, qid = run_res[:3]
                records = []
                for i in range(len(prob)):
                    one_res = [str(prob[i][0]), str(y[i][0]), str(qid[i][0])]
                    ex = []
                    for tensor_name, tensor_value in zip(debug_tensor_names, run_res[3:]):
                        ex.append('{}={}'.format(tensor_name, str(tensor_value[i].tolist())))
                    if len(ex) > 0:
                        ex_str = '#'.join(ex)
                        one_res.append(ex_str)
                    else:
                        one_res.append('-')
                    records.append(one_res)

                self.tablewriter.write(task_index, records)
                self.logger.info(
                    'model_name=%s, size=%s,  step_left=%s' % (self.model_name, str(len(prob)), str(predict_step)))

                predict_step -= 1
                if predict_step < 1:
                    break
            except (ResourceExhaustedError, OutOfRangeError) as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                break
            except ConnectionError as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                self.logger.info("Reset table writer")
                self.odps = myodps.resetOdpsTable(self.algo_config.get('table_name'), task_id=task_index,
                                                  local_mode=False, odps_user=self.algo_config.get('odps_user'))
                self.tablewriter = myodps.getTableWriter(self.odps,
                                                         self.algo_config.get('table_name'),
                                                         task_id=task_index,
                                                         ds_output='eval_part',
                                                         local_mode=False)
            except Exception as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))

        try:
            if self.tablewriter is not None:
                self.tablewriter.close()
        except ConnectionError as e:
            self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
            self.logger.info("Reset table writer when close")
            self.odps = myodps.resetOdpsTable(self.algo_config.get('table_name'), task_id=task_index,
                                              local_mode=False, odps_user=self.algo_config.get('odps_user'))
            self.tablewriter = myodps.getTableWriter(self.odps,
                                                     self.algo_config.get('table_name'),
                                                     task_id=task_index,
                                                     ds_output='eval_part',
                                                     local_mode=False)
            if int(task_index) != 0:
                raise RuntimeError("Reset table writer when close")

            time.sleep(60)
            self.logger.info("Finish run_predict, sleep")

    def run_evaluate(self, context, mon_session, task_index, thread_index):
        localcnt = 0
        while True:
            localcnt += 1
            run_ops = [self.global_step_add, self.global_step, self.metrics, self.label, self.localvar]
            try:
                feed_dict = {'training:0': False}
                _, global_step, metrics, labels, flocalv = mon_session.run(
                    run_ops, feed_dict=feed_dict)
                if len(self.localvar) > 0:
                    index = np.array([0, -1])
                    self.logger.info(('localcnt:%s\t' % str(localcnt)) + '//'.join([x.name for x in self.localvar]))
                    self.logger.info(('localcnt:%s\t' % str(localcnt)) + '//'.join([str(x[index]) for x in flocalv]))

                auc, totalauc = metrics['scalar/auc'], metrics['scalar/total_auc']
                self.logger.info(
                    'Global_Step:{}, poslabel:{}, auc={}, totalauc={} thread={}'.format(
                        str(global_step),
                        str(labels.sum()),
                        str(auc),
                        str(totalauc),
                        str(thread_index)))
                newmark = np.max(flocalv[0][np.array([0, -1])])
                if newmark > self.metric_conf['auc_compute'].get('true_positives', 20000):
                    self.logger.info("positive_num now:{}".format(str(newmark)))
                    self.logger.info("auc_reset_step:{}".format(str(1000)))
                    self.logger.info('reset auc ops run')
                    index = np.array([0, -1])
                    flocalv = mon_session.run(self.reset_auc_ops, feed_dict=feed_dict)
                    self.logger.info(
                        ('localcnt:{}\t'.format(str(localcnt))) + '//'.join([x.name for x in self.localvar]))
                    self.logger.info(
                        ('localcnt:{}\t'.format(str(localcnt))) + '//'.join([str(x[index]) for x in flocalv]))

            except (ResourceExhaustedError, OutOfRangeError) as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                break  # release all
            except ConnectionError as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
            except Exception as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))

    def run_train(self, context, mon_session, task_index, thread_index):
        localcnt = 0
        while True:
            localcnt += 1
            self.id = self.features["id"]
            run_ops = [self.global_step, self.loss_op, self.metrics, self.id]
            try:
                if task_index == 0:
                    feed_dict = {'training:0': False}
                    run_ops.extend(
                        [self.label_ids, self.top_50_item_ids, self.pos_item_ids, self.neg_item_ids,self.inbatch_pos_logits,self.neg_logits
                            ,self.long_sids,self.current_sid,self.match_score,self.selected_indices,self.long_seq_dict,self.long_seq_len_dict])
                    (global_step, loss, metrics, id, label_ids, top_50_item_ids, pos_item_ids, neg_item_ids, inbatch_pos_logits, neg_logits
                         ,long_sids,current_sid,match_score,selected_indices,long_seq_dict,long_seq_len_dict) \
                        = mon_session.run(run_ops, feed_dict=feed_dict)
                    self.logger.info(
                        'Global_Step:{}, loss={}, metrics={}, id={}, '
                        'label_ids={}, top_50_item_ids={}, pos_item_ids={}, neg_item_ids={}, inbatch_pos_logits={}, neg_logits={}'
                        ', long_sids={}, current_sid={}, match_score={}, selected_indices={}, long_seq_dict={}, long_seq_len_dict={}'.format(
                            global_step,
                            loss,
                            metrics,
                            id,
                            label_ids,
                            ','.join(map(str, top_50_item_ids[0])),
                            pos_item_ids,
                            neg_item_ids,
                            inbatch_pos_logits,
                            neg_logits,
                            long_sids,
                            current_sid,
                            match_score,
                            selected_indices,
                            long_seq_dict,
                            long_seq_len_dict,
                        ))

                    # if global_step - local_step > 100000:
                    #     _ = mon_session.run(self.reset_ops, feed_dict=feed_dict)
                    #     self.logger.info('Global_Step:{}, reset...'.format(global_step))
                    #     local_step = global_step
                else:
                    feed_dict = {'training:0': True}
                    run_ops.append(self.train_ops)
                    global_step, loss, metrics, id, _ = mon_session.run(run_ops, feed_dict=feed_dict)

                    self.logger.info(
                        'Global_Step:{}, loss={}, metrics={}, id={}'.format(
                            global_step,
                            loss,
                            metrics,
                            id[0],
                        ))

            except (ResourceExhaustedError, OutOfRangeError) as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                break  # release all
            except ConnectionError as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
            except Exception as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))




