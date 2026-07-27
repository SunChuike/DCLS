# -*- coding: utf-8 -*-

from runner.graph_learn_client import GraphLearnClient
import graphlearn as gl

class MyGraphLearn(GraphLearnClient):
    def __init__(self, context, config):
        self.set_context(context)
        self.config = config

    def _init_graph(self):
        gl.set_padding_mode(gl.CIRCULAR)
        node_table = str(self.config.get_job_config("node_table"))
        attr_types = self.config.get_job_config("attr_types")
        self._graph = self._graph \
            .node(node_table, node_type='i',
                  decoder=gl.Decoder(weighted=True, attr_types=attr_types))

        return True


