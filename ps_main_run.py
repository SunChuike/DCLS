# -*- coding: utf-8 -*-

from runner.base_ps_plugin import BasePsPlugin
from prada_utils.log import logger
from model_util.my_graph_learn import MyGraphLearn

class RuiDeGraphLearnPsPlugin(BasePsPlugin):
    def run(self, context):
        config = context.get_config()
        self._my_graph_learn = MyGraphLearn(context, config)
        if not self._my_graph_learn.init_graph_learn_server():
            logger.error("Init graph learn server failed")
            return False

        logger.info("Run graph learn init suc")
        return True

