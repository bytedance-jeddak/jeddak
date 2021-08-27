import logging
import os
import threading

from common.factory.factory import Factory


top_logger = logging.getLogger('top')
top_logger.setLevel(logging.DEBUG)
top_logger_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../..', 'common', 'log', "top.log")
fh = logging.FileHandler(top_logger_path, mode='a')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
top_logger.addHandler(fh)

mutex = threading.Lock()


class LoggerFactory(Factory):
    ctx = threading.local()
    logger = None

    @staticmethod
    def init(task_id, party_name=None, root_dir=None):
        LoggerFactory.ctx.logger = logging.getLogger(task_id)
        LoggerFactory.ctx.logger.setLevel(logging.DEBUG)

        # default root dir for log files
        if not root_dir:
            root_dir = os.path.split(os.path.realpath(__file__))[0]
        log_path = os.path.join(root_dir, '../..', 'common', 'log', party_name)
        with mutex:
            if not os.path.exists(log_path):
                os.makedirs(log_path)
        log_file_name = os.path.join(log_path, task_id + '.log')

        LoggerFactory.ctx.logger.handlers.clear()

        _fh = logging.FileHandler(log_file_name, mode='a')
        _fh.setLevel(logging.DEBUG)
        _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        _fh.setFormatter(_formatter)

        LoggerFactory.ctx.logger.addHandler(_fh)
        LoggerFactory.ctx.logger.info(task_id)

    @staticmethod
    def get_global_instance():
        return LoggerFactory.ctx.logger
