from common.factory.logger_factory import LoggerFactory
from common.util import constant


class Operator(object):
    """
    The operator base class
    """
    def __init__(self, need_log):
        """

        :param need_log: bool
        """
        self._logger = None
        if need_log:
            self._logger = LoggerFactory.get_global_instance()

    def _log(self, content, level=constant.LogLevel.INFO):
        """

        :param content: str
        :param level:
        :return:
        """
        if self._logger is None:
            return

        class_name = "[{}]".format(type(self).__name__.upper())

        content = class_name + ' ' + str(content)
        if level == constant.LogLevel.INFO:
            self._logger.info(content)
        elif level == constant.LogLevel.DEBUG:
            self._logger.debug(content)
        elif level == constant.LogLevel.ERROR:
            self._logger.error(content)
        else:
            raise ValueError("invalid log level: {}".format(level))
