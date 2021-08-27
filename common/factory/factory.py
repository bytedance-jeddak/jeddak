class Factory(object):
    @staticmethod
    def init(*args):
        """
        Initialize a global instance
        :param args:
        :return:
        """
        pass

    @staticmethod
    def get_global_instance():
        """
        Get the global instance
        :return:
        """
        pass

    @staticmethod
    def get_instance(*args):
        """
        Get an instance
        :param args:
        :return:
        """
        pass

    @staticmethod
    def _raise_value_error(invalid_type, invalid_value):
        """

        :param invalid_type: str
        :param invalid_value: str
        :return:
        """
        raise ValueError("invalid {}: {}".format(invalid_type, invalid_value))
