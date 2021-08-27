class Model(object):
    def __init__(self, parameter, party_info):
        """

        :param parameter: Parameter
        :param party_info: Tuple with length of three
        """
        self._parameter = parameter
        self._party_info = party_info

    @property
    def parameter(self):
        return self._parameter

    @property
    def party_info(self):
        return self._party_info

    def __str__(self):
        content = ''
        content += str(self._parameter)
        content += str(self._party_info)
        return content
