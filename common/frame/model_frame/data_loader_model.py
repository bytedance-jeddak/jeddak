from common.frame.model_frame.model import Model


class DataLoaderModel(Model):
    def __init__(self, parameter, party_info, sparse_index2dict=None):
        """

        :param parameter:
        :param party_info:
        :param sparse_index2dict: map<column_index, string2int_map>
        """
        super(DataLoaderModel, self).__init__(parameter, party_info)
        
        self._sparse_index2dict = sparse_index2dict

    @property
    def sparse_index2dict(self):
        return self._sparse_index2dict
