from common.frame.data_frame.tree import Forest
from common.frame.model_frame.model import Model


class DPGBDTModel(Model):
    def __init__(self, parameter, party_info, forest):
        super(DPGBDTModel, self).__init__(parameter, party_info)
        
        self._forest = forest

    @property
    def forest(self):
        return self._forest

    def __str__(self):
        content = super.__str__(self)

        for forest_idx, forest in enumerate(self._forest):
            content += '\n' + 'forest.' + str(forest_idx) + '\n'
            content += str(forest)

        return content
