from common.factory.factory import Factory
from common.frame.data_frame.gh_pair import GHPair
from common.util import constant


class DataFrameFactory(Factory):
    @staticmethod
    def get_instance(data_frame_type=constant.DataFrameType.GH_PAIR):
        if data_frame_type == constant.DataFrameType.GH_PAIR:
            return GHPair
