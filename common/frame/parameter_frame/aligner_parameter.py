from common.frame.parameter_frame.parameter import Parameter
from common.util import constant


class AlignerParameter(Parameter):
    def __init__(self,
                 task_role=constant.TaskRole.GUEST,
                 align_mode=constant.Encryptor.PLAIN,
                 output_id_only=True,
                 sync_intersection=True,
                 key_size=1024,
                 batch_num="auto"):
        """

        :param task_role:
        :param align_mode:
        :param output_id_only:
        :param sync_intersection: whether host shares the final results to guest
        :param key_size:
        :param batch_num: "auto" or int (will be rounded up to power of 2 in cm20PSI)
        """
        super(AlignerParameter, self).__init__(constant.TaskType.ALIGNER, task_role)

        self.align_mode = align_mode
        self.output_id_only = output_id_only
        self.sync_intersection = sync_intersection
        self.key_size = key_size
        self.batch_num = batch_num
