from common.frame.parameter_frame.parameter import Parameter
from common.util import constant


class EvaluateParameter(Parameter):
    def __init__(self,
                 task_role):
        """

        :param task_role:
        :param objective: learning objective
        """
        super(EvaluateParameter, self).__init__(constant.TaskType.EVALUATE, task_role)