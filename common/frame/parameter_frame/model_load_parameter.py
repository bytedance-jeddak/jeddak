from common.frame.parameter_frame.parameter import Parameter
from common.util import constant


class ModelLoadParameter(Parameter):
    def __init__(self,
                 task_role=constant.TaskRole.GUEST,
                 model_id=None,
                 action=None):
        """

        :param task_role: str
        :param model_id: str
        :param action: str
        """
        super(ModelLoadParameter, self).__init__(constant.TaskType.MODEL_LOADER, task_role)

        self.model_id = model_id
        self.action = action
