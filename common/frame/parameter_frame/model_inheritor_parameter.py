from common.frame.parameter_frame.parameter import Parameter
from common.util import constant


class ModelInheritorParameter(Parameter):
    def __init__(self,
                 task_role=constant.TaskRole.GUEST,
                 model_id=None,
                 target_model=None):
        """
        ModelInheritorParameter
        :param task_role: str
        :param model_id: str
        """
        super(ModelInheritorParameter, self).__init__(constant.TaskType.MODEL_INHERITOR, task_role)

        self.model_id = model_id
        self.target_model = target_model
