from common.util import constant


class Parameter(object):
    def __init__(self, task_type=None, task_role=constant.TaskRole.GUEST):
        """

        :param task_type: str, see AlgorithmFactory for a full list on available tasks
        :parameter task_role: List[str], where str is constant.TASK_ROLE.GUEST or constant.TASK_ROLE.HOST
        """
        self.task_type = task_type
        self.task_role = task_role
