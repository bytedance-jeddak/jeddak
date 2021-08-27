from common.frame.parameter_frame.parameter import Parameter
from common.util import constant


class TaskChainParameter(Parameter):
    def __init__(self,
                 party_names,
                 messenger_server=None,
                 messenger_type=constant.MessengerType.KAFKA,
                 save_model=None):
        """
        :param party_names: str, all parties' names in task chain
        :param messenger_server: str, a messenger hosting the communication channel in federated computing
        :param messenger_type: str
        """
        super(TaskChainParameter, self).__init__(task_type=constant.TaskType.TASK_CHAIN, task_role=None)

        self.party_names = party_names

        if messenger_server is None:
            messenger_server = ['localhost:9092', 'localhost:9000']
        msg_server_length = len(messenger_server)
        self.messenger_server = messenger_server[0] if msg_server_length >= 1 else None
        self.fs_server = messenger_server[1] if msg_server_length >= 2 else None

        self.messenger_type = messenger_type
        
        self.save_model = save_model

        