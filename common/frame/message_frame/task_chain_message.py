from common.frame.message_frame.message import Message


class TaskChainMessage(Message):
    TASK_CHAIN_ID = 'task_chain_id'
    PARAMETER_CHAIN = 'parameter_chain'
    REDIS_TASKINFO = 'redis_taskinfo'
    REDIS_COMMUNICATION = 'redis_communication'
