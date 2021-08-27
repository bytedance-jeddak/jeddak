from common.frame.message_frame.message import Message


class RedisMessage(Message):
    # topic
    CONTROL = 'ctrl'
    REPORT = 'report'
    COMMUNICATION = 'cmm'
    PUSHTASK = 'push_task'
    PULLTASK = 'pull_task'
    UPDATETASK = 'update_task'
    DELETETASK = 'delete_task'
    # header
    FROM = 'from'
    TO = 'to'
    CONTENT = 'content'
    # key
    TASKID = 'task_id'
    TASKINFO = 'taskinfo'
    PARTYNAME = 'party_name'
    PARTYNAMES = 'party_names'
    TASKCHAINPARAMS = 'task_chain_params'
    START_BY_ID = 'start_id'
    ACT = 'act'
    ACK = 'ack'
    SUBMITTIME = 'submit_time'
    STARTTIME = 'start_time'
    ENDTIME = 'end_time'
    # value
    RUN = 'run'
    STOP = 'stop'
    WAIT = 'wait'
    FINISH = 'finish'
    RESET = 'reset'
    STATUS = 'status'
    READY = 'ready'
    OK = 'ok'
    ERROR = 'error'

