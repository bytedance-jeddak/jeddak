import queue
import time
import json
import threading
import traceback

from common.factory.logger_factory import LoggerFactory
from common.factory.messenger_factory import MessengerFactory
from common.factory.parameter_factory import ParameterFactory
from common.frame.message_frame.task_chain_message import TaskChainMessage
from common.frame.message_frame.redis_message import RedisMessage
from common.util import constant
from common.util.random import Random
from common.util.constant import MessengerType
from coordinator.web.console import web_console
from coordinator.core.task_chain import TaskChain
from coordinator.core.sqlite_manager import SqliteManager


class TaskCoordinator(threading.Thread):
    """
    The overall task web_console
    entry for master and worker
    """
    # this is the task parameter queue [(active's, passive's), (active's, passive's), ...]
    # updated by the active party
    task_parameter_pool = queue.Queue()

    def __init__(self, syncer_server, syncer_type, party_name):
        super(TaskCoordinator, self).__init__()

        self._syncer_server = syncer_server
        self._syncer_type = syncer_type
        self.party_name = party_name
        self._syncer = MessengerFactory.get_instance(syncer_server, syncer_type)

    @staticmethod
    def reset_task_queue(syncer_server, syncer_type, party_name=None):
        _syncer = MessengerFactory.get_instance(syncer_server, syncer_type)
        _syncer.clear_all()

    @staticmethod
    def submit(task_json, task_id, parameter_chain):
        """
        Register task JSON to the task chain pool, which is submitted by users
        :param task_json: dict user-input task JSON
        :param task_id: generated unique task_id
        """
        web_console.logger.info(task_id)
        web_console.logger.info(task_json)
        TaskCoordinator.task_parameter_pool.put((task_json, task_id, parameter_chain))

    @staticmethod
    def generate(task_json):
        """
        Generate parameter chain
        :param task_json: dict user-input task JSON
        return list[Parameter]
        """
        parameter_dicts = TaskCoordinator.parse_task_json_to_dict(task_json)
        # construct a parameter chain
        parameter_chain = []
        for parameter_dict in parameter_dicts:
            parameter = {}
            for k, v in parameter_dict.items():
                v1 = ParameterFactory.get_instance(v)
                parameter[k] = v1
            parameter_chain.append(parameter)

        return parameter_chain

    def run(self):
        # PART.1 master for coordinating all parties
        # master is an active thread which synchronizes tasks
        def master():
            while True:
                try:
                    # pop a task parameter and construct the corresponding task chain
                    raw_task_json, task_id, parameter_chain = TaskCoordinator.task_parameter_pool.get(True)

                    web_console.logger.info("+ active receive task request")
                    name = self.party_name
                    _parameter_chain = []
                    for parameter in parameter_chain:
                        _parameter_chain.append(parameter[name])
                    new_chain = {}
                    _party_names = _parameter_chain[0].party_names
                    for p in _party_names:
                        new_chain[p] = []
                        for parameter in parameter_chain:
                            new_chain[p].append(parameter[p])
                    parameter_chain = new_chain
                    # construct message to sync
                    msg1 = {
                        "id": task_id,
                        "par": parameter_chain,
                        "raw": raw_task_json,
                        "master": self.party_name
                    }
                    if self._syncer_type == MessengerType.LIGHT_KAFKA:
                        self._syncer.send(msg1, tag=TaskChainMessage.TASK_CHAIN_ID)

                except Exception as e:
                    web_console.logger.error('master error {}'.format(e))
                    print(str(traceback.format_exc()))
                web_console.logger.info("+ active emit task")

        threading.Thread(target=master).start()

        # PART.2 replica for running tasks
        # replica is a passive loop which listens to new task commands
        while True:
            # executor pull and run
            # this is the main function to run tasks
            def rt2(_parameter_chain, _task_chain_id):
                # construct a task chain

                try:
                    task_chain = TaskChain(_parameter_chain, _task_chain_id, coordinator=self)
                    web_console.logger.info(
                        "- party {} task_id {} thread {} before run".format(self.party_name, _task_chain_id,
                                                                            threading.current_thread()))
                    task_chain.run()
                    # is_finish: the task finishes normally instead of being terminated
                    is_finish = True
                    if is_finish:
                        SqliteManager.task_dbo.update(unique_id=_task_chain_id, args={
                            'field': {
                                'status': RedisMessage.FINISH
                            }
                        })
                        if task_chain.save_model:
                            SqliteManager.model_dbo.create(dict(
                                model_id=_task_chain_id,
                                alias='',
                                deleted=False,
                            ))
                    else:
                        # update task status
                        SqliteManager.task_dbo.update(unique_id=_task_chain_id, args={
                            'field': {
                                'status': RedisMessage.ERROR
                            }
                        })
                        # create task progress with error message
                        SqliteManager.task_progress_dbo.create(args=dict(
                            task_id=_task_chain_id,
                            progress_type=constant.BoardConstants.TASK_PROGRESS_ERROR_MSG,
                            progress_value='the task is stopped manually, or some exceptions happened in other parties'
                        ))

                except Exception as e:
                    # update task status
                    SqliteManager.task_dbo.update(unique_id=_task_chain_id, args={
                        'field': {
                            'status': RedisMessage.ERROR
                        }
                    })
                    # create task progress with error message
                    SqliteManager.task_progress_dbo.create(args=dict(
                        task_id=_task_chain_id,
                        progress_type=constant.BoardConstants.TASK_PROGRESS_ERROR_MSG,
                        progress_value=str(traceback.format_exc())
                    ))

                    web_console.logger.error('task error {}'.format(e))

                LoggerFactory.get_global_instance().info(
                    "- party {} task_id {} thread {} finish".format(self.party_name, _task_chain_id,
                                                                    threading.current_thread()))
                web_console.logger.info(
                    "- party {} task_id {} thread {} after run".format(self.party_name, _task_chain_id,
                                                                       threading.current_thread()))

            # for kafka syncer, receive task chain info and create a new thread to execute the task
            # receive task chain info
            web_console.logger.info("- party {} waiting for task ".format(self.party_name))
            msg_recv = self._syncer.receive(tag=TaskChainMessage.TASK_CHAIN_ID)

            # deal with task messages just received
            task_chain_id = msg_recv["id"]
            msg = msg_recv['par']
            this_msg = msg[self.party_name]
            this_parameter_chain = this_msg
            web_console.logger.info("- party {} task_id {} receive task".format(self.party_name, task_chain_id))

            # run the task thread
            t = threading.Thread(target=rt2, args=(this_parameter_chain, task_chain_id))
            t.start()

            # create task data in local db
            SqliteManager.task_dbo.create(dict(
                task_id=task_chain_id,
                party_name=msg_recv['master'],
                party_names=','.join(msg.keys()),
                task_chain=json.dumps(msg_recv['raw']),
                status=RedisMessage.RUN,
                alias='',
                deleted=False,
            ))

            web_console.logger.info("- party {} task_id {} emit task".format(self.party_name, task_chain_id))

    @staticmethod
    def generate_task_chain_id():
        return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + Random.generate_random_digits()

    @property
    def syncer_type(self):
        return self._syncer_type

    @property
    def syncer_server(self):
        return self._syncer_server

    # pre-process task JSON
    @staticmethod
    def parse_task_json_to_dict(task_json):
        par_task_chain = task_json[0]
        parameter_dicts = []
        party_names = par_task_chain['party_names']
        for data_block in task_json:
            parameter_parties = {}
            for p in party_names:
                parameter_parties[p] = {}
            for k, v in data_block.items():
                if k == constant.KeyWord.TASK_TYPE or \
                        k == constant.KeyWord.PARTY_NAMES or \
                        k == constant.KeyWord.TASK_ROLE or \
                        type(v) != list:
                    for p in party_names:
                        parameter_parties[p][k] = v
                else:
                    i = 0
                    for p in party_names:
                        parameter_parties[p][k] = v[i]
                        i += 1
            parameter_dicts.append(parameter_parties)
        return parameter_dicts
