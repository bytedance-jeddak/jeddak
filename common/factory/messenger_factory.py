from common.util import constant
from common.factory.factory import Factory
from messenger.core.kafka_messenger import KafkaMessenger
from messenger.core.light_kafka_messenger import LightKafkaMessenger
import threading


class MessengerFactory(Factory):
    ctx = threading.local()
    communicator = None

    @staticmethod
    def init(bootstrap_server,
             messenger_type,
             party_name,
             other_party_names,
             fs_server=None,
             task_chain_id=constant.TaskChainID.DEFAULT_CHAIN_ID):
        # init byte_communicator
        MessengerFactory.ctx.communicator = MessengerFactory.pick_communicator(messenger_type)(
            bootstrap_server, party_name, other_party_names, fs_server=fs_server, task_chain_id=task_chain_id)

    @staticmethod
    def pick_communicator(messenger_type):
        communicator = {
            constant.MessengerType.KAFKA: KafkaMessenger,
        }.get(messenger_type, None)
        if communicator is None:
            Factory._raise_value_error('messenger_type', messenger_type)
        return communicator

    @staticmethod
    def get_global_instance():
        return MessengerFactory.ctx.communicator

    @staticmethod
    def get_instance(bootstrap_server,
                     messenger_type=constant.MessengerType.LIGHT_KAFKA,
                     task_chain_id=constant.TaskChainID.DEFAULT_CHAIN_ID):
        if messenger_type == constant.MessengerType.LIGHT_KAFKA:
            return LightKafkaMessenger(bootstrap_server, task_chain_id)
        else:
            Factory._raise_value_error('messenger_type', messenger_type)
