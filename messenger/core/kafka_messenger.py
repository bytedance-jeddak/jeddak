import pickle

from common.factory.logger_factory import LoggerFactory
from common.util import constant
from messenger.core.light_kafka_messenger import LightKafkaMessenger


def cut_text(text):
    """
    cut text with a maximum size of partition
    :param text: text to handle
    :return: list of text partitions
    """
    max_msg_size = constant.MessageThreshold.MAX_MESSAGE_SIZE
    total_length = len(text)
    final_slice = []
    prev_idx = 0
    post_idx = max_msg_size
    while True:
        if post_idx >= total_length:
            final_slice.append((prev_idx, total_length))
            break
        final_slice.append((prev_idx, post_idx))
        prev_idx = post_idx
        post_idx += max_msg_size
    return final_slice


class KafkaMessenger(LightKafkaMessenger):
    """
    A messenger supports both general variable
    """

    def __init__(self,
                 bootstrap_server,
                 party_name,
                 other_party_names,
                 fs_server=None,
                 task_chain_id=constant.TaskChainID.DEFAULT_CHAIN_ID):
        """
        init KafkaMessenger
        :param bootstrap_server: host, port and password(optional) of kafka broker
        :param party_name: name of current party
        :param other_party_names: names of other parties
        :param fs_server: not used
        :param task_chain_id: unique task id
        """
        super(KafkaMessenger, self).__init__(bootstrap_server, task_chain_id)
        self._task_chain_id = str(task_chain_id)
        self._party_name = party_name
        self._other_party_names = other_party_names

        self._produce_topics = self.generate_produce_topics()
        self._consume_topics = self.generate_consume_topics()
        if self._produce_topics:
            self.create_topics(self._produce_topics)
        self.wait_topics(self._consume_topics)

        self._logger = LoggerFactory.get_global_instance()

    @property
    def task_chain_id(self):
        return self._task_chain_id

    def send(self, data, tag: str, suffix=None, parties=None, topics=None):
        """
        unified interface to send data
        :param data: message
        :param tag: tag type
        :param suffix: suffix to concat message key
        :param parties: parties to send to
        :param topics: if topics is not None, send to {topics} only.
        """
        if topics is not None:
            target_topics = topics
        elif parties is None:
            target_topics = self._produce_topics
        else:
            target_topics = self.generate_produce_topics(parties=parties)
        message_tag = self.generate_message_tag(tag, suffix)

        # sends general variables, there goes
        #    ($message_tag:$batch_index/$batch_num, message),
        #    e.g., ('loss:0/5', message)
        self.produce_by_kv(
            message_tag,
            data,
            target_topics,
            data_type="a variable")

    def receive(self, tag: str, suffix=None, parties=None, topics=None):
        """
        unified interface to receive data
        :param tag: tag type
        :param suffix: suffix to concat message key
        :param parties: parties to receive from
        :param topics: if topics is not None, receive from {topics} only.
        :return: list<message>
        """
        if topics is not None:
            target_topics = topics
        elif parties is None:
            target_topics = self._consume_topics
        else:
            target_topics = self.generate_consume_topics(parties=parties)
        message_tag = self.generate_message_tag(tag, suffix)

        # final_data: the list of data, which will be returned
        final_data = []
        for idx in range(len(target_topics)):
            target_topic = target_topics[idx]
            self.switch_consumer_topic(target_topic)

            # receive data from each party
            key, data = self.consume_kv()
            msg_tag, _, _ = KafkaMessenger.parse_byte_dataset_tag(key)
            assert msg_tag == message_tag

            self._logger.info("received: {}".format(message_tag))
            assert data is not None
            final_data.append(data)

        return final_data

    @staticmethod
    def parse_byte_dataset_tag(tag):
        """
        parse tag received from kafka
        :param tag: tag type
        :return: message_tag, cur_batch_index, total_batch_num
        """
        message_tag, batch_info = tag.split(':')
        cur_batch_index, total_batch_num = batch_info.split('/')
        return message_tag, int(cur_batch_index), int(total_batch_num)

    def switch_consumer_topic(self, new_topic):
        """
        switch another topic to consume
        :param new_topic: new kafka topic
        """
        if self._current_topic == new_topic:
            return
        self.assign(new_topic)
        self._current_topic = new_topic

    def generate_produce_topics(self, parties=None):
        """
        concat topics to produce
        :param parties: parties to send message
        :return: topics
        """
        chosen_party_names = parties if parties is not None else self._other_party_names
        return ['.'.join((self._task_chain_id, self._party_name, p))
                for p in chosen_party_names]

    def generate_consume_topics(self, parties=None):
        """
        concat topics to consume
        :param parties: parties to receive message
        :return: topics
        """
        chosen_party_names = parties if parties is not None else self._other_party_names
        return ['.'.join((self._task_chain_id, p, self._party_name))
                for p in chosen_party_names]

    def produce_by_kv(self, message_tag, data, topics, data_type=""):
        """
        produce message
        :param message_tag: tag to concat message key
        :param data: message
        :param topics: topics to send to
        :param data_type: to print logs
        """
        value = pickle.dumps(data)
        value_parts = cut_text(value)
        num_of_parts = len(value_parts)
        idx = 0
        for (prev_idx, post_idx) in value_parts:
            for tt in topics:
                self._producer.produce(topic=tt, key=pickle.dumps(
                    "{}:{}/{}".format(message_tag, idx, num_of_parts)), value=value[prev_idx:post_idx])
                self._producer.poll(0)
            idx += 1
        self._logger.info("sent {}: {}".format(data_type, message_tag))

    def consume_kv(self):
        """
        consume message
        :return: message received
        """
        final_value = bytes()
        while True:
            msg = self._consumer.poll(1.0)
            if msg is None or msg.key() is None:
                continue
            key = pickle.loads(msg.key())
            final_value += msg.value()
            _, curr, total = KafkaMessenger.parse_byte_dataset_tag(key)
            if curr == total - 1:
                return key, pickle.loads(final_value)
