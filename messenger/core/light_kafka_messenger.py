import os
import time
import pickle

from confluent_kafka.admin import AdminClient
from confluent_kafka.cimpl import Producer, Consumer, NewTopic, TopicPartition

from common.util import constant
from common.util.random import Random
from messenger.core.messenger import Messenger


class LightKafkaMessenger(Messenger):
    """
    A light messenger that supports only one-way general variable transmission and does not yield logging
    """

    def __init__(self, bootstrap_server,
                 task_chain_id=constant.TaskChainID.DEFAULT_CHAIN_ID):
        """
        init LightKafkaMessenger
        :param bootstrap_server: host, port and password(optional) of kafka broker
        :param task_chain_id: used as default
        """
        super(LightKafkaMessenger, self).__init__(bootstrap_server)

        bs = bootstrap_server.split(":")
        messenger_url = "{}:{}".format(bs[0], bs[1])
        messenger_passwd = bs[2] if len(bs) == 3 else None

        # producer
        producer_conf = {
            'bootstrap.servers': messenger_url
        }
        self._producer = Producer(
            LightKafkaMessenger.parse_messenger_conf(
                producer_conf, messenger_passwd))
        assert LightKafkaMessenger.is_connected(self._producer)

        # consumer
        consumer_conf = {
            'bootstrap.servers': messenger_url,
            'group.id': Random.generate_random_digits(),
            'auto.offset.reset': 'earliest',
            'max.poll.interval.ms': 86400000
        }
        self._consumer = Consumer(
            LightKafkaMessenger.parse_messenger_conf(
                consumer_conf, messenger_passwd))
        assert LightKafkaMessenger.is_connected(self._consumer)

        # client
        client_conf = {
            'bootstrap.servers': messenger_url
        }
        self._client = AdminClient(
            LightKafkaMessenger.parse_messenger_conf(
                client_conf, messenger_passwd))

        # topic
        self._topic = str(task_chain_id)
        self.create_topics([self._topic])
        self.assign(self._topic)
        self._current_topic = self._topic

    @property
    def producer(self):
        return self._producer

    def clear_all(self):
        """
        clear all topics, except for current
        """
        all_topics = list(self._client.list_topics().topics.keys())
        if len(all_topics) > 0:
            self._client.delete_topics(all_topics)
            self._client.create_topics(
                [NewTopic(topic=self._topic, num_partitions=1)])

    def send(self, data, tag: str, suffix=None):
        """
        send a general variable
        :param data: message
        :param tag: tag type
        :param suffix: suffix to concat message key
        :return: True if sending succeeds
        """
        # prepare key and data
        message_tag = self.generate_message_tag(tag, suffix)
        key = pickle.dumps(message_tag)
        value = pickle.dumps(data)

        # produce the message
        self._producer.produce(topic=self._topic, key=key, value=value)
        self._producer.poll(0)

        return True

    def receive(self, tag: str, suffix=None):
        """
        receive a general variable
        :param tag: tag type
        :param suffix: suffix to concat message key
        :return: message received
        """
        message_tag = self.generate_message_tag(tag, suffix)
        value = None

        while True:
            msg = self._consumer.poll(1.0)
            if msg is None or msg.key() is None:
                continue
            key = pickle.loads(msg.key())
            value = pickle.loads(msg.value())
            if key == message_tag:
                break

        return value

    @staticmethod
    def parse_messenger_conf(conf, passwd):
        """
        parse config json of kafka
        :param conf: default config
        :param passwd: password provided by users
        :return: config json of kafka
        """
        if passwd is not None:
            conf['sasl.username'] = constant.SecurityConfig.SASL_USERNAME
            conf['sasl.password'] = passwd
            conf['sasl.mechanisms'] = constant.SecurityConfig.SASL_MECHANISMS

            # check the existence of ca cert
            ca_file = os.path.join(
                os.path.split(
                    os.path.realpath(__file__))[0],
                '../..',
                'common',
                'ca-cert')
            if os.path.isfile(ca_file):
                conf['ssl.endpoint.identification.algorithm'] = 'none'
                conf['ssl.ca.location'] = ca_file
                conf['security.protocol'] = constant.SecurityConfig.SASL_SSL
            else:
                conf['security.protocol'] = constant.SecurityConfig.SASL_PLAINTEXT
        return conf

    @staticmethod
    def is_connected(connect_ob):
        """
        check the connection to kafka
        :param connect_ob: connection object
        :return: True if connected, otherwise False
        """
        try:
            cur_list = connect_ob.list_topics(timeout=3)
        except BaseException:
            raise TimeoutError("Kafka initial connection is timed out")
        return True if cur_list else False

    def create_topics(self, topics):
        """
        create kafka topic asynchronously
        :param topics: topics to create
        """
        new_topics = set(topics) - set(self._client.list_topics().topics)
        if len(new_topics) == 0:
            return
        self._client.create_topics(
            [NewTopic(topic=t, num_partitions=1) for t in new_topics])

    def create_topics_sync(self, topics, retry=5):
        """
        create kafka topic synchronously. when the creation fails, throw an exception
        :param topics: topics to create
        :param retry: times to retry
        """
        new_topics = set(topics) - set(self._client.list_topics().topics)
        if len(new_topics) == 0:
            return
        self._client.create_topics(
            [NewTopic(topic=t, num_partitions=1) for t in new_topics])
        tries = 0
        while tries < retry:
            tries += 1
            if len(new_topics - set(self._client.list_topics().topics)) == 0:
                return
            time.sleep(0.1)
        raise Exception("messenger: fail to create topic")

    def flush(self):
        """
        flush data queue locally
        """
        self._producer.flush()

    def assign(self, topic):
        """
        assign to a topic
        :param topic: topic to assign
        """
        self._consumer.assign([TopicPartition(topic=topic, partition=0)])

    def wait_topics(self, topics, retry=5):
        """
        wait kafka topics until created
        :param topics: topics to create
        :param retry: times to retry
        """
        tries = 0
        while tries < retry:
            tries += 1
            if len(set(topics) - set(self._client.list_topics().topics)) == 0:
                print('kafka ensures topics({}) created'.format(topics))
                return
            print("wait_topics retry.{}".format(tries))
            time.sleep(2)
        print("warning: wait topics fails")
