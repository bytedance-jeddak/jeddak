import os

from common.factory.logger_factory import LoggerFactory
from common.factory.messenger_factory import MessengerFactory
from common.frame.message_frame.message import Message
from common.util import constant
from joblib import dump, load


class Algorithm(object):
    """
    The algorithm base class
    """

    def __init__(self, parameter, message=Message()):
        """

        :param parameter: from common/frame/parameter_frame
        :param message: from common/frame/message_frame
                        Some modules do not need messengers, and thus can go with a default argument
        """
        self._parameter = parameter

        # get global components
        self._logger = LoggerFactory.get_global_instance()
        self._messenger = MessengerFactory.get_global_instance()

        # init message list
        self._message = message

        # set guest and host party names
        self._all_parties = None
        self._other_parties = None
        self._this_party = None

        # task id
        self._task_chain_id = None
        self._task_id = None

        # placeholders
        self._evaluator = None

        self._save_model = True

        # for validation e.g. GLM, NN
        self.validate_data = None

    @property
    def task_type(self):
        return self._parameter.task_type

    def train(self, input_data=None, input_model=None):
        """
        Training method
        :param input_data: DDataset
        :param input_model: Model
        :return: (DDataset: (str, Sample), Model)
        """
        pass

    def predict(self, input_data=None):
        """
        Predicting method
        :param input_data: features
        :return: DDataset: (str, Sample)
        """
        pass

    @staticmethod
    def model_to_instance(model):
        """
        Create an instance from model such that instance.predict() works
        :param model: Model
        :return: Algorithm
        """
        pass

    def instance_to_model(self):
        """
        Create a model from this instance, for further saving
        :return: Model
        """
        pass

    def evaluate(self, input_data, pred_score):
        """

        :param input_data: DDataset, ((id, Sample))
        :param pred_score: DDataset, ((id, Sample))
        :return:
        """
        self._logger.info("start evaluating")

        if self.get_this_party_role() == constant.TaskRole.GUEST:
            pred_score = pred_score.mapValues(lambda sample: sample.label)
            y = input_data.mapValues(lambda val: val.label)
            metrics = self._evaluator.eval(pred_score, y)
            self._logger.info("metrics: {}".format(metrics))

        elif self.get_this_party_role() == constant.TaskRole.HOST:
            return

        else:
            raise ValueError("invalid task role: {}".format(self.get_this_party_role()))

        self._logger.info("end evaluating")

        return metrics

    def validate(self, input_data=None, evaluate=True):
        metric = None

        if input_data is None:
            if self.validate_data is None:
                return None, metric
            input_data = self.validate_data

        validate_res = self.predict(input_data)

        if evaluate:
            metric = self.evaluate(input_data, validate_res)

        return validate_res, metric

    def set_save_model(self, save):
        """
        param save: boolean
        """
        self._save_model = save

    def save_model(self, model):
        """
        Save model to common/model/${task_chain_id}/${task_id}.model
        :param model: Model
        :return:
        """
        # get model file name
        root_dir = os.path.split(os.path.realpath(__file__))[0]
        model_path = os.path.join(root_dir, '..', 'common', 'model', self._this_party, self._task_chain_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_file_name = os.path.join(model_path, self._task_id + '.model')

        # save model to disk
        dump(model, model_file_name)

    @staticmethod
    def load_model(task_id):
        """
        Load model to create an algorithm instance from common/model/${task_chain_id}/${task_id}.model
        :param task_id: str
        :return: Algorithm
        """
        # get model file name
        task_chain_id = task_id.split('-')[0]

        root_dir = os.path.split(os.path.realpath(__file__))[0]
        model_path = os.path.join(root_dir, '..', 'common', 'model', task_chain_id)
        model_file_name = os.path.join(model_path, task_id + '.model')
        if not os.path.exists(model_file_name):
            raise Exception("Algorithm load_model not find model {}".format(model_file_name))
        # load mode from disk
        model = load(model_file_name)

        return model

    def _allocate_task(self,
                       guest_task=None, host_task=None,
                       server_task=None, client_task=None,
                       sole_task=None, slack_task=None):
        """
        Allocate the task depending on the task role guest or host
        :param guest_task: func
        :param host_task: func
        :param server_task: func
        :param client_task: func
        :param sole_task: func
        :param slack_task: func
        :return:
        """
        if self.get_this_party_role() == constant.TaskRole.GUEST:
            return guest_task()
        elif self.get_this_party_role() == constant.TaskRole.HOST:
            return host_task()
        elif self.get_this_party_role() == constant.TaskRole.SERVER:
            return server_task()
        elif self.get_this_party_role() == constant.TaskRole.CLIENT:
            return client_task()
        elif self.get_this_party_role() == constant.TaskRole.SOLE:
            return sole_task()
        elif self.get_this_party_role() == constant.TaskRole.SLACK:
            return slack_task()
        else:
            raise ValueError("invalid task role: {}".format(self.get_this_party_role()))

    def set_party_info(self, all_parties, other_parties, this_party):
        self._all_parties = all_parties
        self._other_parties = other_parties
        self._this_party = this_party

    def get_party_info(self):
        return self._all_parties, self._other_parties, self._this_party

    def set_task_id(self, task_id):
        self._task_id = task_id
        if self._task_id:
            self._task_chain_id = self._task_id.split('-')[0]

    def get_this_party_role(self):
        """
        Get this party's role
        :return: str
        """
        this_party_idx = self._all_parties.index(self._this_party)
        this_party_role = self._parameter.task_role[this_party_idx]
        return this_party_role

    def get_party_num(self):
        return len(self._all_parties)

    def get_all_guest_names(self):
        all_guest_names = []
        for role, party_name in zip(self._parameter.task_role, self._all_parties):
            if role == constant.TaskRole.GUEST:
                all_guest_names.append(party_name)
        return all_guest_names

    def get_other_guest_names(self):
        other_guest_names = self.get_all_guest_names()
        try:
            other_guest_names.remove(self._this_party)
        except ValueError:
            pass
        return other_guest_names

    def get_all_host_names(self):
        all_host_names = []
        for role, party_name in zip(self._parameter.task_role, self._all_parties):
            if role == constant.TaskRole.HOST:
                all_host_names.append(party_name)
        return all_host_names

    def get_other_host_names(self):
        other_host_names = self.get_all_host_names()
        try:
            other_host_names.remove(self._this_party)
        except ValueError:
            pass
        return other_host_names

    def get_evaluate_type(self):
        return self._evaluator._evaluation_type

    def get_all_server_names(self):
        all_server_names = []
        for role, party_name in zip(self._parameter.task_role, self._all_parties):
            if role == constant.TaskRole.SERVER:
                all_server_names.append(party_name)
        return all_server_names

    def get_other_server_names(self):
        other_server_names = self.get_all_server_names()
        try:
            other_server_names.remove(self._this_party)
        except ValueError:
            pass
        return other_server_names

    def get_all_client_names(self):
        all_client_names = []
        for role, party_name in zip(self._parameter.task_role, self._all_parties):
            if role == constant.TaskRole.CLIENT:
                all_client_names.append(party_name)
        return all_client_names

    def get_other_client_names(self):
        other_client_names = self.get_all_client_names()
        try:
            other_client_names.remove(self._this_party)
        except ValueError:
            pass
        return other_client_names

    def get_absent_client_names(self, clients):
        """
        Get the clients absent in the input client list
        :param clients: List
        :return:
        """
        return list(set(self.get_all_client_names()) - set(clients))

    def set_validate_data(self, validate_data):
        self.validate_data = validate_data
