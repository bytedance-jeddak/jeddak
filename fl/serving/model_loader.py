from common.factory.m2i_factory import Model2InstanceFactory
import os

from common.frame.parameter_frame.model_load_parameter import ModelLoadParameter
from common.util.constant import ModelLoaderAction
from fl.algorithm import Algorithm


class ModelLoader(Algorithm):
    models = {}

    def __init__(self, parameter: ModelLoadParameter):
        super(ModelLoader, self).__init__(parameter)
        self.model_id = parameter.model_id
        self.action = parameter.action

    def run_load_model(self, input_data=None, input_model=None):
        if self.action == ModelLoaderAction.LOAD_MODEL:
            self.do_load_model(self.model_id)
        elif self.action == ModelLoaderAction.UNLOAD_MODEL:
            self.do_unload_model(self.model_id)
        else:
            raise Exception('run_load_model unkown action')
        return input_data, input_model

    @staticmethod
    def get_model_by_id(id):
        return ModelLoader.models[id]

    def do_load_model(self, id):
        # check model.idx
        # load each model
        # set to map
        root_dir = os.path.split(os.path.realpath(__file__))[0]
        model_path = os.path.join(root_dir, '..', '..', 'common', 'model', self._this_party, id)
        model_idx_file_name = os.path.join(model_path, id + '.idx')
        models = []
        if not os.path.exists(model_idx_file_name):
            raise Exception("not find model idx {}".format(model_idx_file_name))
        with open(model_idx_file_name, 'r') as f:
            for line in f.readlines():
                ss = line.split(":")
                model_file_name = ss[0]
                model_type = ss[1][:-1]
                model_file_name = os.path.join(model_path, model_file_name)
                self._logger.info('ModelLoader do_load_model {} {}'.format(model_type, model_file_name))

                model = Algorithm.load_model(model_file_name)
                # create an instance from model
                algorithm_instance = Model2InstanceFactory.get_instance(model_type, model)
                self._logger.info("model_loader instantiate algorithm {}".format(str(algorithm_instance)))
                models.append(algorithm_instance)
        ModelLoader.models[id] = models
        print(models)

    def do_unload_model(self, id):
        # delete in map
        self._logger.info('ModelLoader do_unload_model {}'.format(id))
        del ModelLoader.models[id]
