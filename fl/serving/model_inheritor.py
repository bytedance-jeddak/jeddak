import os

from common.frame.parameter_frame.model_inheritor_parameter import ModelInheritorParameter
from fl.algorithm import Algorithm


class ModelInheritor(Algorithm):
    models = {}

    def __init__(self, parameter: ModelInheritorParameter):
        super(ModelInheritor, self).__init__(parameter)
        self.model_id = parameter.model_id
        self.target_model = parameter.target_model

    def run_load_model(self, input_data=None, input_model=None):
        output_model = self.do_load_model(self.model_id)
        return input_data, output_model

    def do_load_model(self, id):
        # check model.idx
        # load each model
        # set to map
        root_dir = os.path.split(os.path.realpath(__file__))[0]
        model_path = os.path.join(root_dir, '..', '..', 'common', 'model', self._this_party, id)
        if not os.path.exists(model_path):
            raise Exception("not find model path {}".format(model_path))
        exist_flag = False
        model = None
        for file in os.listdir(model_path):
            if file.split(".")[-1] != "model":
                continue
            file = file.split(".")[0]
            model_file = os.path.join(model_path, file)
            model = Algorithm.load_model(model_file)
            if model.parameter.task_type == self.target_model:
                self._logger.info('ModelLoader do_load_model {} {}'.format(model.parameter.task_type,
                                                                           model_file))
                exist_flag = True
                break
        if exist_flag:
            return model
        else:
            raise ValueError("Not found target model: {} in path: {}".format(self.target_model, model_path))

