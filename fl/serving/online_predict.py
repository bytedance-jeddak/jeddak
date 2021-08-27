from fl.serving.model_loader import ModelLoader
from fl.algorithm import Algorithm
from fl.data_io.data_loader import DataLoader
from fl.data_io.data_saver import DataSaver
from common.frame.parameter_frame.online_predict_parameter import OnlinePredictParameter
from common.util import constant
from common.factory.serving_factory import ServingFactory


class OnlinePredict(Algorithm):
    models = {}
    skipped_algo = {
        DataLoader: True
    }

    def __init__(self, parameter: OnlinePredictParameter):
        super(OnlinePredict, self).__init__(parameter)
        self.model_id = parameter.model_id
        self.input_data = parameter.input_data

    def run_predict(self, input_data, input_model=None):
        models = ModelLoader.models[self.model_id]
        data_cache = input_data

        if self.get_this_party_role() == constant.TaskRole.GUEST:
            models[0].set_input_data_source(constant.DataSource.RAW)
            models[0].set_input_data_path(self.input_data)

        elif self.get_this_party_role() == constant.TaskRole.HOST:
            data_cache = ServingFactory.get_data_cache(self._task_chain_id)
            # if data_cache exist, get from ByteOnlinePredictFactory and skip data_loader and feature engineering
            if data_cache is not None:
                new_models = []
                for model in models:
                    if not OnlinePredict.skipped_algo.get(type(model), False):
                        new_models.append(model)
                models = new_models
            # otherwise, cache the data
            else:
                data_cache = None
                for (idx, model) in enumerate(models):
                    if OnlinePredict.skipped_algo.get(type(model), False):
                        data_cache = model.predict(data_cache)
                    else:
                        data_cache.cache()
                        ServingFactory.add_data_cache(self._task_chain_id, data_cache)
                        models = models[idx:]
                        break

        for model in models:
            self._logger.info("start {}".format(type(model)))
            if type(model) == DataSaver:
                continue
            data_cache = model.predict(data_cache)

        if self.get_this_party_role() == constant.TaskRole.GUEST:
            # deal with the result
            id2label = dict()
            for (id, sample) in data_cache.collect():
                id2label[id] = sample.label
            ServingFactory.add_labels(self._task_chain_id, id2label)
        return data_cache, None
