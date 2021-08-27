from fl.serving.model_loader import ModelLoader
from fl.algorithm import Algorithm
from fl.data_io.data_loader import DataLoader
from fl.data_io.data_saver import DataSaver
from common.frame.parameter_frame.data_saver_parameter import DataSaverParameter
from common.frame.parameter_frame.model_predict_parameter import ModelPredictParameter


class ModelPredict(Algorithm):
    models = {}
    
    def __init__(self, parameter: ModelPredictParameter):
        super(ModelPredict, self).__init__(parameter)
        self.model_id = parameter.model_id

    def run_predict(self, input_data, input_model=None):
        print(ModelLoader.models)
        models = ModelLoader.models[self.model_id]
        self._logger.info('ModelPredict run_predict')
        self._logger.info("load model id {}".format(self.model_id))
        self._logger.info(ModelLoader.models)
        print("load model id {}".format(self.model_id))
        print(ModelLoader.models)
        input_path = self._parameter.input_data_path 
        # output_path = self._parameter.output_data_path
        data_cache = input_data
        if type(models[0]) == DataLoader:
            print("set input_path " + input_path)
            models[0].set_input_data_path(input_path)
        # if type(models[-1]) == DataSaver:
        #     print("set output_path " + output_path)
        #     models[-1].set_output_data_path(output_path)
        # _, ss = SparkContextFactory.get_global_instance()
        for model in models:
            self._logger.info(type(model))
            data_cache = model.predict(data_cache)
            self._logger.info(type(data_cache))
            # ss.createDataFrame(data_cache.map(lambda x: [x[0]] + x[1].features.tolist()).sparkRDD).show()
        if type(models[-1]) != DataSaver:
            saver = DataSaver(parameter=DataSaverParameter())
            saver._all_parties = self._all_parties
            saver._other_parties = self._other_parties
            saver._this_party = self._this_party
            saver._task_chain_id = self._task_chain_id
            data_cache = saver.predict(data_cache)

        return data_cache, None
