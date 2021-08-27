from time import sleep


class ServingFactory(object):
    # map<task_chain_id, id2label>
    online_label = dict()

    # map<task_chain_id, byte_dataset>
    online_data_cache = dict()

    @staticmethod
    def add_labels(task_chain_id, id2label):
        ServingFactory.online_label[task_chain_id] = id2label

    @staticmethod
    def get_labels(task_chain_id, internal=0.1, max_try=1000):
        for _ in range(max_try):
            if task_chain_id in ServingFactory.online_label:
                return ServingFactory.online_label[task_chain_id]
            sleep(internal)
        return None

    @staticmethod
    def add_data_cache(task_chain_id, data_cache):
        ServingFactory.online_data_cache[task_chain_id] = data_cache

    @staticmethod
    def get_data_cache(task_chain_id):
        return ServingFactory.online_data_cache.get(task_chain_id, None)
