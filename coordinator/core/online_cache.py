class OnlineCache(object):
    model_cache = dict()

    @staticmethod
    def add_cache(model_id, task_chain, task_role):
        OnlineCache.model_cache[model_id] = (task_chain, task_role)

    @staticmethod
    def get_cache(model_id):
        if model_id in OnlineCache.model_cache:
            return OnlineCache.model_cache.get(model_id)
        return None, None
