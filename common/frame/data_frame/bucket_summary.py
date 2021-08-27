class BucketSummary(object):
    def __init__(self, splits):
        """

        :param splits: List
        """
        self._splits = splits

    def __str__(self):
        return str(self._splits)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._splits[item]
        else:
            return self._splits[int(item)]

    def get_num_bucket(self):
        return len(self._splits) - 1

    def mask(self):
        """
        Replace all splits with default zeros
        :return:
        """
        for i in range(len(self._splits)):
            self._splits[i] = 0.0
