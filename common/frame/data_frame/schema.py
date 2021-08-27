class Schema(object):
    def __init__(self, id_name=None, feature_names=None, label_name=None):
        """

        :param id_name: str
        :param feature_names: List[str]
        :param label_name: str
        """
        if feature_names is None:
            feature_names = []
        self._id_name = id_name
        self._feature_names = feature_names
        self._label_name = label_name

    def __str__(self):
        return "id: {}, features: {}, label: {}".format(
            self._id_name, self._feature_names, self._label_name
        )

    @property
    def id_name(self):
        return self._id_name

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def label_name(self):
        return self._label_name

    @property
    def id_feature_names(self):
        return [self._id_name] + self._feature_names

    @id_name.setter
    def id_name(self, id_name):
        self._id_name = id_name

    @feature_names.setter
    def feature_names(self, feature_names):
        self._feature_names = feature_names

    @label_name.setter
    def label_name(self, label_name):
        self._label_name = label_name

    def get_feature_index(self, f_name):
        """
        Get feature index by its name
        :param f_name: str
        :return:
        """
        return self._feature_names.index(f_name)

    def __add__(self, other):
        assert type(other) == Schema

        new_id_name = self._id_name or other.id_name
        new_feature_names = self._feature_names + other.feature_names
        new_label_name = self._label_name or other.label_name

        return Schema(new_id_name, new_feature_names, new_label_name)

    def __mul__(self, other):
        assert type(other) == Schema

        new_id_name = self._id_name or other.id_name

        new_feature_names = []
        for f_name in self._feature_names:
            for of_name in other.feature_names:
                new_feature_names.append(f_name + of_name)

        new_label_name = self._label_name or other.label_name

        return Schema(new_id_name, new_feature_names, new_label_name)
