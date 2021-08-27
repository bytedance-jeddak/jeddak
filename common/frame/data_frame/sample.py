import numpy as np


class Sample(object):
    def __init__(self, features=None, label=None):
        """

        :param features: numpy.ndarray
        :param label:
        """
        if features is None:
            features = []
        if type(features) is np.ndarray:
            self._features = features
        elif type(features) is tuple or type(features) is list:
            self._features = np.asarray(features)
        else:
            raise TypeError("sample class only supports numpy.ndarray, tuple and list features")
        self._label = label

    def __str__(self):
        return "features: {}, label: {}".format(self._features, self._label)

    @property
    def features(self):
        return self._features

    @property
    def label(self):
        return self._label

    @features.setter
    def features(self, features):
        self._features = features

    @label.setter
    def label(self, label):
        self._label = label
