from common.factory.df_factory import DataFrameFactory


class Histogram(object):
    def __init__(self, freqs):
        """

        :param freqs: List
        """
        self._freqs = freqs

    def __str__(self):
        return str(self._freqs)

    def __add__(self, other):
        assert type(other) == Histogram or other == 0

        if other == 0:
            return self

        new_freqs = [None for _ in range(len(self._freqs))]

        for idx in range(len(self._freqs)):
            new_freqs[idx] = self._freqs[idx] + other.freqs[idx]

        return Histogram(new_freqs)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        assert isinstance(other, Histogram) or other == 0

        if other == 0:
            return self

        new_freqs = [None for _ in range(len(self._freqs))]

        for idx in range(len(self._freqs)):
            new_freqs[idx] = self._freqs[idx] - other.freqs[idx]

        return Histogram(new_freqs)

    def __rsub__(self, other):
        return other - self

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._freqs[item]
        else:
            return self._freqs[int(item)]

    def __setitem__(self, key, value):
        self._freqs[key] = value

    def __len__(self):
        return len(self._freqs)

    @property
    def freqs(self):
        return self._freqs

    @property
    def total_freqs(self):
        return sum(self._freqs)

    @total_freqs.setter
    def total_freqs(self, total_freqs):
        """
        Truncate the freqs to meet total_freqs
        """
        chist = self.to_cumulative_histogram()
        chist.total_freqs = total_freqs
        self._freqs = chist.to_histogram().freqs

    @staticmethod
    def generate(num_freq, element_type=None):
        """

        :param num_freq:
        :param element_type: float or int if not specified
        :return:
        """
        if not element_type:
            freqs = [0.0 for _ in range(num_freq)]

        else:
            data_frame_class = DataFrameFactory.get_instance(data_frame_type=element_type)
            freqs = [data_frame_class.get_identity() for _ in range(num_freq)]

        return Histogram(freqs=freqs)

    def grow(self, value, index):
        """
        Add value to the index-th interval (bucket)
        :param value:
        :param index:
        :return:
        """
        self._freqs[index] += value

    def to_cumulative_histogram(self):
        cum_freqs = None
        for freq in self._freqs:
            if cum_freqs is None:
                cum_freqs = [freq]
            else:
                cum_freqs.append(cum_freqs[-1] + freq)

        return CumulativeHistogram(cum_freqs)


class CumulativeHistogram(Histogram):
    @property
    def total_freqs(self):
        return self._freqs[-1]

    @total_freqs.setter
    def total_freqs(self, total_freqs):
        """
        Truncate the freqs to meet total_freqs
        """
        trunc_idx = len(self._freqs)
        for idx in range(len(self._freqs)):
            if self._freqs[idx] > total_freqs:
                trunc_idx = idx
                self._freqs[idx] = total_freqs
                break
        self._freqs = self._freqs[:trunc_idx]

    def grow(self, value, index):
        for i in range(index, len(self._freqs)):
            self._freqs[i] += value

    def to_histogram(self):
        """
        To native histogram
        :return:
        """
        temp_freq = 0
        new_freqs = []

        for freq in self._freqs:
            new_freqs.append(freq - temp_freq)
            temp_freq = freq

        return Histogram(new_freqs)
