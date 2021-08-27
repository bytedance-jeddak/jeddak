import copy
import functools

from common.util import constant

import numpy as np

import scipy.integrate as integrate


class Distribution(object):
    APPROX_ZERO = 1e-5
    DEFAULT_SPLITS = np.linspace(-5.0, 5.0, 11).tolist()

    def get_pdf_at(self, x):
        """
        Get the value of f(x)
        :param x:
        :return:
        """
        pass

    def sample(self):
        """
        Sample to get a value
        :return:
        """
        pass

    def __mul__(self, other):
        """
        ZeroMeanExactDistribution * ApproximateDistribution or ApproximateDistribution * ZeroMeanExactDistribution
        The theory refers to https://en.wikipedia.org/wiki/Product_distribution, where X, Y denote the approximate
            and the exact one, respectively.
        :param other:
        :return: ApproximateDistribution
        """
        def fy_1_x(x, z, zero_mean_exact_distribution):
            fy = zero_mean_exact_distribution.get_pdf_at(z / x)
            return fy * 1 / np.abs(x)

        if type(self) == ZeroMeanExactDistribution and type(other) == ApproximateDistribution:
            prod_freqs = []

            for freq_idx in range(len(self.DEFAULT_SPLITS) - 1):
                prod_freq = 0.0
                func = functools.partial(fy_1_x,
                                         z=(self.DEFAULT_SPLITS[freq_idx] + self.DEFAULT_SPLITS[freq_idx + 1]) / 2,
                                         zero_mean_exact_distribution=self)

                for int_idx in range(other.get_num_interval()):
                    endpoints = other.get_endpoints_at_index(int_idx)
                    pmf = other.get_pmf_at_index(int_idx)
                    if endpoints[0] < 0 < endpoints[1]:
                        int_val = integrate.quad(func=func, a=endpoints[0], b=-self.APPROX_ZERO)[0] + \
                                  integrate.quad(func=func, a=self.APPROX_ZERO, b=endpoints[1])[0]
                    else:
                        int_val = integrate.quad(func=func,
                                                 a=endpoints[0] or self.APPROX_ZERO,
                                                 b=endpoints[1] or -self.APPROX_ZERO)[0]
                    prod_freq += pmf * int_val

                prod_freqs.append(prod_freq)

            return ApproximateDistribution(splits=copy.deepcopy(self.DEFAULT_SPLITS), freqs=prod_freqs)

        elif type(self) == ApproximateDistribution and type(other) == ZeroMeanExactDistribution:
            return other * self

        else:
            raise TypeError("invalid types: {}, {}".format(type(self), type(other)))


class ExactDistribution(Distribution):
    pass


class ZeroMeanExactDistribution(ExactDistribution):
    VAR_INFERIOR = 1e-3
    DISTRIBUTION_TYPES = (constant.DistributionType.GAUSSIAN,
                          constant.DistributionType.LAPLACE)

    def __init__(self, parameter, distribution_type=constant.DistributionType.GAUSSIAN):
        """

        :param parameter: dict
        :param distribution_type: str
        """
        self._parameter = parameter
        self._distribution_type = distribution_type

        self._step_size = 1

    @property
    def distribution_type(self):
        return self._distribution_type

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, step_size):
        self._step_size = step_size

    def sample(self):
        if self._distribution_type == constant.DistributionType.UNIFORM:
            a = self._parameter['a']
            return np.random.uniform(-a, a)

        elif self._distribution_type == constant.DistributionType.GAUSSIAN:
            mu, sigma = self._parameter['mu'], self._parameter['sigma']
            return np.random.normal(mu, sigma)

        elif self._distribution_type == constant.DistributionType.LAPLACE:
            mu, b = self._parameter['mu'], self._parameter['b']
            return np.random.laplace(mu, b)

        else:
            raise ValueError("invalid distribution_type: {}".format(self._distribution_type))

    def get_pdf_at(self, x):
        if self._distribution_type == constant.DistributionType.UNIFORM:
            a = self._parameter['a']
            if x < a or x > a:
                return self.APPROX_ZERO
            else:
                return 0.5 * a

        elif self._distribution_type == constant.DistributionType.GAUSSIAN:
            mu, sigma = self._parameter['mu'], self._parameter['sigma']
            return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * np.square((x - mu) / sigma))

        elif self._distribution_type == constant.DistributionType.LAPLACE:
            mu, b = self._parameter['mu'], self._parameter['b']
            return 1 / (2 * b) * np.exp(-np.abs(x - mu) / b)

        else:
            raise ValueError("invalid distribution_type: {}".format(self._distribution_type))

    def get_variance(self):
        if self._distribution_type == constant.DistributionType.UNIFORM:
            a = self._parameter['a']
            return np.square(a) / 3

        elif self._distribution_type == constant.DistributionType.GAUSSIAN:
            sigma = self._parameter['sigma']
            return np.square(sigma)

        elif self._distribution_type == constant.DistributionType.LAPLACE:
            b = self._parameter['b']
            return 2 * np.square(b)

        else:
            raise ValueError("invalid distribution_type: {}".format(self._distribution_type))

    def raise_variance(self):
        if self._distribution_type == constant.DistributionType.UNIFORM:
            a = self._parameter['a']
            da = np.sqrt(np.square(a) + 3 * self._step_size) - a
            self._parameter['a'] += da
            return True

        elif self._distribution_type == constant.DistributionType.GAUSSIAN:
            sigma = self._parameter['sigma']
            ds = np.sqrt(np.square(sigma) + self._step_size) - sigma
            self._parameter['sigma'] += ds
            return True

        elif self._distribution_type == constant.DistributionType.LAPLACE:
            b = self._parameter['b']
            db = np.sqrt(np.square(b) + 0.5 * self._step_size) - b
            self._parameter['b'] += db
            return True

        else:
            raise ValueError("invalid distribution_type: {}".format(self._distribution_type))

    def lower_variance(self):
        if self._distribution_type == constant.DistributionType.UNIFORM:
            a = self._parameter['a']
            delta = np.square(a) - 3 * self._step_size
            if delta < 0:
                return False
            da = a - np.sqrt(delta)
            self._parameter['a'] -= da
            return True

        elif self._distribution_type == constant.DistributionType.GAUSSIAN:
            sigma = self._parameter['sigma']
            delta = np.square(sigma) - self._step_size
            if delta < 0:
                return False
            ds = sigma - np.sqrt(delta)
            self._parameter['sigma'] -= ds
            return True

        elif self._distribution_type == constant.DistributionType.LAPLACE:
            b = self._parameter['b']
            delta = np.square(b) - 0.5 * self._step_size
            if delta < 0:
                return False
            db = b - np.sqrt(delta)
            self._parameter['b'] -= db
            return True

        else:
            raise ValueError("invalid distribution_type: {}".format(self._distribution_type))

    def copy(self, variance=None):
        if variance is None:
            variance = self.get_variance()

        if self._distribution_type == constant.DistributionType.UNIFORM:
            parameter = {'a': np.sqrt(3 * variance)}

        elif self._distribution_type == constant.DistributionType.GAUSSIAN:
            parameter = {'mu': 0.0, 'sigma': np.sqrt(variance)}

        elif self._distribution_type == constant.DistributionType.LAPLACE:
            parameter = {'mu': 0.0, 'b': np.sqrt(0.5 * variance)}

        else:
            raise ValueError("invalid distribution_type: {}".format(self._distribution_type))

        return ZeroMeanExactDistribution(parameter, self._distribution_type)

    def copy_inferior(self):
        return self.copy(self.VAR_INFERIOR)

    @staticmethod
    def generate_inferior():
        """
        Randomly pick a zero-mean small-variance distribution
        :return:
        """
        return ZeroMeanExactDistribution.generate(ZeroMeanExactDistribution.VAR_INFERIOR)

    @staticmethod
    def generate_standard():
        """
        Randomly pick a zero-mean one-variance distribution
        :return:
        """
        return ZeroMeanExactDistribution.generate(1.0)

    @staticmethod
    def generate(variance):
        """
        Randomly pick a zero-mean distribution with the specified variance
        :param variance:
        :return:
        """
        distribution_type = np.random.choice(ZeroMeanExactDistribution.DISTRIBUTION_TYPES)

        if distribution_type == constant.DistributionType.UNIFORM:
            parameter = {'a': np.sqrt(3 * variance)}

        elif distribution_type == constant.DistributionType.GAUSSIAN:
            parameter = {'mu': 0.0, 'sigma': np.sqrt(variance)}

        elif distribution_type == constant.DistributionType.LAPLACE:
            parameter = {'mu': 0.0, 'b': np.sqrt(0.5 * variance)}

        else:
            raise ValueError("invalid distribution_type: {}".format(distribution_type))

        return ZeroMeanExactDistribution(parameter, distribution_type)


class ApproximateDistribution(Distribution):
    """
    An approximate probability distribution
    """
    TOLERANCE = 1e-1

    def __init__(self, splits, freqs):
        """

        :param splits: List[float]
        :param freqs: List[float] with length less than len(splits) by one. Unnecessary to normalize
        """
        average_interval = self._check_equal_split_interval(splits)
        if type(average_interval) != bool:
            self._splits = splits
            self._average_interval = average_interval
        else:
            raise ValueError("unequal split intervals")

        self._pdf = self._scale_freqs(freqs)

    def __str__(self):
        return "ave_interval: {}, splits: {}, pdf: {}".format(self._average_interval, self._splits, self._pdf)

    @property
    def splits(self):
        return self._splits

    @property
    def pdf(self):
        """

        :return: pdf array
        """
        return self._pdf

    @property
    def average_interval(self):
        return self._average_interval

    def get_pdf_at(self, x):
        if x < self._splits[0] or x > self._splits[-1]:
            return 0.0

        pdf_idx = self.binary_search(self._pdf, x)
        return self._pdf[pdf_idx]

    def get_rep_var_at_index(self, idx):
        """
        Get the representative variable (middle point) at the idx-th interval
        :param idx:
        :return:
        """
        return (self._splits[idx] + self._splits[idx + 1]) / 2

    def get_pdf_at_index(self, idx):
        return self._pdf[idx]

    def get_endpoints_at_index(self, idx):
        return self._splits[idx:idx + 2]

    def get_pmf_at_index(self, idx):
        return self._pdf[idx] * self._average_interval

    def get_num_interval(self):
        return len(self._pdf)

    def get_statistical_distance(self,
                                 stat_dist_type=constant.StatisticalDistanceType.TOTAL_VARIATION_DISTANCE,
                                 other=None):
        """

        :param stat_dist_type:
        :param other: ApproximateDistribution, delta distribution by default
        :return:
        """
        if other is None:
            freqs = [self.APPROX_ZERO] * self.get_num_interval()
            freqs[len(freqs) // 2] = 1 - sum(freqs)
            other = ApproximateDistribution(splits=copy.deepcopy(self.splits), freqs=freqs)

        elif type(other) != ApproximateDistribution:
            raise TypeError("invalid statistical distance type: {}".format(type(other)))

        if stat_dist_type == constant.StatisticalDistanceType.TOTAL_VARIATION_DISTANCE:
            stat_dist = self._get_total_variation_distance(other)
        elif stat_dist_type == constant.StatisticalDistanceType.KL_DIVERGENCE:
            stat_dist = self._get_kl_divergence(other)
        elif stat_dist_type == constant.StatisticalDistanceType.RENYI_DIVERGENCE:
            stat_dist = self._get_renyi_divergence(other)
        else:
            raise ValueError("invalid statistical_distance_type: {}".format(stat_dist_type))

        return stat_dist

    def _get_total_variation_distance(self, other):
        stat_dist = 0.0
        for interval_idx in range(self.get_num_interval()):
            self_mass = self.get_pmf_at_index(interval_idx)
            other_mass = other.get_pmf_at_index(interval_idx)
            stat_dist += np.abs(self_mass - other_mass)
        stat_dist *= 0.5
        return stat_dist

    def _get_kl_divergence(self, other):
        stat_dist = 0.0
        for interval_idx in range(self.get_num_interval()):
            self_mass = self.get_pmf_at_index(interval_idx)
            other_mass = other.get_pmf_at_index(interval_idx)
            stat_dist += self_mass * np.log(self_mass / other_mass)
        return stat_dist

    def _get_renyi_divergence(self, other, alpha=2):
        stat_dist = 0.0
        for interval_idx in range(self.get_num_interval()):
            self_mass = self.get_pmf_at_index(interval_idx)
            other_mass = other.get_pmf_at_index(interval_idx)
            stat_dist += np.power(self_mass, alpha) / np.power(other_mass, alpha - 1)
        stat_dist = 1 / (alpha - 1) * np.log(stat_dist)
        return stat_dist

    def _check_equal_split_interval(self, splits):
        """

        :param splits: List[float]
        :return: float if True, bool otherwise
        """
        average_interval = 0.0
        prev_interval = None

        for i in range(1, len(splits)):
            this_interval = splits[i] - splits[i - 1]
            average_interval += this_interval

            if prev_interval is None:
                prev_interval = splits[i] - splits[i - 1]

            else:
                if np.abs(prev_interval - this_interval) < self.TOLERANCE:
                    continue
                else:
                    return False

        average_interval /= len(splits) - 1

        return average_interval

    def _scale_freqs(self, freqs):
        """
        Construct piecewise constant probability density function (PDF)
        :param freqs:
        :return: List[float]
        """
        # get scale
        fake_prob = 0.0
        for i in range(1, len(self._splits)):
            fake_prob += (self._splits[i] - self._splits[i - 1]) * freqs[i - 1]
        scale = 1.0 / fake_prob

        # scale down the frequencies
        pdf = []
        for freq in freqs:
            pdf.append(freq * scale)

        return pdf

    @staticmethod
    def binary_search(arr, x):
        """

        :param arr: must be sorted in ascending order, e.g., [0, 2, 5, 10, 11]
        :param x: 3.4
        :return: 1 (index)
        """
        l_idx = 0
        r_idx = len(arr) - 1
        while True:
            m_idx = (l_idx + r_idx) // 2
            if x < arr[m_idx]:
                r_idx = m_idx
            elif x > arr[m_idx]:
                l_idx = m_idx
            else:
                return m_idx
            if r_idx - l_idx <= 1:
                return l_idx
