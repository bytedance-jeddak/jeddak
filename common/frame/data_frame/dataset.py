import functools

from common.frame.data_frame.schema import Schema
from common.util.conversion import Conversion


class Dataset(object):
    def __init__(self, schema, feature_dimension):
        """

        :param schema: Schema
        :param feature_dimension: int
        """
        self._schema = schema
        self._feature_dimension = feature_dimension

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    @property
    def schema(self):
        return self._schema

    @schema.setter
    def schema(self, args):
        """

        :param args: (id_name, feature_names, label_name)
        :return:
        """
        if type(args) is tuple or type(args) is list:
            self._schema = Schema(id_name=args[0], feature_names=args[1], label_name=args[2])
        else:
            self._schema = args

    @property
    def feature_dimension(self):
        return self._feature_dimension

    @feature_dimension.setter
    def feature_dimension(self, feature_dimension):
        self._feature_dimension = feature_dimension

    def has_label(self):
        pass

    def mapPartitions(self, *args):
        """
        Apply a function to each partition or all of the Dataset
        :param args:
        :return:
        """
        pass

    def map(self, *args):
        """
        Apply a function to each row of the Dataset
        :param args:
        :return:
        """
        pass

    def filter(self, f, preserve_schema=False):
        """
        Return a Dataset that contains rows satisfying a predicate
        :param f:
        :param preserve_schema:
        :return:
        """
        pass

    def distinct(self, *args):
        """
        Make a paired Dataset keys or an unpaired Dataset rows distinct
        :param args:
        :return:
        """
        pass

    def sample(self, withReplacement, fraction, preserve_schema=False, seed=None):
        """
        Return a sampled subset in the form of CDataset
        :param withReplacement:
        :param fraction:
        :param preserve_schema:
        :param seed:
        :return:
        """
        pass

    def takeSample(self, withReplacement, num, seed=None):
        """
        Return a sampled subset in the form of list
        :param withReplacement:
        :param num:
        :param seed:
        :return:
        """
        pass

    def sortByKey(self, *args):
        """
        Sorts this Dataset, which is assumed to consist of (key, value) pairs.
        """
        pass

    def glom(self):
        """
        Return a Dataset with rows being a List
        :return:
        """
        pass

    def groupBy(self, *args):
        """
        Return a Dataset of grouped items.
        """
        pass

    def reduce(self, f):
        """
        Reduces the elements of this Dataset using the specified commutative and
        associative binary operator.
        """
        pass

    def fold(self, zeroValue, op):
        """
        Aggregate with operator op and initial condition zeroValue
        :param zeroValue:
        :param op:
        :return:
        """
        pass

    def max(self, key=None):
        """
        Find the maximum item
        :param key:
        :return:
        """
        pass

    def min(self, key=None):
        """
        Find the minimum item
        :param key:
        :return:
        """
        pass

    def sum(self):
        """
        Add up all rows
        """
        pass

    def countByValue(self):
        """
        Return the count of each unique value in the Dataset as a dictionary of
        (value, count) pairs.
        """
        pass

    def take(self, num):
        """
        Take the first num elements of the Dataset.
        """
        pass

    def first(self):
        """
        Return the first element in this Dataset.
        """
        pass

    def isEmpty(self):
        """
        Returns true if and only if the Dataset contains no elements at all.
        """
        pass

    def keys(self):
        pass

    def values(self):
        pass

    def join(self, *args):
        """
        Return a Dataset containing all pairs of elements with matching keys in
        Dataset{self} and Dataset{other}.

        Each pair of elements will be returned as a (k, (v1, v2)) tuple, where
        (k, v1) is in Dataset{self} and (k, v2) is in Dataset{other}.
        """
        pass

    def mapValues(self, f, preserve_schema=False):
        """
        Pass each value in the key-value pair Dataset through a map function
        without changing the keys
        """
        pass

    def collect(self):
        """
        Return an iterator that contains all of the elements in this Dataset.
        """
        pass

    def count(self):
        """
        Return the number of elements in this Dataset.
        """
        pass

    def persist(self, storageLevel):
        pass

    def unpersist(self):
        pass

    def zipWithIndex(self):
        """
        Zips this Dataset with its element indices.
        """
        pass

    def toLocalIterator(self):
        pass

    def is_paired(self):
        """
        Check whether this Dataset is paired, i.e., in the (key, value) form
        :return:
        """
        pass

    def make_key_distinct(self):
        """
        Eliminate duplicate keys while preserving schema
        :return:
        """
        pass

    def update_feature_dimension(self):
        sample_feature_dim = len(self.first()[1].features)
        self._feature_dimension = sample_feature_dim

    def get_key_table(self):
        """

        :return: each row = (key, None)
        """
        key_table = self.map(lambda row: (row[0], None))
        key_table.feature_dimension = 0
        return key_table

    def join_reserve_left(self, *args):
        """
        Join and reserve the left dataset's values
        """
        pass

    def join_reserve_right(self, *args):
        """
        Join and reserve the right dataset's values
        """
        pass

    def join_mapValues(self, other, f):
        """
        Apply first join and then mapValues
        :param other:
        :param f:
        :return:
        """
        return self.join(other).mapValues(f)

    def sum_values(self):
        """
        Sum up values
        :return:
        """
        return self.map(lambda row: row[1]).sum()

    def eliminate_key(self):
        """
        Eliminate the key column
        :return:
        """
        return self.map(lambda row: row[1])

    def key_str2int(self):
        """
        Convert key from string to int
        :return:
        """
        return self.map(lambda row: (Conversion.str2int(row[0]), row[1]))

    def key_int2str(self):
        """
        Convert key from int to string
        :return:
        """
        return self.map(lambda row: (Conversion.int2str(row[0]), row[1]))

    def append(self, other):
        """
        Tackle non-paired DDataset
        [('u0', 0.0, 0.0), ('u1', 1.0, 1.0)].append([('u0', 't0'), ('u1', 't1')]) ==
            [('u0', 0.0, 0.0, 't0'), ('u1', 1.0, 1.0, 't1')]
        :param other:
        :return:
        """
        self_kv_ = self.map(lambda row: (row[0], row[1:]))
        other_kv_ = other.map(lambda row: (row[0], row[1:]))
        return self_kv_.join_mapValues(other_kv_, lambda val: val[0] + val[1]).map(lambda row: (row[0],) + row[1])

    def replace_with_constant(self, new_val):
        """
        Replace the values of a paired DDataset
        :param new_val:
        :return:
        """

        def replace_for_each_value(val, new_val): return new_val

        return self.mapValues(functools.partial(replace_for_each_value, new_val=new_val))

    def filter_missing_value(self, preserve_schema=False):
        return self.filter(lambda x: x is not None, preserve_schema=preserve_schema)

    def partitionBy(self, num_partitions):
        return self

    def repartition(self, num_partitions):
        return self
