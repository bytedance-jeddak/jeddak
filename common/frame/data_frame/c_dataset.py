import copy
import random
from collections import OrderedDict

from common.frame.data_frame.dataset import Dataset
from common.frame.data_frame.sample import Sample
from common.frame.data_frame.schema import Schema

import numpy as np
import pandas as pd


class CDataset(Dataset):
    """
    Centralized Dataset
    """
    def __init__(self, rows, schema=Schema(), feature_dimension=0):
        """

        :param rows: dict {str: Sample} for paired CDataset or List for unpaired CDataset
        :param schema:
        :param feature_dimension:
        """
        super(CDataset, self).__init__(schema, feature_dimension)

        self._rows = rows

    def __add__(self, other):
        if other == 0:
            return self

        elif isinstance(other, CDataset):
            return self.join(other).mapValues(lambda val: val[0] + val[1])
        else:
            raise TypeError("invalid type {} for CDataset addition".format(type(other)))

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float) \
                or isinstance(other, np.int) or isinstance(other, np.float):
            return self.mapValues(lambda val: other * val)
        elif isinstance(other, CDataset):
            return self.join(other).mapValues(lambda val: val[0] * val[1])
        else:
            raise TypeError("invalid type {} for CDataset multiplication".format(type(other)))

    def __iter__(self):
        if self.is_paired():
            for one_id, one_sample in self._rows.items():
                yield one_id, one_sample

        else:
            for one_row in self._rows:
                yield one_row

    def __str__(self):
        return str(self._rows)

    @property
    def rows(self):
        return self._rows

    def eliminate_key(self):
        if isinstance(self._rows, dict):
            return CDataset(list(self._rows.values()), feature_dimension=self._feature_dimension)
        else:
            return self.map(lambda row: row[1])

    def has_label(self):
        if self.isEmpty() or not self.is_paired():
            return False

        _, one_sample = self.first()

        return type(one_sample) == Sample and one_sample.label is not None and self._schema.label_name is not None

    def mapPartitions(self, f, preserve_schema=False):
        new_rows = next(f(self))

        return self._parse_preserve_schema(new_rows, preserve_schema)

    def map_for_list(self, f, preserve_schema=False):
        new_rows = [f(one_row) for one_row in self]
        return self._parse_preserve_schema(new_rows, preserve_schema)

    def map(self, f, preserve_schema=False):
        new_rows = None

        for one_row in self:
            new_row = f(one_row)

            if self.is_paired_row(new_row):
                # paired new_row
                if new_rows is None:
                    new_rows = {}
                new_rows[new_row[0]] = new_row[1]
            else:
                # unpaired new_row
                if new_rows is None:
                    new_rows = []
                new_rows.append(new_row)

        return self._parse_preserve_schema(new_rows, preserve_schema)

    def filter(self, f, preserve_schema=False):
        new_rows = self._init_rows_likewise()

        for one_row in self:
            if f(one_row):
                if self.is_paired():
                    new_rows[one_row[0]] = one_row[1]

                else:
                    new_rows.append(one_row)

        return self._parse_preserve_schema(new_rows, preserve_schema)

    def distinct(self, numPartitions=None):
        if self.is_paired():
            return copy.deepcopy(self)

        else:
            return self._parse_preserve_schema(list(set(self._rows)), False)

    def sample(self, withReplacement, fraction, preserve_schema=False, seed=None):
        num_sample = max(1, int(len(self._rows) * fraction))

        new_rows = self._init_rows_likewise()

        if self.is_paired():
            if withReplacement:
                ids = random.choices(list(self._rows), k=num_sample)

            else:
                ids = random.sample(list(self._rows), k=num_sample)

            for one_id in ids:
                new_rows[one_id] = self.rows[one_id]

        else:
            if withReplacement:
                rows = random.choices(list(self._rows), k=num_sample)

            else:
                rows = random.sample(list(self._rows), k=num_sample)

            new_rows = list(rows)

        return self._parse_preserve_schema(new_rows, preserve_schema)

    def takeSample(self, withReplacement, num, seed=None):
        ret = []

        if self.is_paired():
            if withReplacement:
                ids = random.choices(list(self._rows), k=num)
            else:
                ids = random.sample(list(self._rows), k=num)

            for one_id in ids:
                ret.append((one_id, self.rows[one_id]))

        else:
            if withReplacement:
                rows = random.choices(list(self._rows), k=num)
            else:
                rows = random.sample(list(self._rows), k=num)

            ret = list(rows)

        return ret

    def sortByKey(self, ascending=True, keyfunc=lambda x: x):
        assert self.is_paired()
        t_rows = {}                 # transformed rows
        for one_id, one_sample in self._rows.items():
            t_rows[keyfunc(one_id)] = one_sample
        new_rows = OrderedDict(sorted(t_rows.items(), reverse=not ascending))
        return self._parse_preserve_schema(new_rows, True)

    def glom(self):
        return self._parse_preserve_schema(
            new_rows=list(self._rows.items()) if self.is_paired() else list(self._rows),
            preserve_schema=False
        )

    def groupBy(self, f):
        assert not self.is_paired()

        new_rows = {}

        for row in self._rows:
            ret = f(row)

            if ret not in new_rows:
                new_rows[ret] = {row}

            else:
                new_rows[ret].add(row)

        return self._parse_preserve_schema(new_rows, False)

    def reduce(self, f=None):
        return self._rows

    def fold(self, zeroValue, op):
        assert not self.is_paired()

        ret = zeroValue

        for row in self._rows:
            ret = op(ret, row)

        return ret

    def max(self, key=None):
        if key is None:
            assert not self.is_paired()

            return max(self._rows)

        else:
            assert self.is_paired()

            max_id = max(self._rows)

            return max_id, self._rows[max_id]

    def min(self, key=None):
        if key is None:
            assert not self.is_paired()

            return min(self._rows)

        else:
            assert self.is_paired()

            min_id = min(self._rows)

            return min_id, self._rows[min_id]

    def sum(self):
        assert not self.is_paired()

        return sum(self._rows)

    def countByValue(self):
        assert not self.is_paired()

        summary = {}

        for row in self._rows:
            if row not in summary:
                summary[row] = 0

            else:
                summary[row] += 1

        return summary

    def take(self, num):
        take_count = 0

        ret = []

        for row in self:
            ret.append(row)

            take_count += 1

            if take_count >= num:
                break

        return ret

    def first(self):
        for row in self:
            return row

    def isEmpty(self):
        return not self._rows

    def keys(self):
        assert self.is_paired()

        return self.map(lambda row: row[0])

    def values(self):
        assert self.is_paired()

        return self.map(lambda row: row[1])

    def join(self, other):
        self._to_paired()
        intersect_ids = set(self._rows).intersection(set(other.rows))

        new_rows = {}

        for one_id in intersect_ids:
            new_rows[one_id] = (self._rows[one_id], other.rows[one_id])

        return self._parse_preserve_schema(new_rows, False)
    
    def mapValues(self, f, preserve_schema=False):
        self._to_paired()
        new_rows = {}
        for one_id, one_sample in self._rows.items():
            new_rows[one_id] = f(one_sample)

        return self._parse_preserve_schema(new_rows, preserve_schema)

    def collect(self):
        if self.is_paired():
            return list(self._rows.items())
        return self._rows

    def count(self):
        return len(self._rows)

    def persist(self, storageLevel=None):
        return self

    def unpersist(self):
        return self

    def zipWithIndex(self):
        new_rows = []

        for idx, row in enumerate(self):
            new_rows.append((row, idx))

        return self._parse_preserve_schema(new_rows, False)

    def toLocalIterator(self):
        return self.__iter__()

    def is_paired(self):
        return isinstance(self._rows, dict)

    def make_key_distinct(self):
        return self

    def join_reserve_left(self, other):
        joined_dataset = self.join(other)
        return joined_dataset.mapValues(lambda row_val: row_val[0])

    def join_reserve_right(self, other):
        joined_dataset = self.join(other)
        return joined_dataset.mapValues(lambda row_val: row_val[1])

    @staticmethod
    def is_paired_row(row):
        return (type(row) == list or type(row) == tuple) and len(row) == 2

    def _parse_preserve_schema(self, new_rows, preserve_schema):
        if preserve_schema:
            return CDataset(new_rows, schema=self._schema, feature_dimension=self._feature_dimension)

        else:
            return CDataset(new_rows, feature_dimension=self._feature_dimension)

    def _init_rows_likewise(self):
        if self.is_paired():
            return {}

        else:
            return []

    def to_data_frame(self):
        header = [self.schema.id_name]
        if self.schema.label_name:
            header = header + [self.schema.label_name] + self.schema.feature_names
        else:
            header = header + self.schema.feature_names
        dataset = []
        for item in self:
            identifier, sample = item[0], item[1]
            sample_label = [sample.label] if sample.label is not None else []
            sample_vector = [identifier] + sample_label + list(sample.features)
            dataset.append(sample_vector)
        return pd.DataFrame(dataset, columns=header)

    def from_data_frame(self, data_frame):
        with_label = self.schema.label_name is not None
        # if with_label:
        #     self.schema.feature_names = list(data_frame.columns[2:])
        # else:
        #     self.schema.feature_names = list(data_frame.columns[1:])
        new_rows = []
        for rows in data_frame.itertuples():
            _index, identifier = rows[0], rows[1]
            if with_label:
                label, features = rows[2], rows[3:]
                new_rows.append((identifier, Sample(features, label)))
            else:
                features = rows[2:]
                new_rows.append((identifier, Sample(features, None)))
        self._rows = new_rows

    def _to_paired(self):
        if self.is_paired():
            return
        new_rows = {}
        for row in self:
            if not self.is_paired_row(row):
                raise ValueError("Fail to convert dataset to key-value pair")
            key, value = row[0], row[1]
            new_rows[key] = value
        self._rows = new_rows

