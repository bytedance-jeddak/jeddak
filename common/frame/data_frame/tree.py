import copy

import numpy as np

from common.util import constant


class Forest(object):
    def __init__(self, tree_queue=None):
        """

        :param tree_queue: Tree
        """
        if tree_queue is None:
            tree_queue = []
        self._tree_queue = tree_queue

    def __str__(self):
        forest_display = ''
        for tree_idx, tree in enumerate(self._tree_queue):
            forest_display += 'tree.' + str(tree_idx) + ': ' + str(tree) + '\n'
        return forest_display

    def __getitem__(self, item):
        return self._tree_queue[item]

    def get_tree_num(self):
        return len(self._tree_queue)

    def append(self, tree):
        self._tree_queue.append(tree)


class Tree(object):
    pass


class BFTree(Tree):
    """
    Breadth-first tree
    """
    def __init__(self, max_depth, tree_node_queue=None):
        """

        :param max_depth: int
        :param tree_node_queue: List
        """
        if tree_node_queue is None:
            tree_node_queue = []
        self._max_depth = max_depth
        self._tree_node_queue = tree_node_queue

    def __len__(self):
        return len(self._tree_node_queue)

    def __str__(self):
        tree_display = ''
        for node_idx, node in enumerate(self._tree_node_queue):
            tree_display += 'node.' + str(node_idx) + ': ' + str(node) + '\n'
        return tree_display

    def __getitem__(self, item):
        """

        :param item: node index or slice
        :return:
        """
        if isinstance(item, slice):
            return self._tree_node_queue[item]
        else:
            return self._tree_node_queue[int(item)]

    def __setitem__(self, key, value):
        self._tree_node_queue[key] = value

    @property
    def max_depth(self):
        return self._max_depth

    @staticmethod
    def is_left_node(node_idx):
        """
        Root is also considered left
        :param node_idx:
        :return:
        """
        return node_idx == 0 or node_idx % 2 == 1

    def append(self, tree_node):
        """

        :param tree_node: TreeNode
        :return:
        """
        self._tree_node_queue.append(tree_node)

    def extend(self, tree_nodes):
        """

        :param tree_nodes: List[TreeNode]
        :return:
        """
        self._tree_node_queue += tree_nodes

    def extend_null_nodes(self, num_node=2):
        self.extend([None] * num_node)

    def merge_at_dep(self, nodes, dep):
        """
        Merge nodes to the prescribed level
        :param nodes:
        :param dep:
        :return:
        """
        node_indices = self.get_indices_at_dep(dep)
        assert len(nodes) == len(node_indices)
        for i, node_idx in enumerate(node_indices):
            if self._tree_node_queue[node_idx] is not None:
                self._tree_node_queue[node_idx] = nodes[i].merge(self._tree_node_queue[node_idx])

    def merge_at_latest_dep(self, nodes):
        latest_dep = self.get_latest_dep()
        self.merge_at_dep(nodes, latest_dep)

    def get_latest_dep(self):
        """
        Get the latest depth
        :return:
        """
        return int(np.log2(len(self._tree_node_queue) + 1)) - 1

    def get_indices_at_dep(self, dep):
        """

        :param dep: int, the depth
        :return: node indices at the prescribed depth
        """
        dep = int(dep)
        beg_idx = 2 ** dep - 1
        end_idx = 2 ** (dep + 1) - 1
        return list(range(beg_idx, end_idx))

    def get_indices_at_latest_dep(self):
        latest_dep = self.get_latest_dep()
        latest_node_indices = self.get_indices_at_dep(latest_dep)
        return latest_node_indices

    def get_nodes_at_dep(self, dep):
        dep = int(dep)
        beg_idx = 2 ** dep - 1
        end_idx = 2 ** (dep + 1) - 1
        return self._tree_node_queue[beg_idx:end_idx]

    def get_nodes_at_latest_dep(self):
        latest_dep = self.get_latest_dep()
        latest_nodes = self.get_nodes_at_dep(latest_dep)
        return latest_nodes

    def delete_at(self, index):
        if isinstance(index, slice):
            del self._tree_node_queue[index]
        else:
            del self._tree_node_queue[int(index)]

    def delete_latest_dep(self):
        latest_node_indices = self.get_indices_at_latest_dep()
        self.delete_at(slice(latest_node_indices[0], latest_node_indices[-1] + 1))

    def get_parent_index_at_index(self, node_idx):
        parent_node_idx = (node_idx - 1) // 2
        return parent_node_idx

    def get_parent_node_at_index(self, node_idx):
        """
        Get the parent node of the one at node_idx
        :param node_idx:
        :return:
        """
        parent_node_idx = self.get_parent_index_at_index(node_idx)
        return self._tree_node_queue[parent_node_idx]

    def get_left_son_index_at_index(self, node_idx):
        return int(node_idx * 2 + 1)

    def get_right_son_index_at_index(self, node_idx):
        return int((node_idx + 1) * 2)

    def get_son_indices_at_index(self, node_idx):
        return self.get_left_son_index_at_index(node_idx), self.get_right_son_index_at_index(node_idx)

    def get_sibling_index_at_index(self, node_idx):
        if node_idx % 2 == 0:
            sibling_node_idx = node_idx - 1
        else:
            sibling_node_idx = node_idx + 1
        return sibling_node_idx

    def get_sibling_node_at_index(self, node_idx):
        """
        Get the sibling node of the one at node_idx
        :param node_idx:
        :return:
        """
        sibling_node_idx = self.get_sibling_index_at_index(node_idx)
        return self._tree_node_queue[sibling_node_idx]

    def get_left_son_index_at_index(self, node_idx):
        left_son_node_idx = node_idx * 2 + 1
        return left_son_node_idx

    def get_right_son_index_at_index(self, node_idx):
        right_son_node_idx = (node_idx + 1) * 2
        return right_son_node_idx

    def is_latest_dep_all_nan(self):
        """

        :return: boolean
        """
        return not any(self.get_nodes_at_latest_dep())

    def is_capable_of_histogram_subtraction(self, node_idx):
        """

        :param node_idx:
        :return:
        """
        return self.get_parent_node_at_index(node_idx) is not None and \
            self.get_sibling_node_at_index(node_idx) is not None and \
            self.get_parent_node_at_index(node_idx).histogram is not None and \
            self.get_sibling_node_at_index(node_idx).histogram is not None

    def get_leaf_indices(self):
        """

        :return: set()
        """
        leaf_indices = set()
        for node_idx, node in enumerate(self._tree_node_queue):
            if node is not None and node.is_leaf:
                leaf_indices.add(node_idx)
        return leaf_indices

    def get_leaf_weights(self):
        """

        :return: {node_idx: weight}
        """
        leaf_weights = {}
        leaf_indices = self.get_leaf_indices()
        for leaf_idx in leaf_indices:
            leaf_weights[leaf_idx] = self._tree_node_queue[leaf_idx].weight
        return leaf_weights

    def get_root(self):
        return self._tree_node_queue[0]

    def get_root_index(self):
        return 0

    def clear_all_node_histogram(self):
        for node_idx in range(len(self._tree_node_queue)):
            if self._tree_node_queue[node_idx] is not None:
                self._tree_node_queue[node_idx].clear_histogram()


class DFTree(Tree):
    """
    Depth-first tree
    """
    def __init__(self, root=None):
        """

        :param root: TreeNode
        """
        self._root = root


class TreeNode(object):
    def __init__(self, affiliation=None, total_grad_hess=None, loss=None,
                 split_feature=None, split_bin_idx=None, split_value=None, weight=None, histogram=None,
                 is_leaf=False, left=None, right=None, def_direction = None):
        """

        :param affiliation: str, 'guest' or 'host'
        :param total_grad_hess: GHPair
        :param loss: float
        :param split_feature: str
        :param split_bin_idx: int
        :param split_value: float
        :param weight: not None if leaf
        :param histogram: {'x0': Histogram}
        :param is_leaf: boolean
        :param left: left tree node, for DFTree traverse
        :param right: right tree node, for DFTree traverse
        """
        self._affiliation = affiliation
        self._total_grad_hess = total_grad_hess
        self._loss = loss
        self._split_feature = split_feature
        self._split_bin_idx = split_bin_idx
        self._split_value = split_value
        self._weight = weight
        self._histogram = histogram
        self._is_leaf = is_leaf
        self._left = left
        self._right = right
        self._def_direction = def_direction

    def __str__(self):
        dict_display = copy.deepcopy(self.__dict__)
        if dict_display['_histogram'] is not None:
            dict_display['_histogram'] = 'tl;dr'
        return str(dict_display)

    @property
    def affiliation(self):
        return self._affiliation

    @affiliation.setter
    def affiliation(self, affiliation):
        self._affiliation = affiliation

    @property
    def total_grad_hess(self):
        return self._total_grad_hess

    @total_grad_hess.setter
    def total_grad_hess(self, total_grad_hess):
        self._total_grad_hess = total_grad_hess

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss):
        self._loss = loss

    @property
    def split_feature(self):
        return self._split_feature

    @split_feature.setter
    def split_feature(self, split_feature):
        self._split_feature = split_feature

    @property
    def split_bin_idx(self):
        return self._split_bin_idx

    @split_bin_idx.setter
    def split_bin_idx(self, split_bin_idx):
        self._split_bin_idx = split_bin_idx

    @property
    def split_value(self):
        return self._split_value

    @split_value.setter
    def split_value(self, split_value):
        self._split_value = split_value

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    @property
    def histogram(self):
        return self._histogram

    @histogram.setter
    def histogram(self, histogram):
        self._histogram = histogram

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def def_direction(self):
        return self._def_direction

    @def_direction.setter
    def def_direction(self, def_direction):
        self._def_direction = def_direction

    def make_leaf(self, loss, lam, affiliation=constant.TaskRole.GUEST):
        """
        Make self a leaf by assigning weight
        :param loss: Loss
        :param lam:
        :param affiliation: the affiliation that the leaf belongs to
        :return:
        """
        self._affiliation = affiliation
        # self._total_grad_hess = None
        # self._loss = None
        self._split_feature = None
        self._split_bin_idx = None
        self._split_value = None
        self._weight = loss.get_weight(gh_pair=self._total_grad_hess, lam=lam)
        self._histogram = None
        self._is_leaf = True
        self._left = None
        self._right = None

    def anonymize(self):
        """
        Mute all attributes except for affiliation, split_feature, split_bin_idx and is_leaf
        :return:
        """
        return TreeNode(
            affiliation=self._affiliation,
            total_grad_hess=None,
            loss=None,
            split_feature=self._split_feature,
            split_bin_idx=self._split_bin_idx,
            split_value=None,
            weight=None,
            histogram=None,
            is_leaf=self._is_leaf,
            left=None,
            right=None,
            def_direction=self._def_direction
        )

    def merge(self, other):
        """

        :param other: TreeNode
        :return:
        """
        def get_nonempty(var_1, var_2):
            """
            Get the non-null one
            :param var_1:
            :param var_2:
            :return:
            """
            if var_1 is None:
                return var_2
            else:
                return var_1

        return TreeNode(
            affiliation=get_nonempty(self._affiliation, other.affiliation),
            total_grad_hess=get_nonempty(self._total_grad_hess, other.total_grad_hess),
            loss=get_nonempty(self._loss, other.loss),
            split_feature=get_nonempty(self._split_feature, other.split_feature),
            split_bin_idx=get_nonempty(self._split_bin_idx, other.split_bin_idx),
            split_value=get_nonempty(self._split_value, other.split_value),
            weight=get_nonempty(self._weight, other.weight),
            histogram=get_nonempty(self._histogram, other.histogram),
            is_leaf=get_nonempty(self._is_leaf, other.is_leaf),
            left=get_nonempty(self._left, other.left),
            right=get_nonempty(self._right, other.right),
            def_direction = get_nonempty(self._def_direction,other.def_direction)
        )

    def clear_histogram(self):
        self._histogram = None
