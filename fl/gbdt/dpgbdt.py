import copy
import functools
import time
import json
from datetime import datetime

import numpy as np

from common.factory.encryptor_factory import EncryptorFactory
from common.frame.data_frame.gh_pair import GHPair
from common.frame.data_frame.sample import Sample
from common.frame.data_frame.schema import Schema
from common.frame.data_frame.tree import Forest, TreeNode, BFTree
from common.frame.message_frame.dpgbdt_message import DPGBDTMessage
from common.frame.model_frame.dpgbdt_model import DPGBDTModel
from common.frame.parameter_frame.dpgbdt_parameter import DPGBDTParameter
from common.util import constant
from fl.algorithm import Algorithm
from fl.operator.bucketizer import Bucketizer
from fl.operator.evaluator import Evaluator
from fl.operator.histogram_maker import HistogramMaker
from fl.operator.loss import Loss
from fl.operator.objective import Objective
from coordinator.core.sqlite_manager import SqliteManager


class DPGBDT(Algorithm):
    EPS = 1e-8

    def __init__(self, parameter: DPGBDTParameter, message=DPGBDTMessage()):
        super(DPGBDT, self).__init__(parameter, message)

        self._forest = []

        self._evaluator = self._init_evaluator()

    def train(self, input_data=None, input_model=None):
        """

        :param input_data:
        :param input_model:
        :return: output_data, output_model
        """
        # train
        self._logger.info("start training")
        result = self._allocate_task(
            guest_task=functools.partial(self._train_guest, input_data=input_data, input_model=input_model),
            host_task=functools.partial(self._train_host, input_data=input_data, input_model=input_model))
        self._logger.info("end training")

        return result

    def predict(self, input_data=None):
        """

        :param input_data: features
        :return:
        """
        self._logger.info("start predicting")

        result = self._allocate_task(guest_task=functools.partial(self._predict_guest, input_data=input_data),
                                     host_task=functools.partial(self._predict_host, input_data=input_data))

        self._logger.info("end predicting")

        return result

    def instance_to_model(self):
        return DPGBDTModel(parameter=self._parameter, party_info=self.get_party_info(), forest=self._forest)

    @staticmethod
    def model_to_instance(model):
        algorithm_instance = DPGBDT(model.parameter)

        algorithm_instance.set_party_info(*model.party_info)

        algorithm_instance.set_forest(model.forest)

        return algorithm_instance

    def _train_guest(self, input_data, input_model):
        # parse label
        input_data = self._parse_label(input_data)
        self._logger.info("parsed labels")
        if isinstance(input_data.take(1)[0][1].label, list):
            label_dim = len(input_data.take(1)[0][1].label)
        else:
            label_dim = 1

        # send label_dimension
        self._messenger.send(label_dim,
                             tag=self._message.LABEL_DIM,
                             suffix=None,
                             parties=self.get_all_host_names())
        self._logger.info("sent label_dim: {}".format(label_dim))

        # find all_none_features
        all_none_features_index = self._find_all_none_feature(input_data)
        self._logger.info("finished finding all none features")

        # init a homomorphic cryptosystem
        homo_encryptor = self._init_homomorphic_encryptor()
        self._logger.info("init homomorphic encryptor")

        # init local differential privacy encoders
        grad_dp_list = []
        hess_dp_list = []
        for i in range(label_dim):
            grad_dp, hess_dp = self._init_differential_privacy_encoder()
            grad_dp_list.append(grad_dp)
            hess_dp_list.append(hess_dp)
        self._logger.info("init dp encoders")

        # bucketize features
        b_data, b_summary = self._bucketize(input_data, all_none_features_index)
        self._logger.info("bucketized features")

        # init predicted labels
        y_hat_list = []
        for i in range(label_dim):
            y_hat = self._init_predicted_label(input_data)
            y_hat_list.append(y_hat)
        self._logger.info("init y_hat")

        # init objective
        objective = self._init_objective()
        self._logger.info("init objective")

        # get init loss value
        loss_val_list = []
        for i in range(label_dim):
            if label_dim == 1:
                loss_val = self._get_loss(objective=objective,
                                          y=input_data.mapValues(lambda sample: sample.label),
                                          y_hat=y_hat_list[i])
            else:
                loss_val = self._get_loss(objective=objective,
                                          y=input_data.mapValues(lambda sample: sample.label[i]),
                                          y_hat=y_hat_list[i])
            loss_val_list.append(loss_val)
        self._logger.info("init loss value: {}".format(loss_val_list))

        # init loss reduction calculator
        loss = self._init_loss(constant.LossType.GBDT)
        self._logger.info("init loss")

        # init forest
        for i in range(label_dim):
            forest = self._init_forest()
            self._forest.append(forest)
        self._logger.info("init forest")

        fea_imp = Importance(self._parameter)

        # parse privacy mode
        privacy_mode = self._parse_privacy_mode()
        self._logger.info("privacy mode: {}".format(privacy_mode))

        # parse first-order approximation
        self._parse_first_order_approx(privacy_mode)
        self._logger.info("first order approximation: {}".format(self._parameter.first_order_approx))

        # start boosting
        for forest_idx in range(label_dim):
            self._logger.info("start training {}-th forest".format(forest_idx))

            # record the start timestamp
            old_time4db = datetime.now()
            for iter_idx in range(self._parameter.num_round):
                self._logger.info("start training {}-th round".format(iter_idx))
                # init & get train_auc and validate_auc

                train_auc = 0
                validate_auc = 0
                if self._parameter.train_validate_freq is not None and iter_idx % self._parameter.train_validate_freq == 0:
                    train_auc = self.validate(input_data)[1][0]["auc"]
                    self._logger.info("start validation, evaluate train data auc:{}".format(train_auc))
                    validate_result = self.validate()[1]
                    if validate_result:
                        validate_auc = validate_result[0]["auc"]
                        self._logger.info("start validation, evaluate validate data:{}".format(validate_auc))

                # sub-sample
                sub_input_data = input_data.sample(withReplacement=False,
                                                   fraction=self._parameter.sub_sample,
                                                   preserve_schema=True)
                self._logger.info("sampled input data: {}".format(sub_input_data.count()))

                # compute sample-wise grad and hess
                grad_hess = self._get_grad_hess(sub_input_data, y_hat_list[forest_idx], objective, forest_idx)
                self._logger.info("got grad_hess: {}".format(grad_hess.count()))

                # encrypt or encode
                enc_beg_time = time.time()
                enc_grad_hess = self._encrypt_grad_hess(grad_hess, homo_encryptor, grad_dp_list[forest_idx],
                                                        hess_dp_list[forest_idx], privacy_mode)
                self._logger.info("encrypted grad_hess: {} in {} seconds".format(
                    enc_grad_hess.count(), time.time() - enc_beg_time))

                # send grad and hess
                self._messenger.send(enc_grad_hess,
                                     tag=self._message.GRAD_HESS,
                                     suffix=[iter_idx],
                                     parties=self.get_all_host_names())
                self._logger.info("sent {}-iter grad_hess: {}".format(iter_idx, enc_grad_hess.count()))

                # init tree
                tree = self._init_tree()
                self._logger.info("init tree")

                # init root node
                self._init_root_node(tree, loss, grad_hess)
                self._logger.info("init root node")

                # init sample locations and input_data locations, i.e., the nodes that each sample is assigned to
                # DDataset ('u0', set(node_idx))
                sample_loc = self._init_sample_location(sub_input_data)
                all_loc = self._init_sample_location(input_data)
                self._logger.info("init sample and input_data locations")

                # breadth-first grow
                for dep in range(self._parameter.max_depth):
                    self._logger.info("forest-iter-dep: {}-{}-{}".format(forest_idx, iter_idx, dep))

                    # node-wise grow at each level
                    for node_idx in tree.get_indices_at_dep(dep):
                        self._logger.info("forest-iter-dep-node: {}-{}-{}-{}".format(
                            forest_idx, iter_idx, dep, node_idx))

                        # skip null nodes
                        if tree[node_idx] is None or tree[node_idx].is_leaf:
                            self._logger.info("encountered a null node, will skip")
                            tree.extend_null_nodes()
                            continue

                        # make all non-null nodes leaves at last depth
                        if dep == self._parameter.max_depth - 1:
                            self._make_leaf(tree, node_idx, loss, grad_hess, sample_loc, privacy_mode)
                            self._logger.info("reached leaves at maximum depth: {}".format(tree.max_depth))
                            continue

                        # check min child weight and early stop if breached
                        if not self._parameter.first_order_approx:
                            min_child_weight_breach = self._check_min_child_weight_early_stop(tree, node_idx)
                            self._messenger.send(min_child_weight_breach,
                                                 tag=self._message.MIN_CHILD_WEIGHT_BREACH,
                                                 suffix=[iter_idx, dep, node_idx],
                                                 parties=self.get_all_host_names())
                            self._logger.info(
                                "sent min_child_weight_breach: {} at {}-foreset {}-iter {}-dep {}-node".format(
                                    min_child_weight_breach, forest_idx, iter_idx, dep, node_idx))
                            if min_child_weight_breach:
                                self._early_stop_at_leaf(
                                    tree, node_idx, loss, grad_hess, sample_loc, privacy_mode,
                                    early_stop_type=constant.EarlyStopType.MIN_CHILD_WEIGHT
                                )
                                continue

                        if b_data is not None:
                            # compute histogram
                            set_hist_beg_time = time.time()
                            self._set_current_node_histogram(
                                tree=tree, node_idx=node_idx, grad_hess=grad_hess, b_data=b_data,
                                b_summary=b_summary, sample_loc=sample_loc
                            )
                            self._logger.info("finished setting histogram in {} seconds".format(
                                time.time() - set_hist_beg_time))

                            # find best local (guest) split
                            guest_max_loss_red, guest_max_feat_name, guest_max_bin_idx, guest_def_direction = self._find_best_split(
                                tree=tree, node_idx=node_idx,
                                histogram=tree[node_idx].histogram, loss=loss, loss_val=tree[node_idx].loss
                            )
                            self._logger.info(
                                "found guest local best split with loss reduction: {}, feature: {}, bin index: {}".format(
                                    guest_max_loss_red, guest_max_feat_name, guest_max_bin_idx))
                        else:
                            guest_max_loss_red = None
                            guest_max_feat_name = None
                            guest_max_bin_idx = None
                            guest_def_direction = None
                            tree[node_idx].histogram = None

                        # receive encrypted histogram from host
                        host_histogram = self._messenger.receive(tag=self._message.ENC_HISTOGRAM,
                                                                 suffix=[iter_idx, dep, node_idx],
                                                                 parties=self.get_all_host_names())[0]
                        self._logger.info(
                            "received encrypted histogram at {}-forest {}-iter {}-dep {}-node".format(
                                forest_idx, iter_idx, dep, node_idx))

                        if privacy_mode == constant.PrivacyMode.HOMO:
                            # decrypt host histogram
                            dec_hist_beg_time = time.time()
                            self._decrypt_histogram(host_histogram, homo_encryptor)
                            self._logger.info("decrypted histogram in {} seconds".format(
                                time.time() - dec_hist_beg_time))

                            # find best host split
                            self._logger.info("start finding host best split")
                            host_max_loss_red, host_max_feat_name, host_max_bin_idx, host_def_direction = self._find_best_split(
                                tree=tree, node_idx=node_idx,
                                histogram=host_histogram, loss=loss, loss_val=tree[node_idx].loss
                            )
                            self._logger.info(
                                "found host best split with loss reduction: {}, feature: {}, bin index: {}".format(
                                    host_max_loss_red, host_max_feat_name, host_max_bin_idx))

                        elif privacy_mode == constant.PrivacyMode.LDP:
                            # receive host's local best computed by host itself
                            host_max_loss_red, host_max_feat_name, host_max_bin_idx, host_def_direction = \
                                self._messenger.receive(
                                    tag=self._message.HOST_LOCAL_BEST,
                                    suffix=[iter_idx, dep, node_idx],
                                    parties=self.get_all_host_names()
                                )[0]
                            self._logger.info(
                                "received host_max_loss_red: {}, host_max_feat_name: {}, host_max_bin_idx: {}".format(
                                    host_max_loss_red, host_max_feat_name, host_max_bin_idx))
                        else:
                            raise ValueError("invalid privacy_mode: {}".format(privacy_mode))

                        # compare guest's and host's best split and choose the even better one
                        affiliation, histogram, max_feat_name, max_bin_idx, max_loss_red, def_direction = self._find_global_best_split(
                            guest_histogram=tree[node_idx].histogram,
                            guest_max_loss_red=guest_max_loss_red,
                            guest_max_feat_name=guest_max_feat_name,
                            guest_max_bin_idx=guest_max_bin_idx,
                            guest_def_direction=guest_def_direction,
                            host_histogram=host_histogram,
                            host_max_loss_red=host_max_loss_red,
                            host_max_feat_name=host_max_feat_name,
                            host_max_bin_idx=host_max_bin_idx,
                            host_def_direction=host_def_direction
                        )
                        self._logger.info("global best split's affiliation: {}".format(affiliation))

                        # grow
                        gamma_early_stop = self._check_gamma_early_stop(guest_max_loss_red, host_max_loss_red)
                        if gamma_early_stop:
                            self._early_stop_at_leaf(
                                tree, node_idx, loss, grad_hess, sample_loc, privacy_mode,
                                early_stop_type=constant.EarlyStopType.SMALL_LOSS_REDUCTION
                            )

                        else:
                            self._grow(tree=tree, dep=dep, node_idx=node_idx, loss=loss,
                                       affiliation=affiliation, histogram=histogram, max_feat_name=max_feat_name,
                                       max_bin_idx=max_bin_idx, def_direction=def_direction)
                            self._logger.info("finished growing")

                            if affiliation + ':' + max_feat_name in fea_imp.weight.keys():
                                fea_imp.weight[affiliation + ':' + max_feat_name] = fea_imp.weight[
                                                                                        affiliation + ':' + max_feat_name] + 1
                                fea_imp.total_gain[affiliation + ':' + max_feat_name] = fea_imp.total_gain[
                                                                                            affiliation + ':' + max_feat_name] + max_loss_red
                            else:
                                fea_imp.weight[affiliation + ':' + max_feat_name] = 1
                                fea_imp.total_gain[affiliation + ':' + max_feat_name] = max_loss_red

                    # sync current-level node info
                    self._sync_node_info_at_current_level(tree, iter_idx, dep)
                    self._logger.info("finished syncing node info at current level")

                    # update node split value
                    update_loc_or_not, update_all_or_not = self._update_node_split_value_at_current_level(tree, dep,
                                                                                                          b_summary)
                    self._logger.info("updated node split value:{}, {}".format(update_loc_or_not, update_all_or_not))

                    # sync next-level node info
                    self._sync_node_info_at_next_level(tree, iter_idx, dep)
                    self._logger.info("finished syncing node info at next level")

                    # exit this tree if the latest level contains all None
                    if tree.is_latest_dep_all_nan():
                        self._logger.info("early stop due to no further levels")
                        break

                    # update, sync and merge location info
                    all_loc = self._sync_sample_location(tree, iter_idx, dep, b_data, all_loc, update_loc_or_not,
                                                         update_all_or_not)
                    self._logger.info("finished syncing location info")
                    sample_loc = all_loc.join_reserve_left(sample_loc)

                    def count_cover_for_each_partition(partition):
                        my_list = []
                        for item in partition:
                            lenth = len(item[1])
                            for it in item[1]:
                                my_list.append(it)
                        my_set = set(my_list)
                        re_list = [0] * (2 ** lenth - 1)
                        for item in my_set:
                            re_list[item] = my_list.count(item)
                        yield re_list

                    cover_count = sample_loc.mapPartitions(count_cover_for_each_partition).reduce(
                        lambda x, y: np.array(x) + np.array(y))
                    for i in range(len(cover_count)):
                        node_fea = tree[i].split_feature
                        if node_fea is not None:
                            if tree[i].affiliation + ':' + node_fea in fea_imp.total_cover.keys():
                                fea_imp.total_cover[tree[i].affiliation + ':' + node_fea] = fea_imp.total_cover[tree[
                                                                                                                    i].affiliation + ':' + node_fea] + \
                                                                                            cover_count[i]
                            else:
                                fea_imp.total_cover[tree[i].affiliation + ':' + node_fea] = cover_count[i]

                # update y_hat
                y_hat_list[forest_idx] = self._update_y_hat(y_hat_list[forest_idx], tree, all_loc)

                # get loss value
                if isinstance(sub_input_data.first()[1].label, list):
                    loss_val_list[forest_idx] = self._get_loss(objective=objective,
                                                               y=sub_input_data.mapValues(
                                                                   lambda row: row.label[forest_idx]),
                                                               y_hat=y_hat_list[forest_idx])
                else:
                    loss_val_list[forest_idx] = self._get_loss(objective=objective,
                                                               y=sub_input_data.mapValues(lambda row: row.label),
                                                               y_hat=y_hat_list[forest_idx])
                self._logger.info(
                    "{}-forest {}-iter loss value: {}".format(forest_idx, iter_idx, loss_val_list[forest_idx]))

                # save task progress to db
                new_time4db = datetime.now()
                SqliteManager.task_progress_dbo.create(args=dict(
                    task_id=self._task_chain_id,
                    progress_type=constant.BoardConstants.TASK_PROGRESS_LOSS,
                    progress_value=json.dumps(dict(
                        loss=loss_val_list[forest_idx],
                        time=(new_time4db - old_time4db).seconds,
                        train_auc=train_auc,
                        validate_auc=validate_auc
                    ))
                ))
                old_time4db = new_time4db

                # append tree to forest
                tree.clear_all_node_histogram()
                self._forest[forest_idx].append(tree)
                self._logger.info("appended tree to forest")

        # construct output model
        output_model = self.instance_to_model()

        fea_imp.get_importance(self._parameter.importance_type)

        return input_data, output_model

    def _train_host(self, input_data, input_model):
        # find all_none_features
        all_none_features_index = self._find_all_none_feature(input_data)
        self._logger.info("finished finding all none features")

        label_dim = self._messenger.receive(tag=self._message.LABEL_DIM,
                                            suffix=None,
                                            parties=self.get_all_guest_names())[0]
        self._logger.info("received label_dim: {}".format(label_dim))

        # bucketize features
        b_data, b_summary = self._bucketize(input_data, all_none_features_index)
        self._logger.info("bucketized features")

        # init loss reduction calculator
        loss = Loss(loss_type=constant.LossType.GBDT)
        self._logger.info("init loss")

        # init forest
        for i in range(label_dim):
            forest = self._init_forest()
            self._forest.append(forest)
        self._logger.info("init forest")

        # parse privacy mode
        privacy_mode = self._parse_privacy_mode()
        self._logger.info("privacy mode: {}".format(privacy_mode))

        # parse first-order approximation
        self._parse_first_order_approx(privacy_mode)
        self._logger.info("first order approximation: {}".format(self._parameter.first_order_approx))

        # start boosting
        for forest_idx in range(label_dim):
            self._logger.info("start training {}-th forest".format(forest_idx))

            # record the start timestamp
            old_time4db = datetime.now()
            for iter_idx in range(self._parameter.num_round):
                self._logger.info("start training {}-th round".format(iter_idx))

                if self._parameter.train_validate_freq is not None and iter_idx % self._parameter.train_validate_freq == 0:
                    # validate using train data
                    self.validate(input_data)
                    # validate using validate data
                    self.validate()

                # init tree
                tree = self._init_tree()
                self._logger.info("init tree")

                # init root node
                self._init_root_node(tree, loss)
                self._logger.info("init root node")

                # receive grad and hess
                enc_grad_hess = self._messenger.receive(tag=self._message.GRAD_HESS,
                                                        suffix=[iter_idx],
                                                        parties=self.get_all_guest_names())[0]
                self._logger.info(
                    "received {}-forest {}-iter grad_hess: {}".format(forest_idx, iter_idx, enc_grad_hess.count()))

                # init sample locations and input_data locations, i.e., the nodes that each sample is assigned to
                # DDataset ('u0', set(node_idx))
                sub_input_data = input_data.join_reserve_left(enc_grad_hess)
                sample_loc = self._init_sample_location(sub_input_data)
                all_loc = self._init_sample_location(input_data)
                self._logger.info("init sample and input_data locations")

                # breadth-first grow
                for dep in range(self._parameter.max_depth):
                    self._logger.info("forest-iter-dep: {}-{}-{}".format(forest_idx, iter_idx, dep))

                    # split finding
                    for node_idx in tree.get_indices_at_dep(dep):
                        self._logger.info("forest-iter-dep-node: {}-{}-{}-{}".format(
                            forest_idx, iter_idx, dep, node_idx))

                        # skip null nodes
                        if tree[node_idx] is None or tree[node_idx].is_leaf:
                            self._logger.info("encountered a null node, will skip")
                            continue

                        if dep == self._parameter.max_depth - 1:
                            self._logger.info("slack at last depth")
                            continue

                        # check min child weight and early stop if breached
                        if not self._parameter.first_order_approx:
                            min_child_weight_breach = self._messenger.receive(
                                tag=self._message.MIN_CHILD_WEIGHT_BREACH,
                                suffix=[iter_idx, dep, node_idx],
                                parties=self.get_all_guest_names()
                            )[0]
                            self._logger.info(
                                "received min_child_weight_breach: {} at {}-forest {}-iter {}-dep {}-node".format(
                                    min_child_weight_breach, forest_idx, iter_idx, dep, node_idx))
                            if min_child_weight_breach:
                                self._logger.info("stop splitting at node {} due to min_child_weight")
                                continue

                        # compute histogram
                        set_hist_beg_time = time.time()
                        self._set_current_node_histogram(
                            tree=tree, node_idx=node_idx, grad_hess=enc_grad_hess,
                            b_data=b_data, b_summary=b_summary, sample_loc=sample_loc
                        )
                        self._logger.info("finished setting histogram in {} seconds".format(
                            time.time() - set_hist_beg_time))

                        self._messenger.send(tree[node_idx].histogram,
                                             tag=self._message.ENC_HISTOGRAM,
                                             suffix=[iter_idx, dep, node_idx],
                                             parties=self.get_all_guest_names())
                        self._logger.info("sent encrypted histogram at {}-forest {}-iter {}-dep {}-node".format(
                            forest_idx, iter_idx, dep, node_idx))

                        if privacy_mode == constant.PrivacyMode.HOMO:
                            # slack
                            pass

                        elif privacy_mode == constant.PrivacyMode.LDP:
                            # set total gh from histogram
                            self._set_total_gh_from_histogram(tree, node_idx)

                            # set loss from total gh
                            self._set_loss_from_total_gh(tree, node_idx, loss)

                            # find best host split
                            self._logger.info("start finding host best split")
                            host_max_loss_red, host_max_feat_name, host_max_bin_idx, host_def_direction = self._find_best_split(
                                tree=tree, node_idx=node_idx,
                                histogram=tree[node_idx].histogram, loss=loss, loss_val=tree[node_idx].loss
                            )
                            self._logger.info(
                                "found host best split with loss reduction: {}, feature: {}, bin index: {}".format(
                                    host_max_loss_red, host_max_feat_name, host_max_bin_idx))

                            # send over
                            self._messenger.send(
                                (host_max_loss_red, host_max_feat_name, host_max_bin_idx, host_def_direction),
                                tag=self._message.HOST_LOCAL_BEST,
                                suffix=[iter_idx, dep, node_idx],
                                parties=self.get_all_guest_names())
                            self._logger.info(
                                "sent host_max_loss_red: {}, host_max_feat_name: {}, host_max_bin_idx: {}".format(
                                    host_max_loss_red, host_max_feat_name, host_max_bin_idx))

                        else:
                            raise ValueError("invalid privacy_mode: {}".format(privacy_mode))

                    # sync and append current-level node info
                    self._sync_node_info_at_current_level(tree, iter_idx, dep)
                    self._logger.info("finished syncing node info at current level")

                    # update current-level node split value
                    update_loc_or_not, update_all_or_not = self._update_node_split_value_at_current_level(tree, dep,
                                                                                                          b_summary)
                    self._logger.info("updated node split value:{}, {}".format(update_loc_or_not, update_all_or_not))

                    # sync next-level node info
                    self._sync_node_info_at_next_level(tree, iter_idx, dep)
                    self._logger.info("finished syncing node info at next level")

                    if tree.is_latest_dep_all_nan():
                        self._logger.info("early stop due to no further levels")
                        break

                    # update, sync and merge location info
                    all_loc = self._sync_sample_location(tree, iter_idx, dep, b_data, all_loc, update_loc_or_not,
                                                         update_all_or_not)
                    self._logger.info("finished syncing location info")
                    sample_loc = all_loc.join_reserve_left(sample_loc)

                # save task progress to db
                new_time4db = datetime.now()
                SqliteManager.task_progress_dbo.create(args=dict(
                    task_id=self._task_chain_id,
                    progress_type=constant.BoardConstants.TASK_PROGRESS_LOSS,
                    progress_value=json.dumps(dict(
                        loss=0,
                        time=(new_time4db - old_time4db).seconds
                    ))
                ))
                old_time4db = new_time4db

                # append tree to forest
                tree.clear_all_node_histogram()
                self._forest[forest_idx].append(tree)
                self._logger.info("appended tree to forest")

        # construct output model
        output_model = self.instance_to_model()

        return input_data, output_model

    def _predict_guest(self, input_data):
        self._logger.info("start predicting at guest")

        traverse_sample_loc_list = self._init_traverse_sample_loc_guest(input_data)
        self._logger.info("init traverse sample location: {}".format(traverse_sample_loc_list[0].count()))

        y_hat_list = []
        host_sample_loc_list = []

        for i in range(len(self._forest)):
            host_sample_loc_i = self._messenger.receive(
                tag=self._message.TRAVERSE_SYNC,
                suffix=None,
                parties=self.get_all_host_names()
            )[0]
            self._logger.info(
                "received {}-th traverse_sample_loc: {}".format(i, host_sample_loc_i.count()))
            host_sample_loc_list.append(host_sample_loc_i)

        for i in range(len(traverse_sample_loc_list)):
            forest_idx = i
            self._logger.info("predict at {}th forest".format(forest_idx))

            traverse_sample_loc_list[i] = self._traverse_round_guest(
                traverse_sample_loc_list[i], input_data, forest_idx, host_sample_loc_list)

            traverse_sample_loc_list[i] = self._fill_null_weights(traverse_sample_loc_list[i], forest_idx)

            y_hat_i = self._aggregate_score(traverse_sample_loc_list[i])
            y_hat_list.append(y_hat_i)

        y_hat = y_hat_list[0]
        if len(y_hat_list) > 1:
            for i in range(1, len(y_hat_list)):
                y_hat = y_hat.join(y_hat_list[i])

            def unpack(element, stack):
                if isinstance(element, float):
                    stack.append(element)
                else:
                    for i in element:
                        unpack(i, stack)
                return stack

            y_hat = y_hat.mapValues(lambda row: unpack(row, []))

        # construct DDataset as output
        y_hat = y_hat.mapValues(lambda val: Sample(label=val))
        y_hat.schema = Schema(id_name=input_data.schema.id_name, label_name='y_pred')

        return y_hat

    def _predict_host(self, input_data):
        self._logger.info("start predicting at host")

        traverse_sample_loc_list = self._init_traverse_sample_loc_host(input_data)
        self._logger.info("init traverse sample location: {}".format(traverse_sample_loc_list[0].count()))

        for i in range(len(self._forest)):
            forest_idx = i
            self._logger.info("predict at {}th forest".format(forest_idx))

            traverse_sample_loc_list[i] = self._traverse_round_host(traverse_sample_loc_list[i], input_data, forest_idx)

            self._messenger.send(traverse_sample_loc_list[i],
                                 tag=self._message.TRAVERSE_SYNC,
                                 suffix=None,
                                 parties=self.get_all_guest_names())
            self._logger.info("sent {}-th traverse_sample_loc".format(i))

        return input_data

    def _aggregate_score(self, tree_leaf_dict):
        """

        :param tree_leaf_dict: DDataset(uid, {tree_idx: (leaf_idx, weight)})
        :return:
        """

        def aggregate_score_for_each_value(value, base_score, eta):
            y_hat = base_score
            for _, weight in value.values():
                y_hat += eta * weight
            return y_hat

        def pred_for_each_value(value, objective): return objective.pred(value)

        y_hat = tree_leaf_dict.mapValues(functools.partial(
            aggregate_score_for_each_value,
            base_score=self._parameter.base_score,
            eta=self._parameter.eta
        ))

        objective = self._init_objective()

        y_hat = y_hat.mapValues(functools.partial(
            pred_for_each_value, objective=objective
        ))

        return y_hat

    def _get_tree_leaf_dict(self, forest_idx):
        """

        :return: {tree_idx: {leaf_indices}}
        """
        tree_leaf_dict = {}
        for tree_idx in range(self._forest[forest_idx].get_tree_num()):
            tree_leaf_dict[tree_idx] = self._forest[forest_idx][tree_idx].get_leaf_indices()
        return tree_leaf_dict

    def _init_traverse_sample_loc_guest(self, input_data):
        traverse_sample_loc_list = []
        init_traverse_sample_loc_value = {}

        for i in range(len(self._forest)):
            for tree_idx in range(self._forest[i].get_tree_num()):
                init_traverse_sample_loc_value[tree_idx] = 0
            traverse_sample_loc = input_data.replace_with_constant(init_traverse_sample_loc_value)
            traverse_sample_loc_list.append(traverse_sample_loc)

        return traverse_sample_loc_list

    def _init_traverse_sample_loc_host(self, input_data):
        traverse_sample_loc_list = []
        init_traverse_sample_loc_value = []

        for i in range(len(self._forest)):
            loc_list = []
            init_traverse_sample_loc_value.append(0)
            for tree_idx in range(self._forest[i].get_tree_num()):
                loc_list.append(init_traverse_sample_loc_value)
            traverse_sample_loc = input_data.replace_with_constant(loc_list)
            traverse_sample_loc_list.append(traverse_sample_loc)

        return traverse_sample_loc_list

    def _traverse_round_guest(self, traverse_sample_loc, input_data, forest_idx, host_sample_loc_list):
        def traverse_round_for_each_value(value, forest, forest_idx, schema, this_party_role):
            """

            :param value: (sample, {tree_idx: cur_node_idx}) if no leaves reached
                            otherwise (sample, {tree_idx: (leaf_node, weight)}})
            :param forest: Forest
            :param schema: Schema
            :param this_party_role: str
            :return: {tree_idx: cur_node_idx}
            """
            features = value[0].features
            cur_sample_loc = value[1][0]
            host_sample_loc = value[1][1]

            new_sample_loc = {}

            for tree_idx, cur_node_idx in cur_sample_loc.items():
                tree = forest[forest_idx][tree_idx]

                while True:
                    if type(cur_node_idx) == tuple:
                        # already reach a leaf
                        break

                    else:
                        cur_node = tree[cur_node_idx]

                        if cur_node.is_leaf:
                            # make a leaf
                            cur_node_idx = (cur_node_idx, cur_node.weight)

                        else:
                            if cur_node.affiliation == this_party_role:
                                f_idx = schema.get_feature_index(cur_node.split_feature)
                                if features[f_idx] != None:
                                    if features[f_idx] <= cur_node.split_value:
                                        cur_node_idx = tree.get_left_son_index_at_index(cur_node_idx)
                                    else:
                                        cur_node_idx = tree.get_right_son_index_at_index(cur_node_idx)
                                else:
                                    if cur_node.def_direction == "left":
                                        cur_node_idx = tree.get_left_son_index_at_index(cur_node_idx)
                                    elif cur_node.def_direction == "right" or cur_node.def_direction == None:
                                        cur_node_idx = tree.get_left_son_index_at_index(cur_node_idx)
                            else:
                                if tree.get_left_son_index_at_index(cur_node_idx) in host_sample_loc[tree_idx]:
                                    cur_node_idx = tree.get_left_son_index_at_index(cur_node_idx)
                                elif tree.get_right_son_index_at_index(cur_node_idx) in host_sample_loc[tree_idx]:
                                    cur_node_idx = tree.get_right_son_index_at_index(cur_node_idx)

                new_sample_loc[tree_idx] = cur_node_idx

            return new_sample_loc

        traverse_sample_loc = traverse_sample_loc.join(host_sample_loc_list[forest_idx])

        traverse_sample_loc = input_data.join_mapValues(
            traverse_sample_loc,
            functools.partial(
                traverse_round_for_each_value,
                forest=self._forest,
                forest_idx=forest_idx,
                schema=input_data.schema,
                this_party_role=self.get_this_party_role()
            ))

        return traverse_sample_loc

    def _traverse_round_host(self, traverse_sample_loc, input_data, forest_idx):
        """

        :param traverse_sample_loc: DDataset(uid, {tree_idx: cur_node_idx})
        :param input_data: DDataset(uid, Sample)
        :return:
        """

        def traverse_round_for_each_value(value, forest, forest_idx, schema, this_party_role):
            """

            :param value: (sample, {tree_idx: cur_node_idx}) if no leaves reached
                            otherwise (sample, {tree_idx: (leaf_node, weight)}})
            :param forest: Forest
            :param schema: Schema
            :param this_party_role: str
            :return: {tree_idx: cur_node_idx}
            """
            features = value[0].features

            new_sample_loc = []
            sample_loc_temp = []

            for tree_idx in range(forest[forest_idx].get_tree_num()):
                tree = forest[forest_idx][tree_idx]
                sample_loc_temp.append([0])

                for node_idx in range(3):
                    cur_node = tree[node_idx]
                    if cur_node == None:
                        continue
                    else:
                        if cur_node.affiliation == this_party_role:
                            f_idx = schema.get_feature_index(cur_node.split_feature)
                            if features[f_idx] != None:
                                if features[f_idx] <= cur_node.split_value:
                                    cur_node_idx = tree.get_left_son_index_at_index(node_idx)
                                else:
                                    cur_node_idx = tree.get_right_son_index_at_index(node_idx)
                                sample_loc_temp[tree_idx].append(cur_node_idx)
                            else:
                                if cur_node.def_direction == "left":
                                    cur_node_idx = tree.get_left_son_index_at_index(node_idx)
                                    sample_loc_temp[tree_idx].append(cur_node_idx)
                                elif cur_node.def_direction == "right" or cur_node.def_direction == None:
                                    cur_node_idx = tree.get_left_son_index_at_index(node_idx)
                                    sample_loc_temp[tree_idx].append(cur_node_idx)
                        else:
                            continue

                new_sample_loc.append(sample_loc_temp[tree_idx])

            return new_sample_loc

        traverse_sample_loc = input_data.join_mapValues(
            traverse_sample_loc,
            functools.partial(
                traverse_round_for_each_value,
                forest=self._forest,
                forest_idx=forest_idx,
                schema=input_data.schema,
                this_party_role=self.get_this_party_role()
            ))

        return traverse_sample_loc

    def _all_reach_leaves(self, traverse_sample_loc):
        def all_dict_values_tuples(row):
            """

            :param row: (uid, {0: 0}) or (uid, {0: (1, -0.15)})
            :return:
            """
            tree_leaf_dict = row[1]
            all_tuple = True
            for leaf in tree_leaf_dict.values():
                if type(leaf) != tuple:
                    all_tuple = False
                    break
            return all_tuple

        num_sample = traverse_sample_loc.count()
        num_sample_reach_leaf = traverse_sample_loc.filter(all_dict_values_tuples).count()

        num_sample_not_reach_leaf = num_sample - num_sample_reach_leaf
        self._logger.info("{} samples have not reached a leaf".format(num_sample_not_reach_leaf))

        return num_sample_not_reach_leaf == 0

    def _parse_label(self, input_data):
        learning_type = self._parameter.objective.split('_')[0]
        if learning_type == 'reg':
            return input_data

        elif learning_type == 'binary':
            def parse_binary_label(sample):
                if isinstance(sample.label, list):
                    for i in range(len(sample.label)):
                        if sample.label[i] == 1.0:
                            sample.label[i] = np.float(1.0)
                        else:
                            sample.label[i] = np.float(0.0)
                else:
                    if sample.label == 1.0:
                        sample.label = np.float(1.0)
                    else:
                        sample.label = np.float(0.0)
                return sample

            return input_data.mapValues(parse_binary_label, preserve_schema=True)

        elif learning_type == 'count':
            def parse_count_label(sample):
                if isinstance(sample.label, list):
                    for i in range(len(sample.label)):
                        if sample.label[i] < 0.0:
                            sample.label[i] = np.float(0.0)
                        else:
                            sample.label[i] = np.round(sample.label[i])
                else:
                    if sample.label < 0.0:
                        sample.label = np.float(0.0)
                    else:
                        sample.label = np.round(sample.label)
                return sample

            return input_data.mapValues(parse_count_label, preserve_schema=True)

        else:
            raise ValueError("invalid objective: {}".format(learning_type))

    def _get_grad_hess(self, input_data, y_hat, objective, forest_idx):
        """

        :param input_data: DDataset
        :param y_hat: DDataset
        :param objective: Objective
        :param forest_idx:
        :return: (uid, GHPair)
        """

        def get_gh_for_each_value(value, objective):
            """

            :param value: (y, y_hat)
            :param objective:
            :return:
            """
            grad = objective.grad(*value)
            hess = objective.hess(*value)

            return GHPair(grad, hess)

        if isinstance(input_data.take(1)[0][1].label, list):
            y_yhat = input_data.join_mapValues(y_hat, lambda val: (val[0].label[forest_idx], val[1]))
        else:
            y_yhat = input_data.join_mapValues(y_hat, lambda val: (val[0].label, val[1]))

        grad_hess = y_yhat.mapValues(functools.partial(get_gh_for_each_value,
                                                       objective=objective))
        return grad_hess

    def _get_total_grad_hess(self, feat_hist):
        """

        :param feat_hist: OrderedDict {'x0': Histogram}
        :return: GHPair
        """
        for _, hist in feat_hist.items():
            return hist.total_freqs

    def _compute_sensitivity(self, grad_hess):
        """

        :param grad_hess: DDataset (uid, GHPair)
        :return:
        """
        grad_sens = grad_hess.mapValues(lambda val: val.grad).max() - grad_hess.mapValues(lambda val: val.grad).min()
        hess_sens = grad_hess.mapValues(lambda val: val.hess).max() - grad_hess.mapValues(lambda val: val.hess).min()
        return grad_sens, hess_sens

    def _filter_by_sample_loc(self, b_data, sample_loc, target_node_idx):
        """

        :param b_data: DDataset ('u0', Sample)
        :param sample_loc: ('u0', set())
        :param target_node_idx: the target node index
        :return:
        """
        sample_loc = sample_loc.filter(lambda row: target_node_idx in row[1])
        node_data = b_data.join_reserve_left(sample_loc)
        return node_data

    def _set_current_node_histogram(self, tree, node_idx, grad_hess, b_data, b_summary, sample_loc):
        """
        Set current tree node histogram
        :param tree: current tree
        :param node_idx: current node index
        :param grad_hess: DDataset
        :param b_data: DDataset
        :param b_summary: {'x0': BucketSummary}
        :param sample_loc: DDataset
        :return:
        """
        if tree.is_left_node(node_idx) or not tree.is_capable_of_histogram_subtraction(node_idx):
            # find the samples assigned to current node
            node_data = self._filter_by_sample_loc(b_data=b_data,
                                                   sample_loc=sample_loc,
                                                   target_node_idx=node_idx)
            self._logger.info("finished filtering by sample location: {}".format(node_data.count()))

            # compute guest bucket-wise grad and hess
            tree[node_idx].histogram = HistogramMaker(
                to_cumulative=False,
                need_log=True
            ).fit(b_data=node_data, b_summary=b_summary, targets=grad_hess)

        else:
            # apply histogram subtraction
            parent_histogram = tree.get_parent_node_at_index(node_idx).histogram
            sibling_histogram = tree.get_sibling_node_at_index(node_idx).histogram
            tree[node_idx].histogram = {}
            for f_name in parent_histogram.keys():
                tree[node_idx].histogram[f_name] = parent_histogram[f_name] - sibling_histogram[f_name]

    def _get_loss_reduction(self, loss, total_loss, left_gh_pair, total_gh_pair):
        """

        :param loss: Loss
        :param total_loss: float
        :param left_gh_pair: GHPair
        :param total_gh_pair: GHPair
        :return:
        """
        # compute left loss
        left_loss = loss.eval(gh_pair=left_gh_pair, lam=self._parameter.lam)

        # compute right loss
        right_gh_pair = total_gh_pair - left_gh_pair
        right_loss = loss.eval(gh_pair=right_gh_pair, lam=self._parameter.lam)

        # compute loss reduction
        loss_red = total_loss - left_loss - right_loss - self._parameter.gamma

        return loss_red

    def _find_best_split(self, tree, node_idx, histogram, loss, loss_val):
        """

        :param tree:
        :param node_idx:
        :param histogram: {'x0': Histogram}
        :param loss: Loss
        :param loss_val: float, loss value
        :return:
        """
        max_loss_red = -np.inf  # maximum loss reduction
        max_feat_name = ''  # maximum feature name
        max_bin_idx = -1  # maximum bucket index
        default_direction = None

        # find best local split
        for f_name, hist in histogram.items():
            total_gh_pair = tree[node_idx].total_grad_hess
            left_gh_pair = 0

            for bin_idx, bin_gh_pair in enumerate(hist.freqs):
                none_gh_pair = total_gh_pair - hist.total_freqs
                if bin_idx + 1 == len(hist.freqs):
                    # unnecessary to further split when all bins go to left
                    break

                left_gh_pair = left_gh_pair + bin_gh_pair
                loss_red = self._get_loss_reduction(loss=loss,
                                                    total_loss=loss_val,
                                                    left_gh_pair=left_gh_pair,
                                                    total_gh_pair=total_gh_pair)

                loss_red_prime = self._get_loss_reduction(loss=loss,
                                                          total_loss=loss_val,
                                                          left_gh_pair=left_gh_pair + none_gh_pair,
                                                          total_gh_pair=total_gh_pair)

                if loss_red_prime > loss_red:
                    loss_red = loss_red_prime
                    def_direction = 'left'
                else:
                    def_direction = 'right'

                if loss_red > max_loss_red:
                    max_loss_red = loss_red
                    max_feat_name = f_name
                    max_bin_idx = bin_idx
                    default_direction = def_direction

        return max_loss_red, max_feat_name, max_bin_idx, default_direction

    def _decrypt_histogram(self, histogram, homo_encryptor):
        for f_name in histogram.keys():
            for freq_idx in range(len(histogram[f_name])):
                if histogram[f_name][freq_idx].is_ciphertext():
                    histogram[f_name][freq_idx] = histogram[f_name][freq_idx].apply(
                        homo_encryptor.decrypt
                    )

    def _encrypt_grad_hess(self, grad_hess, homo_encryptor, grad_dp, hess_dp, privacy_mode):
        if privacy_mode == constant.PrivacyMode.HOMO:
            return grad_hess.mapValues(lambda gh_pair: gh_pair.apply(homo_encryptor.encrypt))
        else:
            raise ValueError("invalid privacy_mode: {}".format(privacy_mode))

    def _init_tree(self):
        tree = BFTree(max_depth=self._parameter.max_depth)
        return tree

    def _init_sample_location(self, input_data):
        sample_loc = input_data.replace_with_constant({0})
        return sample_loc

    def _init_predicted_label(self, input_data):
        y_hat = input_data.replace_with_constant(new_val=self._parameter.base_score)
        return y_hat

    def _init_forest(self):
        forest = Forest()
        return forest

    def _init_root_node(self, tree, loss, grad_hess=None):
        if grad_hess is None:
            root_node = TreeNode()

        else:
            total_grad_hess = grad_hess.map(lambda row: row[1]).sum()
            root_node = TreeNode(
                total_grad_hess=total_grad_hess,
                loss=loss.eval(gh_pair=total_grad_hess, lam=self._parameter.lam)
            )

        tree.append(root_node)

    def _init_differential_privacy_encoder(self):
        return None, None

    def _init_homomorphic_encryptor(self):
        homo_encryptor = EncryptorFactory.get_instance(task_type=self._parameter.task_type,
                                                       encrypter=self._parameter.homomorphism,
                                                       key_size=self._parameter.key_size)
        return homo_encryptor

    def _bucketize(self, input_data, all_none_features_index):
        b_data, b_summary = Bucketizer(bucketizer_type=constant.BucketizerType.QUANTILE,
                                       num_bucket=int(1 / self._parameter.sketch_eps),
                                       remove_original_columns=True,
                                       output_sampleset=True,
                                       feature_suffix='',
                                       need_log=True
                                       ).fit(input_data, all_none_features_index)
        return b_data, b_summary

    def _check_gamma_early_stop(self, guest_max_loss_red, host_max_loss_red):
        if guest_max_loss_red == None:
            return host_max_loss_red <= self.EPS
        else:
            return max(guest_max_loss_red, host_max_loss_red) <= self.EPS

    def _find_global_best_split(self,
                                guest_histogram, guest_max_loss_red, guest_max_feat_name, guest_max_bin_idx,
                                guest_def_direction,
                                host_histogram, host_max_loss_red, host_max_feat_name, host_max_bin_idx,
                                host_def_direction):
        """

        :param guest_histogram:
        :param guest_max_loss_red:
        :param guest_max_feat_name:
        :param guest_max_bin_idx:
        :param host_histogram:
        :param host_max_loss_red:
        :param host_max_feat_name:
        :param host_max_bin_idx:
        :return:
        """
        if guest_max_loss_red == None or guest_max_loss_red < host_max_loss_red:
            affiliation = constant.TaskRole.HOST
            histogram = host_histogram
            max_feat_name = host_max_feat_name
            max_bin_idx = host_max_bin_idx
            def_direction = host_def_direction
            max_loss_red = host_max_loss_red
        else:
            affiliation = constant.TaskRole.GUEST
            histogram = guest_histogram
            max_feat_name = guest_max_feat_name
            max_bin_idx = guest_max_bin_idx
            def_direction = guest_def_direction
            max_loss_red = guest_max_loss_red

        return affiliation, histogram, max_feat_name, max_bin_idx, max_loss_red, def_direction

    def _early_stop_at_leaf(self, tree, node_idx, loss, grad_hess, sample_loc, privacy_mode, early_stop_type):
        """
        Make the node a leaf and make its sons None
        :param tree:
        :param node_idx:
        :param loss:
        :param grad_hess:
        :param sample_loc:
        :param privacy_mode:
        :param early_stop_type:
        :return:
        """
        self._make_leaf(tree, node_idx, loss, grad_hess, sample_loc, privacy_mode)
        self._logger.info("stop splitting at node {} due to {}".format(
            node_idx, early_stop_type
        ))

        # append left and right nodes
        tree.extend_null_nodes()

    def _grow(self, tree, dep, node_idx, loss,
              affiliation, histogram, max_feat_name, max_bin_idx, def_direction):
        """

        :param tree:
        :param dep:
        :param node_idx: current node index
        :param loss:
        :param affiliation:
        :param histogram:
        :param max_feat_name:
        :param max_bin_idx:
        :return:
        """
        # fill current node's vacant info
        tree[node_idx].affiliation = affiliation
        tree[node_idx].split_feature = max_feat_name
        tree[node_idx].split_bin_idx = max_bin_idx
        tree[node_idx].def_direction = def_direction

        # construct left and right nodes
        none_grad_hess = tree[node_idx].total_grad_hess - histogram[max_feat_name].total_freqs
        left_node = TreeNode()
        left_node.total_grad_hess = GHPair(0, 0)
        right_node = TreeNode()

        if def_direction == 'left':
            for i in range(max_bin_idx + 1):
                left_node.total_grad_hess = left_node.total_grad_hess + histogram[max_feat_name][i]
            left_node.total_grad_hess = left_node.total_grad_hess + none_grad_hess
            left_node.loss = loss.eval(gh_pair=left_node.total_grad_hess, lam=self._parameter.lam)
        else:
            for i in range(max_bin_idx + 1):
                left_node.total_grad_hess = left_node.total_grad_hess + histogram[max_feat_name][i]
            left_node.loss = loss.eval(gh_pair=left_node.total_grad_hess, lam=self._parameter.lam)

        right_node.total_grad_hess = tree[node_idx].total_grad_hess - left_node.total_grad_hess
        right_node.loss = loss.eval(gh_pair=right_node.total_grad_hess, lam=self._parameter.lam)

        # append left and right nodes
        tree.extend([left_node, right_node])

    def _init_objective(self):
        objective = Objective(objective_type=self._parameter.objective,
                              max_delta_step=self._parameter.max_delta_step)
        return objective

    def _init_loss(self, loss_type):
        loss = Loss(loss_type)
        return loss

    def _sync_node_info_at_current_level(self, tree, iter_idx, dep):
        """
        Guest and host sync node info at current level
        :return:
        """
        if self.get_this_party_role() == constant.TaskRole.GUEST:
            # anonymize nodes
            nodes = []
            for node in tree.get_nodes_at_dep(dep):
                if node:
                    node = node.anonymize()
                nodes.append(node)

            # sync nodes at current level
            # these nodes only contain three occupied attributes: affiliation, split_feature and split_bin_idx (bin)
            self._messenger.send(nodes,
                                 tag=self._message.CURRENT_LEVEL_NODES,
                                 suffix=[iter_idx, dep],
                                 parties=self.get_all_host_names())
            self._logger.info("sent {} nodes at {}-iter_idx {}-dep".format(
                len(nodes), iter_idx, dep
            ))

        else:
            nodes = self._messenger.receive(tag=self._message.CURRENT_LEVEL_NODES,
                                            suffix=[iter_idx, dep],
                                            parties=self.get_all_guest_names())[0]
            self._logger.info("received {} nodes at {}-iter_idx {}-dep".format(
                len(nodes), iter_idx, dep
            ))

            # first delete then extend
            tree.merge_at_latest_dep(nodes)

    def _update_node_split_value_at_current_level(self, tree, dep, b_summary):
        """
        Replace split bin index with real split value if owning it
        :param tree:
        :param dep:
        :return:
        """
        tree_node_self_num = 0
        tree_node_num = 0
        for node_idx in tree.get_indices_at_dep(dep):
            if tree[node_idx] is not None and not tree[node_idx].is_leaf:
                tree_node_num = tree_node_num + 1
                if tree[node_idx].affiliation == self.get_this_party_role():
                    max_f_name = tree[node_idx].split_feature
                    max_split_value_idx = tree[node_idx].split_bin_idx + 1
                    split_value = b_summary[max_f_name][max_split_value_idx]
                    tree[node_idx].split_value = split_value
                    tree_node_self_num = tree_node_self_num + 1

        return (tree_node_self_num != 0), (tree_node_num == tree_node_self_num)

    def _sync_node_info_at_next_level(self, tree, iter_idx, dep):
        """
        Guest and host sync node info at next level
        :return:
        """
        # go to next level
        dep += 1

        if self.get_this_party_role() == constant.TaskRole.GUEST:
            # anonymize nodes
            nodes = []
            for node in tree.get_nodes_at_dep(dep):
                if node:
                    node = node.anonymize()
                nodes.append(node)

            # sync nodes at current level
            # these nodes only own three occupied attributes: affiliation, split_feature and split_bin_idx
            self._messenger.send(nodes,
                                 tag=self._message.NEXT_LEVEL_NODES,
                                 suffix=[iter_idx, dep],
                                 parties=self.get_all_host_names())
            self._logger.info("sent {} nodes at {}-iter_idx {}-dep".format(
                len(nodes), iter_idx, dep
            ))

        else:
            nodes = self._messenger.receive(tag=self._message.NEXT_LEVEL_NODES,
                                            suffix=[iter_idx, dep],
                                            parties=self.get_all_guest_names())[0]
            self._logger.info("received {} nodes at {}-iter_idx {}-dep".format(
                len(nodes), iter_idx, dep
            ))

            # extend
            tree.extend(nodes)

    def _sync_sample_location(self, tree, iter_idx, dep, b_data, sample_loc, update_loc_or_not, update_all_or_not):
        def update_sample_loc_for_each_value(value,
                                             cur_node_idx, left_son_node_idx, right_son_node_idx,
                                             cur_split_feature_idx, cur_split_bin_idx, cur_def_direction):
            """
            If a sample has been assigned to a parent node and
                has its split bin less than or equal to the specified bin index, then assign it to the left node.
                Otherwise assign it to the right node.
            :param value: (Sample, set())
            :param cur_node_idx:
            :param left_son_node_idx:
            :param right_son_node_idx:
            :param cur_split_feature_idx:
            :param cur_split_bin_idx:
            :return: set(), updated sample location set
            """
            sample, sample_loc = value
            new_sample_loc = copy.deepcopy(sample_loc)
            if cur_node_idx in sample_loc:
                if sample.features[cur_split_feature_idx] == None:
                    if cur_def_direction == 'right':
                        new_sample_loc.add(right_son_node_idx)
                    elif cur_def_direction == 'left':
                        new_sample_loc.add(left_son_node_idx)
                else:
                    if sample.features[cur_split_feature_idx] <= cur_split_bin_idx:
                        new_sample_loc.add(left_son_node_idx)
                    else:
                        new_sample_loc.add(right_son_node_idx)
            return new_sample_loc

        # update local sample location info
        for node_idx in tree.get_indices_at_dep(dep):
            if tree[node_idx] is not None and not tree[node_idx].is_leaf and \
                    tree[node_idx].affiliation == self.get_this_party_role():
                b_data_sample_loc = b_data.join(sample_loc)
                sample_loc = b_data_sample_loc.mapValues(functools.partial(
                    update_sample_loc_for_each_value,
                    cur_node_idx=node_idx,
                    left_son_node_idx=tree.get_left_son_index_at_index(node_idx),
                    right_son_node_idx=tree.get_right_son_index_at_index(node_idx),
                    cur_split_feature_idx=b_data.schema.get_feature_index(tree[node_idx].split_feature),
                    cur_split_bin_idx=tree[node_idx].split_bin_idx,
                    cur_def_direction=tree[node_idx].def_direction,
                ))
        self._logger.info("updated local sample location info: {}".format(sample_loc.count()))

        # sync sample location info
        if self.get_this_party_role() == constant.TaskRole.GUEST:
            if update_loc_or_not is False and update_all_or_not is True:
                pass
            elif update_loc_or_not is False and update_all_or_not is False:
                other_sample_loc = self._messenger.receive(tag=self._message.SAMPLE_LOC_H2G,
                                                           suffix=[iter_idx, dep],
                                                           parties=self.get_all_host_names())[0]
                self._logger.info("received sample location from host to guest at {}-iter {}-dep".format(
                    iter_idx, dep))
                # merge sample location info
                sample_loc = sample_loc.join_mapValues(other_sample_loc,
                                                       lambda loc: loc[0].union(loc[1]))
            elif update_loc_or_not is True and update_all_or_not is True:
                self._messenger.send(sample_loc,
                                     tag=self._message.SAMPLE_LOC_G2H,
                                     suffix=[iter_idx, dep],
                                     parties=self.get_all_host_names())
                self._logger.info("sent sample location from guest to host at {}-iter {}-dep".format(
                    iter_idx, dep))
            else:
                self._messenger.send(sample_loc,
                                     tag=self._message.SAMPLE_LOC_G2H,
                                     suffix=[iter_idx, dep],
                                     parties=self.get_all_host_names())
                self._logger.info("sent sample location from guest to host at {}-iter {}-dep".format(
                    iter_idx, dep))
                other_sample_loc = self._messenger.receive(tag=self._message.SAMPLE_LOC_H2G,
                                                           suffix=[iter_idx, dep],
                                                           parties=self.get_all_host_names())[0]
                self._logger.info("received sample location from host to guest at {}-iter {}-dep".format(
                    iter_idx, dep))
                # merge sample location info
                sample_loc = sample_loc.join_mapValues(other_sample_loc,
                                                       lambda loc: loc[0].union(loc[1]))

        else:
            if update_loc_or_not is False and update_all_or_not is True:
                pass
            elif update_loc_or_not is False and update_all_or_not is False:
                other_sample_loc = self._messenger.receive(tag=self._message.SAMPLE_LOC_G2H,
                                                           suffix=[iter_idx, dep],
                                                           parties=self.get_all_guest_names())[0]
                self._logger.info("received sample location from host to guest at {}-iter {}-dep".format(
                    iter_idx, dep))

                # merge sample location info
                sample_loc = sample_loc.join_mapValues(other_sample_loc,
                                                       lambda loc: loc[0].union(loc[1]))
            elif update_loc_or_not is True and update_all_or_not is True:
                self._messenger.send(sample_loc,
                                     tag=self._message.SAMPLE_LOC_H2G,
                                     suffix=[iter_idx, dep],
                                     parties=self.get_all_guest_names())
                self._logger.info("sent sample location from guest to host at {}-iter {}-dep".format(
                    iter_idx, dep))
            else:
                self._messenger.send(sample_loc,
                                     tag=self._message.SAMPLE_LOC_H2G,
                                     suffix=[iter_idx, dep],
                                     parties=self.get_all_guest_names())
                self._logger.info("sent sample location from host to guest at {}-iter {}-dep".format(
                    iter_idx, dep))

                other_sample_loc = self._messenger.receive(tag=self._message.SAMPLE_LOC_G2H,
                                                           suffix=[iter_idx, dep],
                                                           parties=self.get_all_guest_names())[0]
                self._logger.info("received sample location from guest to host at {}-iter {}-dep".format(
                    iter_idx, dep))

                # merge sample location info
                sample_loc = sample_loc.join_mapValues(other_sample_loc,
                                                       lambda loc: loc[0].union(loc[1]))

        return sample_loc

    def _check_min_child_weight_early_stop(self, tree, node_idx):
        """
        True if min_child_weight is breached
        :param tree:
        :return:
        """
        return tree[node_idx].total_grad_hess.hess < self._parameter.min_child_weight

    def _update_y_hat(self, y_hat, tree, sample_loc):
        def get_new_y_hat_for_each_value(value, leaf_indices, leaf_weights):
            """

            :param value:
            :param leaf_indices: set(int)
            :param leaf_weights: {node_idx: weight}
            :return:
            """
            leaf_idx = value.intersection(leaf_indices).pop()
            new_y_hat = leaf_weights[leaf_idx]
            return new_y_hat

        leaf_indices = tree.get_leaf_indices()
        leaf_weights = tree.get_leaf_weights()
        self._logger.info("tree leaf-weights: {}".format(leaf_weights))

        new_y_hat = sample_loc.mapValues(functools.partial(
            get_new_y_hat_for_each_value,
            leaf_indices=leaf_indices, leaf_weights=leaf_weights
        ))

        return y_hat + self._parameter.eta * new_y_hat

    def _get_loss(self, objective, y, y_hat):
        """

        :param objective: Objective
        :param y:
        :param y_hat:
        :return:
        """

        def _get_loss_for_each_value(value, loss_eval_func):
            return loss_eval_func(*value)

        loss_val = y.join_mapValues(
            y_hat,
            functools.partial(_get_loss_for_each_value, loss_eval_func=objective.eval)
        ).sum_values() / y.count()

        return loss_val

    def _init_evaluator(self):
        if self._parameter.objective == constant.Objective.REG_SQUAREDERROR:
            return Evaluator(constant.EvaluationType.REGRESSION)
        elif self._parameter.objective == constant.Objective.BINARY_LOGISTIC:
            return Evaluator(constant.EvaluationType.BINARY)
        elif self._parameter.objective == constant.Objective.COUNT_POISSON:
            return Evaluator(constant.EvaluationType.MULTICLASS)
        else:
            raise ValueError("no valid evaluation type for objective: {}".format(self._parameter.objective))

    def _fill_null_weights(self, traverse_sample_loc, forest_idx):
        def fill_null_weights_for_each_value(value, forest, forest_idx):
            """

            :param value: {tree_idx: (leaf_idx, weight)}
            :param forest:
            :return:
            """
            for tree_idx, (leaf_idx, weight) in value.items():
                if weight is None:
                    weight = forest[forest_idx][tree_idx][leaf_idx].weight
                    value[tree_idx] = (leaf_idx, weight)
            return value

        traverse_sample_loc = traverse_sample_loc.mapValues(functools.partial(
            fill_null_weights_for_each_value, forest=self._forest, forest_idx=forest_idx
        ))
        return traverse_sample_loc

    def _is_guest_traverse_sync_round(self, traverse_sync_round):
        return traverse_sync_round % 2 == 0

    def _parse_privacy_mode(self):
        if self._parameter.privacy_mode == constant.PrivacyMode.ADA:
            pass

        else:
            return self._parameter.privacy_mode

    def _set_total_gh_from_histogram(self, tree, node_idx):
        for hist in tree[node_idx].histogram.values():
            tree[node_idx].total_grad_hess = hist.total_freqs
            break

    def _set_loss_from_total_gh(self, tree, node_idx, loss):
        tree[node_idx].loss = loss.eval(gh_pair=tree[node_idx].total_grad_hess, lam=self._parameter.lam)

    def _set_accurate_total_grad_hess(self, tree, node_idx, grad_hess, sample_loc):
        """
        For LDP mode, get a node's accurate total gh before making it a leaf
        :param tree:
        :param node_idx:
        :param grad_hess:
        :return:
        """
        samples_on_node = sample_loc.filter(lambda row: node_idx in row[1])
        acc_grad_hess_on_node = grad_hess.join_reserve_left(samples_on_node)
        acc_grad_hess = acc_grad_hess_on_node.map(lambda row: row[1]).sum()
        tree[node_idx].total_grad_hess = acc_grad_hess

    def _make_leaf(self, tree, node_idx, loss, grad_hess, sample_loc, privacy_mode):
        if privacy_mode == constant.PrivacyMode.LDP or self._parameter.first_order_approx:
            # both LDP privacy mode and first-order-approximation mode need finding accurate total grad and hess
            self._set_accurate_total_grad_hess(tree, node_idx, grad_hess, sample_loc)

        tree[node_idx].make_leaf(loss, self._parameter.lam)

    def _parse_first_order_approx(self, privacy_mode):
        if privacy_mode == constant.PrivacyMode.HOMO:
            self._parameter.first_order_approx = False

    def set_forest(self, forest):
        self._forest = forest

    def _find_all_none_feature(self, input_data):
        def check_none_feature_for_each_partition(partition):
            none_feature = None
            for item in partition:
                temp = []
                for it in item[1].features:
                    if it != None:
                        temp.append(True)
                    else:
                        temp.append(False)
                if not isinstance(none_feature, np.ndarray):
                    none_feature = np.array(temp)
                else:
                    none_feature = np.array(temp) + none_feature
            yield none_feature.tolist()

        all_none_features_index = []
        check_array = input_data.mapPartitions(check_none_feature_for_each_partition).reduce(
            lambda x, y: np.array(x) + np.array(y))
        for i in range(len(check_array)):
            if check_array[i] == False:
                all_none_features_index.append(i)
        return all_none_features_index


class Importance(DPGBDT):
    def __init__(self, parameter: DPGBDTParameter, message=DPGBDTMessage()):
        super(Importance, self).__init__(parameter, message)
        self.weight = {}
        self.gain = {}
        self.cover = {}
        self.total_gain = {}
        self.total_cover = {}

    def get_importance(self, importance_type='weight'):
        if importance_type == 'weight':
            importance_weight = sorted(self.weight.items(), key=lambda d: d[1], reverse=True)
            self._logger.info("feature importance ranking by weight:" + str(importance_weight))
        elif importance_type == 'gain':
            self.gain = {key: value / self.weight[key] for key, value in self.total_gain.items()}
            importance_gain = sorted(self.gain.items(), key=lambda d: d[1], reverse=True)
            self._logger.info("feature importance ranking by gain:" + str(importance_gain))
        elif importance_type == 'cover':
            self.cover = {key: value / self.weight[key] for key, value in self.total_cover.items()}
            importance_cover = sorted(self.cover.items(), key=lambda d: d[1], reverse=True)
            self._logger.info("feature importance ranking by cover:" + str(importance_cover))
        elif importance_type == 'total_gain':
            importance_total_gain = sorted(self.total_gain.items(), key=lambda d: d[1], reverse=True)
            self._logger.info("feature importance ranking by total gain:" + str(importance_total_gain))
        elif importance_type == 'total_cover':
            importance_total_cover = sorted(self.total_cover.items(), key=lambda d: d[1], reverse=True)
            self._logger.info("feature importance ranking by total cover:" + str(importance_total_cover))
        elif importance_type == 'all':
            self.gain = {key: value / self.weight[key] for key, value in self.total_gain.items()}
            self.cover = {key: value / self.weight[key] for key, value in self.total_cover.items()}

            importance_weight = sorted(self.weight.items(), key=lambda d: d[1], reverse=True)
            self._logger.info("feature importance ranking by weight:" + str(importance_weight))

            importance_gain = sorted(self.gain.items(), key=lambda d: d[1], reverse=True)
            self._logger.info("feature importance ranking by gain:" + str(importance_gain))

            importance_cover = sorted(self.cover.items(), key=lambda d: d[1], reverse=True)
            self._logger.info("feature importance ranking by cover:" + str(importance_cover))

            importance_total_gain = sorted(self.total_gain.items(), key=lambda d: d[1], reverse=True)
            self._logger.info("feature importance ranking by total gain:" + str(importance_total_gain))

            importance_total_cover = sorted(self.total_cover.items(), key=lambda d: d[1], reverse=True)
            self._logger.info("feature importance ranking by total cover:" + str(importance_total_cover))
