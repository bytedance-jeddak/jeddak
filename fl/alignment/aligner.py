import math
import random
import functools

import numpy as np

from common.frame.message_frame.aligner_message import AlignerMessage
from common.frame.model_frame.aligner_model import AlignerModel
from common.frame.parameter_frame.aligner_parameter import AlignerParameter
from common.frame.data_frame.c_dataset import CDataset

from common.util import constant
from fl.algorithm import Algorithm
from privacy.crypto.symmetric.diffie_hellman import DiffieHellman


try:
    from privacy.crypto.symmetric.ot_extension.IKNP_OTe import IKNP_Sender, IKNP_Receiver
    from privacy.crypto.PSI.cmPSI import CMPsiSender, CMPsiReceiver
    import privacy.crypto.PSI.PSI as PSI
    import privacy.crypto.PSI.cmPSI as cmPSI
except ImportError:
    pass


class Aligner(Algorithm):
    """
    Aligner
    """
    def __init__(self, parameter: AlignerParameter, message=AlignerMessage()):
        super(Aligner, self).__init__(parameter, message=message)

    def train(self, input_data=None, input_model=None):
        if self._parameter.align_mode == constant.Encryptor.PLAIN:
            return self._train_with_plain(input_data, input_model)
        elif self._parameter.align_mode == constant.Encryptor.DIFFIE_HELLMAN:
            if self.get_party_num() > 2:
                raise ValueError("diffie hellman does not support multiparty setup")
            return self._train_with_diffie_hellman(input_data, input_model)
        elif self._parameter.align_mode == constant.Encryptor.CM20:
            return self._train_with_cm20(input_data, input_model)
        else:
            raise ValueError("invalid align mode: {}".format(self._parameter.align_mode))

    def predict(self, input_data=None):
        output_data, _ = self.train(input_data)
        return output_data

    def instance_to_model(self):
        return AlignerModel(self._parameter, self.get_party_info())

    @staticmethod
    def model_to_instance(model):
        algorithm_instance = Aligner(model.parameter)
        algorithm_instance.set_party_info(*model.party_info)
        return algorithm_instance

    def _train_with_plain(self, input_data, input_model):
        """
        Guest -> host, then perform intersection at host, and finally be shared if parameter.sync_intersection == True
        :param input_data:
        :param input_model:
        :return:
        """
        return self._allocate_task(
            guest_task=functools.partial(self._train_with_plain_guest, input_data=input_data, input_model=input_model),
            host_task=functools.partial(self._train_with_plain_host, input_data=input_data, input_model=input_model))

    def _train_with_plain_guest(self, input_data, input_model):
        # get id table
        guest_id_table = input_data.get_key_table()
        self._logger.info("got id table")

        # get host id tables (host_id, None)
        host_id_tables = self._messenger.receive(tag=self._message.PLAIN_IDS,
                                                 parties=self.get_other_host_names())
        self._logger.info("received host id tables: {}".format(len(host_id_tables)))

        # compute intersection
        intersection_table = guest_id_table
        for host_id_table in host_id_tables:
            intersection_table = intersection_table.join_reserve_left(host_id_table)

        return self._parse_and_output(input_data, intersection_table)

    def _train_with_plain_host(self, input_data, input_model):
        # get id table
        host_id_table = input_data.get_key_table()
        self._logger.info("got id table")

        # send over (host_id, None)
        self._messenger.send(host_id_table, tag=self._message.PLAIN_IDS, parties=self.get_all_guest_names())
        self._logger.info("sent host id table: {}".format(host_id_table.count()))

        return self._parse_and_output(input_data)

    def _train_with_diffie_hellman(self, input_data, input_model):
        """
        Guest -> host public knowledge, and the intersect operation takes place at guest
        :param input_data:
        :param input_model:
        :return:
        """
        return self._allocate_task(guest_task=functools.partial(
            self._train_with_diffie_hellman_guest, input_data=input_data, input_model=input_model
        ), host_task=functools.partial(
            self._train_with_diffie_hellman_host, input_data=input_data, input_model=input_model
        ))

    def _train_with_diffie_hellman_guest(self, input_data, input_model):
        # receive Diffie-Hellman public knowledge
        diffie_hellman = self._messenger.receive(tag=self._message.DIFFIE_HELLMAN_PUBLIC_KNOWLEDGE,
                                                 parties=self.get_all_host_names())[0]
        self._logger.info("received diffie hellman public knowledge: {}".format(diffie_hellman.modulus))

        # generate Diffie-Hellman key
        diffie_hellman.fill_exponent_at_random()

        # get id table and encode string key to int
        guest_id_table = input_data.get_key_table().key_str2int()       # (g, None)

        # first encrypt and exchange
        enc_g = guest_id_table.map(lambda row: (diffie_hellman.encrypt(row[0]), None))  # (eg, None)
        self._messenger.send(enc_g, tag=self._message.ENC_G, parties=self.get_all_host_names())
        self._logger.info("sent enc_g: {}".format(enc_g.count()))

        enc_h = self._messenger.receive(tag=self._message.ENC_H,
                                        parties=self.get_all_host_names())[0]  # (eh, None)
        self._logger.info("received enc_h: {}".format(enc_h.count()))

        # second encrypt and receive
        eenc_h = enc_h.map(lambda row: (diffie_hellman.encrypt(row[0]), None))  # (eeh, None)
        self._messenger.send(eenc_h, tag=self._message.EENC_H, parties=self.get_all_host_names())
        self._logger.info("sent eenc_h: {}".format(eenc_h.count()))

        # receive
        enc_i = self._messenger.receive(tag=self._message.ENC_I,
                                        parties=self.get_all_host_names())[0]
        self._logger.info("received enc_i: {}".format(enc_i.count()))

        # final decrypt and decode
        intersection_table = enc_i.map(lambda row: (diffie_hellman.decrypt(row[0]), None))
        intersection_table = intersection_table.key_int2str()
        
        return self._parse_and_output(input_data, intersection_table)

    def _initialize_diffie_hellman(self, input_data, host_or_guest):
        # host_or_guest: True for host, False for guest
        diffie_hellman = DiffieHellman.generate_with_public_knowledge(self._parameter.key_size)
        self._messenger.send(diffie_hellman,
                             tag=self._message.DIFFIE_HELLMAN_PUBLIC_KNOWLEDGE,
                             parties=self.get_all_guest_names())
        self._logger.info("sent diffie hellman public knowledge: {}".format(diffie_hellman.modulus))
        # generate Diffie-Hellman key
        diffie_hellman.fill_exponent_at_random()
        # get id table and encode string key to int
        id_table = input_data.get_key_table().key_str2int()  # (h, None)
        tag = self._message.ENC_H if host_or_guest else self._message.ENC_G
        # first encrypt and exchange
        enc_h = id_table.map(lambda row: (diffie_hellman.encrypt(row[0]), None))  # (eh, None)
        self._messenger.send(enc_h, tag=tag, parties=self.get_all_guest_names())
        self._logger.info("sent enc_h: {}".format(enc_h.count()))
        return diffie_hellman, enc_h

    def _train_with_diffie_hellman_host(self, input_data, input_model):
        # init Diffie-Hellman public knowledge and send over
        diffie_hellman, enc_h = self._initialize_diffie_hellman(input_data, True)
        self._logger.info("sent enc_h: {}".format(enc_h.count()))
        enc_g = self._messenger.receive(tag=self._message.ENC_G,
                                        parties=self.get_all_guest_names())[0]       # (eg, None)
        self._logger.info("received enc_g: {}".format(enc_g.count()))

        # second encrypt and receive
        eenc_g = enc_g.map(lambda row: (diffie_hellman.encrypt(row[0]), None))  # (eeg, None)
        eenc_h = self._messenger.receive(tag=self._message.EENC_H,
                                         parties=self.get_all_guest_names())[0]
        self._logger.info("received eenc_h: {}".format(eenc_h.count()))

        # perform intersection
        eenc_i = eenc_h.join_reserve_left(eenc_g)           # (eei, None)

        # first decryption and send over
        enc_i = eenc_i.map(lambda row: (diffie_hellman.decrypt(row[0]), None))  # (ei, None)
        self._messenger.send(enc_i, tag=self._message.ENC_I, parties=self.get_all_guest_names())
        self._logger.info("sent enc_i: {}".format(enc_i.count()))

        return self._parse_and_output(input_data)

    @staticmethod
    def _get_psi_parameter_data_array(log_height, size, role_list):
        height = int(pow(2, log_height))
        height_in_bytes = int((height + 7) / 8)
        location_in_bytes = int((log_height + 7) / 8)

        data_array_low = PSI.u64Array(size)
        data_array_high = PSI.u64Array(size)
        for i in range(size):
            data = cmPSI.convertBits(role_list[i], 128)
            data_array_low[i] = int(data) & constant.Math.U64_MAX
            data_array_high[i] = int(data) >> 64
        # Create PSI Receiver input set and caching set
        role_set = PSI.dataArray(size)
        PSI.setDataArray(role_set, data_array_low, data_array_high, size)
        return height_in_bytes, location_in_bytes, role_set

    def _train_with_cm20(self, input_data, input_model):
        """
        Guest -> host public knowledge, and the intersect operation takes place at guest
        :param input_data:
        :param input_model:
        :return:
        """
        return self._allocate_task(guest_task=functools.partial(
            self._train_with_cm20_guest, input_data=input_data, input_model=input_model
        ), host_task=functools.partial(
            self._train_with_cm20_host, input_data=input_data, input_model=input_model
        ))

    def _train_with_cm20_host(self, input_data, input_model):
        if len(self.get_all_host_names()) > 1:
            self._logger.warning("basic cm20 is not fully secure for multi-party private set intersection")
        ''' OT phase'''
        # PSI matrix width will be smaller than 2048
        ot_choices = IKNP_Receiver.gen_random_choices(2048)
        ot_receiver = IKNP_Receiver(2048, ot_choices, 0xFFFFFFFF)
        # only support one guest party
        guest_party = self.get_all_guest_names()[0]
        recv_ot_messages = ot_receiver.run(self._messenger, guest_party)

        ''' PSI phase '''
        total_id_table = input_data.get_key_table().key_str2int().map(lambda row: row[0])
        if self._parameter.batch_num == "auto":
            total_sender_size = total_id_table.count()
            self._messenger.send(total_sender_size, parties=self.get_all_guest_names(), tag="PSI sender size")
            sh_len = self._messenger.receive(parties=self.get_all_guest_names(), tag="short hash length")[0]
        elif type(self._parameter.batch_num) == int and self._parameter.batch_num >= 1:
            sh_len = math.ceil(math.log(self._parameter.batch_num, 2))
        else:
            raise ValueError("Invalid batch_num: {}".format(self._parameter.batch_num))
        data_bucket = pow(2, sh_len)
        pp = PSI.PARAMETERS()
        pp.h1LengthInBytes = 32
        ''' split data set '''
        if sh_len == 0:
            total_id_table = total_id_table.persist()
        else:
            total_id_table = total_id_table.map(
                lambda data_id: (data_id, cmPSI.shortHash(data_id, sh_len))).persist()
        self._logger.info("start CM20 PSI, batch number: {}".format(data_bucket))
        for sh in range(data_bucket):
            # public parameters
            if sh_len == 0:
                sub_data = total_id_table
            else:
                sub_data = total_id_table.filter(lambda row: row[1] == sh).map(lambda row: row[0])
            sender_list = sub_data.collect()
            pp.senderSize = len(sender_list)
            self._messenger.send(int(pp.senderSize), parties=self.get_all_guest_names(),
                                      tag="PSI sender size, sh=%d" % sh)
            psi_parameters = self._messenger.receive(parties=self.get_all_guest_names(),
                                                          tag="PSI parameters, sh=%d" % sh)[0]
            if not psi_parameters:
                continue
            pp.width, pp.logHeight, pp.hashLengthInBytes = psi_parameters

            # Sender Init
            seed = self._messenger.receive(parties=self.get_all_guest_names(), tag="PSI Seed")[0]
            height_in_bytes, location_in_bytes, sender_set = \
                self._get_psi_parameter_data_array(pp.logHeight, pp.senderSize, sender_list)

            trans_locations = PSI.gen2DArray(pp.width, int(pp.senderSize * location_in_bytes + 4))
            sender = PSI.PsiSender()
            sender.senderInit(seed, sender_set, pp, trans_locations)
            self._logger.info("PSI sender init success")
            # Sender Output
            ot_messages_data_array = PSI.dataArray(pp.width)
            choices_data_array = PSI.uncharArray(pp.width)
            for i in range(pp.width):
                ot_messages_data_array[i] = PSI.setData(int(recv_ot_messages[i]) & constant.Math.U64_MAX)
                choices_data_array[i] = int(ot_choices[i])
            # receive matrix
            matrix_delta_array = self._messenger.receive(parties=self.get_all_guest_names(), tag="PSI MatrixDelta")[0]

            matrix_delta = PSI.gen2DArray(pp.width, height_in_bytes)
            for i in range(pp.width):
                PSI.set2DArrayRow(matrix_delta, i, matrix_delta_array[i])
            hash_outputs = PSI.gen2DArray(int(pp.senderSize / 256) + 1, 256 * pp.hashLengthInBytes)
            sender.senderOutput(seed, trans_locations, ot_messages_data_array, choices_data_array, matrix_delta, pp, hash_outputs)
            PSI.del2DArray(matrix_delta, pp.width)
            PSI.del2DArray(trans_locations, pp.width)
            self._logger.info("PSI sender output success")
            outputs_list = []
            for i in range(int(pp.senderSize / 256) + 1):
                hash_outputs_row = np.zeros([256 * pp.hashLengthInBytes], dtype=np.uint8)
                PSI.get2DArrayRow(hash_outputs_row, hash_outputs, i)
                outputs_list.append(hash_outputs_row)
            PSI.del2DArray(hash_outputs, int(pp.senderSize / 256) + 1)
            self._messenger.send(outputs_list, parties=self.get_all_guest_names(), tag="PSI sender hash outputs")
        total_id_table.unpersist()
        res = self._parse_and_output(input_data)
        self._logger.info("PSI sender end!")
        return res

    def _train_with_cm20_guest(self, input_data, input_model):
        # Basic CM20 is not fully secure for multi-party private set intersection. For an extended semi-honest
        # secure scheme of CM20, please refer to: https://eprint.iacr.org/2021/484.pdf.
        if len(self.get_all_host_names()) > 1:
            self._logger.warning("basic cm20 is not fully secure for multi-party private set intersection")
        ''' OT phase'''
        host_parties = self.get_all_host_names()
        # PSI matrix width will be smaller than 2048
        ot_messages = IKNP_Sender.gen_random_messages(2048)
        for host_party in host_parties:
            ot_sender = IKNP_Sender(2048, ot_messages[0], ot_messages[1], 0xFFFFFFFF)
            ot_sender.run(self._messenger, host_party)

        ''' PSI phase '''
        total_id_table = input_data.get_key_table().key_str2int().map(lambda row: row[0])
        if self._parameter.batch_num == "auto":
            total_receiver_size = total_id_table.count()
            total_sender_size_list = self._messenger.receive(
                parties=self.get_all_host_names(), tag="PSI sender size")
            sh_len = cmPSI.binNumComp(max(total_sender_size_list), total_receiver_size)
            self._messenger.send(sh_len, parties=self.get_all_host_names(), tag="short hash length")
        elif type(self._parameter.batch_num) == int and self._parameter.batch_num >= 1:
            sh_len = math.ceil(math.log(self._parameter.batch_num, 2))
        else:
            raise ValueError("Invalid batch_num: {}".format(self._parameter.batch_num))

        data_bucket = pow(2, sh_len)
        pp = PSI.PARAMETERS()
        pp.h1LengthInBytes = 32
        intersection_list = []
        # split data set
        if sh_len == 0:
            total_id_table = total_id_table.persist()
        else:
            total_id_table = total_id_table.map(lambda data_id: (data_id, cmPSI.shortHash(data_id, sh_len))).persist()
        self._logger.info("start CM20 PSI, batch number: {}".format(data_bucket))
        for sh in range(data_bucket):
            # public parameters
            if sh_len == 0:
                sub_data = total_id_table
            else:
                sub_data = total_id_table.filter(lambda row: row[1] == sh).map(lambda row: row[0])
            receiver_list = sub_data.collect()
            pp.receiverSize = len(receiver_list)

            sender_size_list = self._messenger.receive(parties=self.get_all_host_names(),
                                                       tag="PSI sender size, sh=%d" % sh)
            if pp.receiverSize * min(sender_size_list) == 0:
                self._messenger.send(None, parties=self.get_all_host_names(), tag="PSI parameters, sh=%d" % sh)
                continue

            pp.width = cmPSI.widthComp(max(sender_size_list), pp.receiverSize)
            pp.logHeight = cmPSI.logHeightComp(max(sender_size_list), pp.receiverSize)
            pp.hashLengthInBytes = cmPSI.hashLenComp(max(sender_size_list), pp.receiverSize)
            psi_parameters = (pp.width, pp.logHeight, pp.hashLengthInBytes)
            self._messenger.send(
                psi_parameters, parties=self.get_all_host_names(), tag="PSI parameters, sh=%d" % sh)
            height_in_bytes, location_in_bytes, receiver_set = \
                self._get_psi_parameter_data_array(pp.logHeight, pp.receiverSize, receiver_list)

            receiver = PSI.PsiReceiver()

            # Receiver Init
            trans_locations = PSI.gen2DArray(pp.width, int(pp.receiverSize * location_in_bytes + 4))
            matrix_d = PSI.gen2DArray(pp.width, height_in_bytes)
            seed = random.randint(0, constant.Math.U32_MAX)
            self._messenger.send(seed, parties=self.get_all_host_names(), tag="PSI Seed")
            receiver.receiverInit(seed, receiver_set, pp, trans_locations, matrix_d)
            self._logger.info("PSI receiver init success")

            # Receiver evaluate matrix Delta
            matrix_a = PSI.gen2DArray(pp.width, height_in_bytes)
            matrix_delta = PSI.gen2DArray(pp.width, height_in_bytes)

            ot_messages1 = PSI.dataArray(pp.width)
            ot_messages2 = PSI.dataArray(pp.width)
            for i in range(pp.width):
                ot_messages1[i] = PSI.setData(int(int(ot_messages[0][i]) & constant.Math.U64_MAX))
                ot_messages2[i] = PSI.setData(int(int(ot_messages[1][i]) & constant.Math.U64_MAX))
            ot_msg_data = PSI.combineOtMessages(ot_messages1, ot_messages2, pp.width)
            receiver.receiverEvalDelta(ot_msg_data, pp, matrix_d, matrix_a, matrix_delta)
            PSI.del2DArray(matrix_d, pp.width)
            self._logger.info("PSI receiver evaluate matrix Delta success")
            matrix_delta_array = np.zeros([pp.width, height_in_bytes], dtype=np.uint8)

            # send matrix
            for i in range(pp.width):
                PSI.get2DArrayRow(matrix_delta_array[i], matrix_delta, i)
            PSI.del2DArray(matrix_delta, pp.width)
            self._messenger.send(matrix_delta_array, parties=self.get_all_host_names(), tag="PSI MatrixDelta")
            # Receiver outputs
            inter_index = np.array([255 for _ in range(int(pp.receiverSize / 8) + 1)], dtype=np.uint8)
            for i in range(len(host_parties)):
                sender_size = sender_size_list[i]
                hash_outputs = PSI.gen2DArray(int(sender_size / 256) + 1, 256 * pp.hashLengthInBytes)
                outputs_list = self._messenger.receive(parties=[host_parties[i]], tag="PSI sender hash outputs")[0]

                for i in range(int(sender_size / 256) + 1):
                    PSI.set2DArrayRow(hash_outputs, i, outputs_list[i])
                psi_res_idx = PSI.uncharVector(int(pp.receiverSize / 8) + 1)
                # psiResIdx 每个8位，1表示该元素是交集
                receiver.receiverOutput(pp, trans_locations, matrix_a, hash_outputs, sender_size, psi_res_idx)
                inter_index = inter_index & np.array(psi_res_idx[:], dtype=np.uint8)
                PSI.del2DArray(hash_outputs, int(sender_size / 256) + 1)

            PSI.del2DArray(trans_locations, pp.width)
            PSI.del2DArray(matrix_a, pp.width)
            cur_inter_list = [int(table_id) for idx, table_id in enumerate(receiver_list) if
                              inter_index[int(idx / 8)] & (1 << (idx % 8))]
            intersection_list += cur_inter_list
            self._logger.info(
                "PSI receiver current intersection number: " + str(len(cur_inter_list)))
        self._logger.info("PSI receiver output intersectionList success, number: " + str(len(intersection_list)))
        total_id_table.unpersist()
        intersection_table = CDataset(intersection_list).map(lambda row: (row, None)).key_int2str()
        res = self._parse_and_output(input_data, intersection_table)
        self._logger.info("PSI receiver end!")
        return res

    def _parse_and_output(self, input_data, intersection_table=None):
        """
        Parse output_id_only, sync_intersection mode
        :return: output_data, output_model
        """
        if self.get_this_party_role() == constant.TaskRole.GUEST:
            # sync if turned on (intersection_id, None)
            if self._parameter.sync_intersection:
                self._messenger.send(intersection_table,
                                     tag=self._message.PLAIN_INTERSECTION,
                                     parties=self.get_other_host_names())
                self._logger.info("sent intersection id table: {}".format(intersection_table.count()))

            # construct output model
            output_model = self.instance_to_model()

            # output id only if turned on
            if not self._parameter.output_id_only:
                intersection_table = intersection_table
                intersection_table = intersection_table.join_reserve_right(input_data)
                intersection_table.schema = input_data.schema
                intersection_table.feature_dimension = input_data.feature_dimension

            return intersection_table, output_model

        elif self.get_this_party_role() == constant.TaskRole.HOST:
            # sync intersection if turned on
            if self._parameter.sync_intersection:
                intersection_table = self._messenger.receive(tag=self._message.PLAIN_INTERSECTION,
                                                             parties=self.get_all_guest_names())[0]
                self._logger.info("received intersection id table: {}".format(intersection_table.count()))
            else:
                return None, self.instance_to_model()
            # construct output model
            output_model = self.instance_to_model()
            # output id only if turned on
            if not self._parameter.output_id_only:
                intersection_table = intersection_table.join_reserve_right(input_data)
                intersection_table.schema = input_data.schema
                intersection_table.feature_dimension = input_data.feature_dimension
            return intersection_table, output_model
