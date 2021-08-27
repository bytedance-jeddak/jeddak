from common.factory.factory import Factory
from common.util import constant
from privacy.crypto.plain.plain import Plain
from privacy.crypto.symmetric.diffie_hellman import DiffieHellman
from privacy.crypto.asymmetric.cpaillier import CPaillier


class EncryptorFactory(Factory):
    @staticmethod
    def get_instance(task_type, encrypter, key_size):
        """

        :param task_type: str
        :param encrypter: str
        :param key_size: int
        :return:
        """
        if encrypter == constant.Encryptor.PLAIN:
            if task_type == constant.TaskType.NEURAL_NETWORK:
                return Plain.generate()

            else:
                Factory._raise_value_error('task_type', task_type)

        if encrypter == constant.Encryptor.DIFFIE_HELLMAN:
            if task_type in (constant.TaskType.ALIGNER,):
                return DiffieHellman.generate(key_size)
            else:
                Factory._raise_value_error('task_type', task_type)

        elif encrypter == constant.Encryptor.CPAILLIER:
            if task_type in (constant.TaskType.LINEAR_REGRESSION,
                             constant.TaskType.LOGISTIC_REGRESSION,
                             constant.TaskType.POISSON_REGRESSION,
                             constant.TaskType.DPGBDT):
                return CPaillier.generate(key_size)
            else:
                Factory._raise_value_error('task_type', task_type)

        else:
            Factory._raise_value_error('encryptor', encrypter)
