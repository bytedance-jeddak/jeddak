import random


class Random(object):
    GREAT_NUMBER = 1e16

    @staticmethod
    def generate_random_digits(length=5):
        """

        :param length:
        :return: str
        """
        random_string = ''
        for _ in range(length):
            random_string += str(random.randint(0, 9))
        return random_string

    @staticmethod
    def generate_uniform_number(lower_bound=None, upper_bound=None):
        """
        Generate a real number uniformly at random
        :param lower_bound:
        :param upper_bound:
        :return:
        """
        if lower_bound is None:
            assert upper_bound is None
            lower_bound = -Random.GREAT_NUMBER
            upper_bound = Random.GREAT_NUMBER

        real_number = random.uniform(lower_bound, upper_bound)

        return real_number
