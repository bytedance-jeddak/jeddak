class Conversion(object):
    @staticmethod
    def str2int(s):
        """
        Convert an arbitrary string to int
        :param s:
        :return:
        """
        return int.from_bytes(s.encode('utf-8'), byteorder='big')

    @staticmethod
    def int2str(i):
        return i.to_bytes((i.bit_length() + 7) // 8, byteorder='big').decode('utf-8')

    @staticmethod
    def hex2dec(h):
        """

        :param h: a hex number in string
        :return:
        """
        return int(h, 16)

    @staticmethod
    def dec2hex(d):
        """

        :param d: int
        :return: str, with '0x' excluded
        """
        return hex(d)[2:]
