import random
import sys
import math
import numpy as np


class EncodedNumber(object):
    """Represents a float or int encoded.

    For end users, this class is mainly useful for specifying precision
    when adding/multiplying an :class:`EncryptedNumber` by a scalar.

    If you want to manually encode a number, then use :meth:`encode`;
    if de-serializing then use :meth:`__init__`.


    .. note::
        If working with other libraries you will have to agree on
        a specific :attr:`BASE` and :attr:`LOG2_BASE` - inheriting from this
        class and overriding those two attributes will enable this.

    Notes:
      This class supports encryption that only defined for non-negative integers less 
        than :attr:`n`.
      Since we frequently want to use signed integers and/or floating point numbers,
      values should be encoded as a valid integer before encryption.

      The operations of addition and multiplication must be preserved under this encoding.
      Namely:
      1. Decode(Encode(a) + Encode(b)) = a + b
      2. Decode(Encode(a) * Encode(b)) = a * b
      for any real numbers a and b.

      Representing signed integers is achieved as follows:
      we choose to represent only integers between +/-:attr:`max_int`, 
        where `max_int` is approximately :attr:`n`/3.
      The range of values between `max_int` and  `n` - `max_int` is reserved for detecting overflows.
      This encoding scheme supports properties #1 and #2 above.

      Representing floating point numbers is achieved as follows:
        Here we use a variant of fixed-precision arithmetic.
      In fixed precision, you encode by multiplying every float
        by a large number (e.g. 1e6) and rounding the resulting product.
      You decode by dividing by that number.
      However, this encoding scheme does not satisfy property #2 above:
        upon every multiplication, you must divide by the large number,
      which is not supported by some encryption (e.g., Paillier) withtout decryption.

      In our scheme, the "large number" is allowed to vary, and we keep track of it. It is:

        :attr:`BASE` ** :attr:`exponent`

      One number has many possible encodings; this property can be used to mitigate
        the leak of information due to the fact that :attr:`exponent` is never encrypted.

      For more details, see :meth:`encode`.


    Args:
      n (int): modulus for which to encode
      encoding (int): The encoded number to store.
        Must be positive and less than :attr:`max_int = n/3`.
      exponent (int): Together with :attr:`BASE`,
        determines the level of fixed-precision used in encoding the number.

    Attributes:
      n (int): modulus for which to encode
      max_int (int): max_int = n/3
      encoding (int): The encoded number to store.
        Must be positive and less than :attr:`max_int = n/3`.
      exponent (int): Together with :attr:`BASE`,
        determines the level of fixed-precision used in encoding the number.
    """
    BASE = 16
    """Base to use when exponentiating. Larger `BASE` means that :attr:`exponent` leaks less information.
    If you vary this, you'll have to manually inform anyone decoding your numbers.
    """
    LOG2_BASE = math.log(BASE, 2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig

    def __init__(self, n, encoding, exponent):
        self.n = n
        self.max_int = n // 3 - 1
        self.encoding = encoding
        self.exponent = exponent

    @classmethod
    def encode(cls, n, scalar, precision=None, max_exponent=None):
        """Return an encoding of an int or float.

        If *scalar* is a float, first approximate it as an int, `int_rep`:

            scalar = int_rep * (:attr:`BASE` ** :attr:`exponent`),
        
        for some (typically negative) integer exponent, which can be tuned using *precision* and *max_exponent*.
        Specifically, :attr:`exponent` is chosen to be equal to or less than *max_exponent*, 
        and such that the number *precision* is not rounded to zero.

        Having found an integer representation for the float (or having been given an int `scalar`),
         we then represent this integer as a non-negative integer < :attr:`max`.

        We take the convention that
            acnumber 0 < x < n/3 is positive,
            and that a number -n / 3 < x < 0 is negative.
            the range x > n/3 or x < -n / 3  overflow detection.

        Args:
          n (int): specify the range of encoded data.
          scalar: an int or float to be encoded.
            If int, it must satisfy abs(*value*) < : :attr:`n`/3.
            If float, it must satisfy abs(*value* / *precision*) << :attr:`n`/3
                (i.e. if a float is near the limit then detectable overflow may still occur)
          precision (float): Choose exponent (i.e. fix the precision) so that this number is distinguishable from zero.
            If `scalar` is a float, then this is set so that minimal precision is lost.
                Lower precision leads to smaller encodings, which might yield faster computation.
          max_exponent (int): Ensure that the exponent of the returned `EncodedNumber` is at most this.

        Returns:
          EncodedNumber: Encoded form of *scalar*, ready for operation (e.g., encryption) with *modulus*.
        """
        if np.abs(scalar) < 1e-200:
          scalar = 0
        # Calculate the maximum exponent for desired precision
        if precision is None:
            if isinstance(scalar, int):
                prec_exponent = 0
            elif isinstance(scalar, float):
                # Encode with *at least* as much precision as the python float
                # What's the base-2 exponent on the float?
                bin_flt_exponent = math.frexp(scalar)[1]

                # What's the base-2 exponent of the least significant bit?
                # The least significant bit has value 2 ** bin_lsb_exponent
                bin_lsb_exponent = bin_flt_exponent - cls.FLOAT_MANTISSA_BITS

                # What's the corresponding base BASE exponent? Round that down.
                prec_exponent = math.floor(bin_lsb_exponent / cls.LOG2_BASE)
            else:
                raise TypeError("Don't know the precision of type %s."
                                % type(scalar))
        else:
            prec_exponent = math.floor(math.log(precision, cls.BASE))

        # Remember exponents are negative for numbers < 1.
        # If we're going to store numbers with a more negative
        # exponent than demanded by the precision, then we may
        # as well bump up the actual precision.
        if max_exponent is None:
            exponent = prec_exponent
        else:
            exponent = min(max_exponent, prec_exponent)

        int_rep = int(round(scalar * pow(cls.BASE, -exponent)))

        if abs(int_rep) > (n // 3 - 1):
            raise ValueError('Integer needs to be within +/- %d but got %d'
                             % (n, int_rep))

        return cls(n, int_rep, exponent)

    def decode(self):
        """Decode plaintext and return the result.

        Returns:
          an int or float: the decoded number. N.B. if the number
            returned is an integer, it will not be of type float.

        Raises:
          OverflowError: if overflow is detected in the decrypted number.
        """
        if abs(self.encoding) >= self.max_int:
          raise OverflowError('Overflow detected in decrypted number')
        return self.encoding * pow(self.BASE, self.exponent)