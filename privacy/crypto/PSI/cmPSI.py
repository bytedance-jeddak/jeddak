import numpy as np
import os
from ctypes import *
import math
import operator as op
from functools import reduce
import hashlib
u32Mx = 0xFFFFFFFF
u64Mx = 0xFFFFFFFFFFFFFFFF

_file = '_PSI.so'
_path = os.path.join(*(os.path.split(__file__)[:-1] + (_file,)))
_PSI = cdll.LoadLibrary(_path)

class PARAMETERS(Structure):
    _fields_ = [
        ("width", c_ulong),
        ("logHeight", c_ulong),
        ("senderSize", c_ulong),
        ("receiverSize", c_ulong),
        ("h1LengthInBytes", c_ulong),
        ("hashLengthInBytes", c_ulong),
    ]

class DATA(Structure):
    _fields_ = [
        ("d", c_uint8*16)
    ]

def __ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

def hashLenComp(n1, n2):
    sigma = 40
    return max(math.ceil((sigma + math.log(n1*n2, 2))/8), 8)

def binNumComp(n1, n2):
    threshold = pow(2, 22)
    if n1 <= threshold and n2 <=threshold:
        return 0
    return math.ceil(math.log(max(n1, n2), 2)) - 22

def logHeightComp(n1, n2):
    return max(math.ceil(math.log(n2, 2)), 2)

def widthComp(n1, n2):
    d = 128
    negl = pow(2, -40)
    logHeight = logHeightComp(n1, n2)
    height = 1 << logHeight
    p = pow((1 - 1 / height), n2)
    w = 256
    wMax = 4096
    while True:
        if w>=wMax:
            return wMax
        res = 0
        for i in range((min(d, w))):
            res += __ncr(w, i) * pow(p, i) * pow(1 - p, w - i)
        res *= n1
        if res < negl:
            w = int(w/2)
            break
        else:
            w *= 2
    while True:
        res = 0
        for i in range((min(d, w))):
            res += __ncr(w, i) * pow(p, i) * pow(1 - p, w - i)
        res *= n1
        if res < negl:
            break
        else:
            w += 16
    return w

def convertBits(data, lengthInBits):
    upper = 1 << lengthInBits
    res = 0
    while data:
        res ^= (data & (upper-1))
        data = data >> lengthInBits
    return res

def shortHash(data, lengthInBits):
    hashLenInBytes = math.ceil(lengthInBits/8)
    data = convertBits(data, 64)
    c_hashInput = c_uint64(data)
    c_hashOutput = (c_uint8 * hashLenInBytes)()
    _PSI.shortHash(c_hashInput, c_hashOutput, hashLenInBytes)
    h = 0
    for i in range(int(lengthInBits/8)):
        h = (h << 8) | int(c_hashOutput[i])
    remain = lengthInBits & 7
    if remain != 0:
        h = (h << remain) | (int(c_hashOutput[int(lengthInBits/8)]) & ((1 << remain) - 1))
    return h

def filterSizeInBytes(n):
    if n == 0:
        return 1
    return math.ceil((1/(1-pow(0.5, 1/n)))/8)

class CMPsiSender:
    def __init__(self, pp: PARAMETERS):
        self.pp = pp

    def evaluateLocationElement(self, seed, data):
        locationInBytes = int((self.pp.logHeight + 7) / 8)
        c_Data = DATA()
        p = pointer(c_Data)
        _PSI.setData(p, data)
        c_LocationElement = ((c_uint8 * locationInBytes) * self.pp.width)()
        c_seed = c_ulong(seed)
        _PSI.senderEvaluateLocationElement(c_seed, self.pp, c_Data, c_LocationElement)
        pyLocationElement = np.zeros([self.pp.width, locationInBytes], dtype=np.uint8)
        pyLocationElement[:] = c_LocationElement[:]
        return pyLocationElement

    def senderEvalMatrxCElement(self, otMessage, choice, matrixDelta_element):
        height = 1 << self.pp.logHeight
        heightInBytes = int((height + 7) / 8)
        c_otMessage = DATA()
        p = pointer(c_otMessage)
        _PSI.setData(p, int(otMessage & u64Mx))
        c_matrixDelta_element = (c_uint8 * heightInBytes)(* list(matrixDelta_element))
        c_matrixC_element = (c_uint8 * heightInBytes)()
        choice = c_uint8(choice)
        _PSI.senderEvalMatrxCElement(self.pp, c_otMessage, choice, c_matrixDelta_element, c_matrixC_element)
        return np.array(c_matrixC_element[:], dtype=np.uint8)

    def senderOutputElement(self, pyLocationElement, matrixC):
        locationInBytes = int((self.pp.logHeight + 7) / 8)
        height = 1 << self.pp.logHeight
        heightInBytes = int((height + 7) / 8)
        c_LocationElement = ((c_uint8 * locationInBytes) * self.pp.width)()
        c_matrixC = (POINTER(c_uint8) * self.pp.width)()
        _PSI.mallocArray(c_matrixC, self.pp.width, heightInBytes)
        c_hashOutput = (c_uint8 * self.pp.hashLengthInBytes)()
        for i in range(self.pp.width):
            c_LocationElement[i] = (c_uint8 * locationInBytes) (* list(pyLocationElement[i]))
            c_row = (c_uint8 * heightInBytes) (* list(matrixC[i]))
            _PSI.set2DArrayRow(c_matrixC, i, c_row, heightInBytes)
        _PSI.senderOutputElement(self.pp, c_LocationElement, c_matrixC, c_hashOutput)
        hashOutput = 0
        for i in range(self.pp.hashLengthInBytes):
            hashOutput += (hashOutput << 8) + c_hashOutput[i]
        _PSI.freeArray(c_matrixC, self.pp.width)
        return hashOutput

    def senderOutputs(self, pyLocations, matrixC):
        locationInBytes = int((self.pp.logHeight + 7) / 8)
        height = 1 << self.pp.logHeight
        heightInBytes = int((height + 7) / 8)
        c_Locations = (POINTER(c_uint8) * self.pp.width)()
        c_matrixC = (POINTER(c_uint8) * self.pp.width)()
        c_hashOutputs = (POINTER(c_uint8) * self.pp.senderSize)()
        _PSI.mallocArray(c_Locations, self.pp.width, locationInBytes * self.pp.senderSize+4)
        _PSI.mallocArray(c_matrixC, self.pp.width, heightInBytes)
        _PSI.mallocArray(c_hashOutputs, self.pp.senderSize, self.pp.hashLengthInBytes)
        for i in range(self.pp.width):
            c_row = (c_uint8 * (locationInBytes * self.pp.senderSize)) (* list(pyLocations[i]))
            _PSI.set2DArrayRow(c_Locations, i, c_row, locationInBytes * self.pp.senderSize)
            c_row = (c_uint8 * heightInBytes) (* list(matrixC[i]))
            _PSI.set2DArrayRow(c_matrixC, i, c_row, heightInBytes)
        _PSI.senderOutputs(self.pp, c_Locations, c_matrixC, c_hashOutputs)
        hashOutputs = []
        c_output =(c_uint8 * self.pp.hashLengthInBytes)()
        for i in range(self.pp.senderSize):
            hashOutput = 0
            _PSI.get2DArrayRow(c_output, self.pp.hashLengthInBytes, c_hashOutputs, i)
            for j in range(self.pp.hashLengthInBytes):
                hashOutput += (hashOutput << 8) + c_output[j]
            hashOutputs.append(hashOutput)
        _PSI.freeArray(c_Locations, self.pp.width)
        _PSI.freeArray(c_matrixC, self.pp.width)
        _PSI.freeArray(c_hashOutputs, self.pp.senderSize)
        return hashOutputs


class CMPsiReceiver:
    def __init__(self, pp: PARAMETERS):
        self.pp = pp

    def evaluateLocationElement(self, seed, data):
        locationInBytes = int((self.pp.logHeight + 7) / 8)
        c_Data = DATA()
        p = pointer(c_Data)
        _PSI.setData(p, c_ulonglong(data))
        c_LocationElement = ((c_uint8 * locationInBytes) * self.pp.width)()
        c_seed = c_ulong(seed)
        _PSI.receiverEvaluateLocationElement(c_seed, self.pp, c_Data, c_LocationElement)
        pyLocationElement = np.zeros([self.pp.width, locationInBytes], dtype=np.uint8)
        pyLocationElement[:] = c_LocationElement[:]
        return pyLocationElement

    def receiverRowLoc(self, locationElement_bucket, widthBucket):
        shift = (1 << self.pp.logHeight) - 1
        locationInBytes = int((self.pp.logHeight + 7) / 8)
        locationBucketList = []
        for i in range(widthBucket):
            location = 0
            for j in range(locationInBytes):
                location |= (locationElement_bucket[i][j] << (8 * j))
            locationBucketList.append(location & shift)
        return locationBucketList

    def receiverEvalMatrixDRow(self, locationList, widthBucket):
        height = 1 << self.pp.logHeight
        heightInBytes = int((height + 7) / 8)
        matrixD_rows = np.array([[255 for i in range(heightInBytes)] for _ in range(widthBucket)], np.uint8)
        for i in range(widthBucket):
            for location in locationList:
                mask = c_uint8(~(1 << (location[i] & 7)))
                matrixD_rows[i][location[i]>>3] &= mask.value
        return matrixD_rows

    def receiverEvalMatrxD(self, locationElement_list):
        locationInBytes = int((self.pp.logHeight + 7) / 8)
        height = 1 << self.pp.logHeight
        heightInBytes = int((height + 7) / 8)
        c_transLocations = (POINTER(c_uint8) * self.pp.width)()
        _PSI.mallocArray(c_transLocations, self.pp.width, self.pp.receiverSize * locationInBytes + 4)
        c_matrixD = (POINTER(c_uint8) * self.pp.width)()
        _PSI.mallocArray(c_matrixD, self.pp.width, heightInBytes)
        c_locationRow = (c_uint8 * (self.pp.receiverSize * locationInBytes))()
        c_matrixRow = (c_uint8 * heightInBytes)()
        locationArray = np.hstack(locationElement_list)
        for i in range(self.pp.width):
            c_locationRow[:] = locationArray[i][:]
            _PSI.set2DArrayRow(c_transLocations, i, c_locationRow, self.pp.receiverSize * locationInBytes)
        _PSI.receiverEvalMatrxD(self.pp, c_transLocations, c_matrixD)
        matrixD = np.zeros([self.pp.width, heightInBytes], dtype=np.uint8)
        for i in range(self.pp.width):
            _PSI.get2DArrayRow(c_matrixRow, heightInBytes, c_matrixD, i)
            matrixD[i][:] = c_matrixRow[:]
        _PSI.freeArray(c_matrixD, self.pp.width)
        _PSI.freeArray(c_transLocations, self.pp.width)
        return matrixD

    def receiverEvalDeltaElement(self, otMessagesElement, matrixD_element):
        c_otMessages1 = DATA()
        p = pointer(c_otMessages1)
        _PSI.setData(p, int(int(otMessagesElement[0]) & u64Mx))
        c_otMessages2 = DATA()
        p = pointer(c_otMessages2)
        _PSI.setData(p, int(int(otMessagesElement[1]) & u64Mx))
        height = 1 << self.pp.logHeight
        heightInBytes = int((height + 7) / 8)
        c_matrixD_element = (c_uint8 * heightInBytes)(* list(matrixD_element))
        c_matrixA_element = (c_uint8 * heightInBytes)()
        c_matrixDelta_element = (c_uint8 * heightInBytes)()

        _PSI.receiverEvalDeltaElement(c_otMessages1, c_otMessages2, self.pp, c_matrixD_element,
                                               c_matrixA_element, c_matrixDelta_element)
        matrixA_element = np.array([c_matrixA_element[i] for i in range(heightInBytes)], dtype=np.uint8)
        matrixDelta_element = np.array([c_matrixDelta_element[i] for i in range(heightInBytes)], dtype=np.uint8)
        return matrixA_element, matrixDelta_element

    def receiverHashBit(self, locationElement, matrixA, widthBucket):
        locationInBytes = int((self.pp.logHeight + 7) / 8)
        widthBucketInBytes = int((widthBucket + 7) / 8)
        height = 1 << self.pp.logHeight
        heightInBytes = int((height + 7) / 8)
        bitsArray = np.zeros(widthBucketInBytes, dtype=np.uint8)
        for i in range(widthBucket):
            c_LocationElement_row = (c_uint8 * locationInBytes)(* list(locationElement[i]))
            c_matrixA_row = (c_uint8 * heightInBytes) (* list(matrixA[i]))
            bit = _PSI.receiverGetHashInputBit(self.pp, c_LocationElement_row, c_matrixA_row)
            bitsArray[int(i/8)] |= (bit<<(i%8))
        return bitsArray

    def receiverHashOutput(self, hashInputWidth):
        c_hashInputWidth = (c_uint8 * self.pp.width)(* list(hashInputWidth))
        c_hashOutput = (c_uint8 * self.pp.hashLengthInBytes)()
        _PSI.receiverHashOutput(self.pp, c_hashInputWidth, c_hashOutput)
        hashOutput = 0
        for i in range(self.pp.hashLengthInBytes):
            hashOutput += (hashOutput << 8) + c_hashOutput[i]
        return hashOutput

    def receiverOutputElement(self, locationElement, matrixA):
        locationInBytes = int((self.pp.logHeight + 7) / 8)
        height = 1 << self.pp.logHeight
        heightInBytes = int((height + 7) / 8)

        c_LocationElement = ((c_uint8 * locationInBytes) * self.pp.width) ()
        c_matrixA = (POINTER(c_uint8) * self.pp.width)()
        _PSI.mallocArray(c_matrixA, self.pp.width, heightInBytes)
        c_hashOutput = (c_uint8 * self.pp.hashLengthInBytes)()
        for i in range(self.pp.width):
            c_LocationElement[i] = (c_uint8 * locationInBytes) (* list(locationElement[i]))
            c_row = (c_uint8 * heightInBytes) (* list(matrixA[i]))
            _PSI.set2DArrayRow(c_matrixA, i, c_row, heightInBytes)
        _PSI.receiverOutputElement(self.pp, c_LocationElement, c_matrixA, c_hashOutput)
        hashOutput = 0
        for i in range(self.pp.hashLengthInBytes):
            hashOutput += (hashOutput << 8) + c_hashOutput[i]
        _PSI.freeArray(c_matrixA, self.pp.width)
        return hashOutput

    def receiverOutputs(self, pyLocations, matrixA):
        locationInBytes = int((self.pp.logHeight + 7) / 8)
        height = 1 << self.pp.logHeight
        heightInBytes = int((height + 7) / 8)
        c_Locations = (POINTER(c_uint8) * self.pp.width)()
        c_matrixA = (POINTER(c_uint8) * self.pp.width)()
        c_hashOutputs = (POINTER(c_uint8) * self.pp.receiverSize)()
        _PSI.mallocArray(c_Locations, self.pp.width, locationInBytes * self.pp.receiverSize)
        _PSI.mallocArray(c_matrixA, self.pp.width, heightInBytes)
        _PSI.mallocArray(c_hashOutputs, self.pp.receiverSize, self.pp.hashLengthInBytes)
        for i in range(self.pp.width):
            c_row = (c_uint8 * heightInBytes) (* list(pyLocations[i]))
            _PSI.set2DArrayRow(c_Locations, i, c_row, locationInBytes * self.pp.receiverSize)
            c_row = (c_uint8 * heightInBytes) (* list(matrixA[i]))
            _PSI.set2DArrayRow(c_matrixA, i, c_row, heightInBytes)
        _PSI.receiverOutputs(self.pp, c_Locations, c_matrixA, c_hashOutputs)
        hashOutputs = []
        c_output =(c_uint8 * self.pp.hashLengthInBytes)()
        for i in range(self.pp.receiverSize):
            hashOutput = 0
            _PSI.get2DArrayRow(c_output, self.pp.hashLengthInBytes, c_hashOutputs, i)
            for j in range(self.pp.hashLengthInBytes):
                hashOutput += (hashOutput << 8) + c_output[j]
            hashOutputs.append(hashOutput)
        _PSI.freeArray(c_Locations, self.pp.width)
        _PSI.freeArray(c_matrixA, self.pp.width)
        _PSI.freeArray(c_hashOutputs, self.pp.receiverSize)
        return hashOutputs



