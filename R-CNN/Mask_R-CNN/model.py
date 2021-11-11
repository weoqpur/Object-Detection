import datetime
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# 로그 출력
def log(text, array=None):
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5}".format(str(array.shape),
                                                                    array.min() if array.size else "",
                                                                    array.min() if array.size else ""))
        print(text)

# 작업 척도 출력
def printProgressBer (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """

    :param iteration  - 필수   : 현재 반복 (int)
    :param total      - 필수   : 총 반복 (int)
    :param prefix     - 선택   : 접두사 문자열 (str)
    :param suffix     - 선택적 : 접미사 문자열 (str)
    :param decimals   - 선택적 : 완료율의 소수점 양수 (int)
    :param length     - 선택적 : 막대의 문자 길이 (Int)
    :param fill       - 선택적 : 박대에 채울 문자 (str)
    :return:
    """

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end= '\n')
    # 완료시 개행
    if iteration == total:
        print()


def unique1d(tensor):
    if tensor.size() [0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor [:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cpu()
    unique_bool = torch.cat((first_element, unique_bool), dim=0)
    return tensor[unique_bool.data]

def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]

