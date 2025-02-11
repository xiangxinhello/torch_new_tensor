# Owner(s): ["module: cpp"]

import os
import unittest

import psutil
# import pdb
# pdb.set_trace()

import pytorch_openreg

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

tensor = torch.Tensor(1,2).to('new_one')
print(tensor)
